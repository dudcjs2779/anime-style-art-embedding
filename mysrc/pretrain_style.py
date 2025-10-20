import json
import re
import toml
import logging
import os
import shutil
from datetime import datetime
import time
import math
import torch
import numpy as np
import random
import copy
import wandb
from PIL import Image

import torch.utils.tensorboard as tensorboard

from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging

from mysrc.utils.pretrain_params import setup_parser
from mysrc.utils.config import read_config
from mysrc.utils.dataset import make_style_dataset
from mysrc.utils.optim_utils import make_clip_param_groups
from mysrc.utils.loss import SigLipLoss2, CosineSimilarityLoss, StyleSigLip
from mysrc.utils.scheduler import CosineScheduler
from mysrc.utils.precision import get_autocast, get_input_dtype
from mysrc.clip_model import create_clip
from mysrc.utils.file_utils import pt_load
from mysrc.utils.metrics import get_embed_metrics, get_class_clip_metrics, get_clip_metrics, get_style_clip_metrics
from mysrc.utils.data import tensor_image_save

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)

def load_random_state(checkpoint):
    torch.random.set_rng_state(checkpoint['torch_rng_state'])
    if torch.cuda.is_available() and checkpoint['cuda_rng_state'] is not None: 
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    np.random.set_state(checkpoint['numpy_rng_state'])
    random.setstate(checkpoint['random_state'])

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

def apply_optim_scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        lr_multiplier = param_group["lr_multiplier"]

        # if isinstance(optimizer, Prodigy):
            # return
        param_group["lr"] =  lr * lr_multiplier

def train_one_epoch(model, device, dataloader, clip_loss, style_loss, epoch, optimizer, scaler, scheduler, tb_writer, args):
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    model.train()
    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_classes, accum_times = [], [], [], []
        accum_tags, accum_keep_tags = [], []
        accum_features = {}

    losses_m = {}
    losses_log = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        lr = scheduler[step]
        apply_optim_scheduler(optimizer, lr)

        images, texts, times = batch['image'], batch['text'], batch['time'], 
        classes, artists = batch['label'], batch['artist']
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, dtype=input_dtype, non_blocking=True)
        times = times.to(device=device, dtype=input_dtype, non_blocking=True)
        classes = classes.to(device=device, dtype=input_dtype, non_blocking=True)

        tags, keep_tags = batch['tags'], batch['keep_tags']
        styled_label_token, content_label_token = batch['styled_label_token'], batch['content_label_token']
        tags = tags.to(device=device, dtype=input_dtype, non_blocking=True)
        keep_tags = keep_tags.to(device=device, dtype=input_dtype, non_blocking=True)
        styled_label_token = styled_label_token.to(device=device, dtype=input_dtype, non_blocking=True)

        # temp = torch.chunk(images, 2, dim=0)  # tensor_image_save(temp)

        bsz = len(images) // 2
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        if args.accum_freq == 1:
            with autocast():
                model_out = model(images, texts)
                feat_dict = {
                    'features1': model_out['image_features'],
                    'image_sty_feat': model_out['image_sty_feat'],
                    'features2': model_out['text_features'],
                    'text_sty_feat': model_out['text_sty_feat'],
                }                
                logit_scale = model_out["logit_scale"]
                logit_bias = model_out.get("logit_bias", None)
                style_logit_scale = model_out["style_logit_scale"]
                style_logit_bias = model_out.get("style_logit_bias", None)

                losses = {}
                losses['clip_loss'] = clip_loss(
                    features1=feat_dict['features1'], features2=feat_dict['features2'],
                    logit_scale=logit_scale, logit_bias=logit_bias,
                    full_tags=tags, sel_tags=keep_tags,
                    classes=classes, same_class_weight=args.same_class_weight,
                    output_dict=False
                )
                losses['image_style_loss'] = style_loss(
                    features1=feat_dict['image_sty_feat'], features2=feat_dict['image_sty_feat'],
                    logit_scale=style_logit_scale, logit_bias=style_logit_bias,
                    classes=classes[:bsz], same_class_weight=args.same_class_weight,
                    times=times[:bsz], time_range=args.time_range, time_weight=args.time_weight,
                    output_dict=False,
                ) * 0.5

                losses['text_style_loss'] = style_loss(
                    features1=feat_dict['text_sty_feat'], features2=feat_dict['text_sty_feat'],
                    logit_scale=style_logit_scale, logit_bias=style_logit_bias,
                    classes=classes, same_class_weight=args.same_class_weight,
                    times=times, time_range=args.time_range, time_weight=args.time_weight,
                    output_dict=False,
                ) * 0.5
                
                losses['style_loss'] = style_loss(
                    features1=feat_dict['image_sty_feat'], features2=feat_dict['text_sty_feat'][:bsz],
                    logit_scale=style_logit_scale, logit_bias=style_logit_bias,
                    classes=classes[:bsz], same_class_weight=args.same_class_weight,
                    times=times[:bsz], time_range=args.time_range, time_weight=args.time_weight,
                    output_dict=False,
                ) * 0.1

                # losses["style_contrastive_loss"] = clip_loss(
                #     features1=feat_dict['style_features'], 
                #     features2=feat_dict['style_features'],
                #     logit_scale=logit_scale, logit_bias=logit_bias,
                #     classes=classes, same_class_weight=args.same_class_weight, 
                #     times=times, time_range=args.time_range, time_weight=args.time_weight,
                #     output_dict=False
                # )
                # losses['proj_style_loss'] = cos_sim_loss(feat_dict['style_features'], feat_dict['proj_style_features'])
                
                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)

            # accum_freq 때문에 mae_lossr의 기록이 왜곡되는 것 방지
            losses_log.update(losses)
            del losses_log["loss"]

            # if args.proj_style_loss:
            #     losses_log['proj_style_loss'] = losses_log['proj_style_loss'] * args.accum_freq

            losses_log['loss'] = sum(losses_log.values())
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, texts)
                    feat_dict = {
                        'features1': model_out['image_features'],
                        'image_sty_feat': model_out['image_sty_feat'],
                        'features2': model_out['text_features'],
                        'text_sty_feat': model_out['text_sty_feat'],
                    }        

                    for key, val in feat_dict.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]
                    
                accum_images.append(images)
                accum_texts.append(texts)
                accum_classes.append(classes)
                accum_times.append(times)
                accum_tags.append(tags)
                accum_keep_tags.append(keep_tags)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            accum_classes = torch.concat(accum_classes, dim=0)
            accum_times = torch.concat(accum_times, dim=0)
            accum_tags = torch.concat(accum_tags, dim=0)
            accum_keep_tags = torch.concat(accum_keep_tags, dim=0)
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast():
                    model_out = model(images, texts)
                    feat_dict = {
                        'features1': model_out['image_features'],
                        'image_sty_feat': model_out['image_sty_feat'],
                        'features2': model_out['text_features'],
                        'text_sty_feat': model_out['text_sty_feat'],
                    }         

                    logit_scale = model_out["logit_scale"]
                    logit_bias = model_out.get("logit_bias", None)
                    style_logit_scale = model_out["style_logit_scale"]
                    style_logit_bias = model_out.get("style_logit_bias", None)

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [feat_dict[key]] + accumulated[j + 1:])

                    losses = {}
                    # Image & Tag Contrastive Loss(SigLIP)
                    losses['clip_loss'] = clip_loss(
                        features1=inputs['features1'], features2=inputs['features2'],
                        logit_scale=logit_scale, logit_bias=logit_bias,
                        full_tags=accum_tags, sel_tags=accum_keep_tags,
                        classes=accum_classes, same_class_weight=args.same_class_weight,
                        output_dict=False
                    )

                    losses['image_style_loss'] = style_loss(
                        features1=inputs['image_sty_feat'], features2=inputs['image_sty_feat'],
                        logit_scale=style_logit_scale, logit_bias=style_logit_bias,
                        classes=accum_classes, same_class_weight=args.same_class_weight,
                        times=accum_times, time_range=args.time_range, time_weight=args.time_weight,
                        output_dict=False,
                    ) * 0.5

                    losses['text_style_loss'] = style_loss(
                        features1=inputs['text_sty_feat'], features2=inputs['text_sty_feat'],
                        logit_scale=style_logit_scale, logit_bias=style_logit_bias,
                        classes=accum_classes, same_class_weight=args.same_class_weight,
                        times=accum_times, time_range=args.time_range, time_weight=args.time_weight,
                        output_dict=False,
                    ) * 0.5

                    losses['style_loss'] = style_loss(
                        features1=inputs['image_sty_feat'], features2=inputs['text_sty_feat'],
                        logit_scale=style_logit_scale, logit_bias=style_logit_bias,
                        classes=accum_classes, same_class_weight=args.same_class_weight,
                        times=accum_times, time_range=args.time_range, time_weight=args.time_weight,
                        output_dict=False,
                    ) * 0.1

                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)

                # accum_freq 때문에 mae_lossr의 기록이 왜곡되는 것 방지
                losses_log.update(losses)
                del losses_log["loss"]

                # if args.proj_style_loss:
                #     losses_log['proj_style_loss'] = losses_log['proj_style_loss'] * args.accum_freq

                losses_log['loss'] = sum(losses_log.values())

        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()


        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_classes, accum_times = [], [], [], []
            accum_tags, accum_keep_tags = [], []
            accum_features = {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))
            unwrap_model(model).style_logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images) // 2
            num_samples = batch_count * batch_size * args.accum_freq * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            for key, val in losses_log.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)

            logit_scale_scalar = logit_scale.item()
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f} " + loss_log
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_second": samples_per_second,
                "samples_per_second_per_gpu": samples_per_second_per_gpu,
                "scale": logit_scale_scalar,
                "bias": logit_bias.item(),
                "style_scale": style_logit_scale.item(),
                "style_bias": style_logit_bias.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }            
            log_data.update({name:val.val for name, val in losses_m.items()})
            log_data.update({f"{name}_avg":val.avg for name, val in losses_m.items()})

            log_data = {"train/" + name: val for name, val in log_data.items()}

            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step  # for backwards compatibility
                args.step = step
                wandb.log(log_data, step=step)
            
            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, device, dataloader, clip_loss, style_loss, epoch, tb_writer, step, args):
    metrics = {}
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    model.eval()

    if args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs):
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # cumulative_loss = {}
        losses_m = {}
        all_image_features, all_text_features, all_styled_features, all_style_features, all_classes, style_classes = [], [], [], [], [], []
        all_im_sty_feats, all_txt_sty_feats = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts, times = batch['image'], batch['text'], batch['time']
                classes, artists = batch['label'], batch['artist']
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, dtype=input_dtype, non_blocking=True)
                classes = classes.to(device=device, dtype=input_dtype, non_blocking=True)
                times = times.to(device=device, dtype=input_dtype, non_blocking=True)

                tags, keep_tags = batch['tags'], batch['keep_tags']
                styled_label_token, content_label_token = batch['styled_label_token'], batch['content_label_token']
                tags = tags.to(device=device, dtype=input_dtype, non_blocking=True)
                keep_tags = keep_tags.to(device=device, dtype=input_dtype, non_blocking=True)
                styled_label_token = styled_label_token.to(device=device, dtype=input_dtype, non_blocking=True)
                # content_label_token = content_label_token.to(device=device, dtype=input_dtype, non_blocking=True)

                bsz = len(images) // 2
                # images = images[:bsz]
                # texts = styled_label_token
                # tags = tags[:bsz]
                # keep_tags = keep_tags[:bsz]
                # classes = classes[:bsz]

                with autocast():
                    model_out = model(images, texts)
                    # style_features, _ = model.encode_text(styled_label_token, normalize=True)
                    feat_dict = {
                        'features1': model_out['image_features'],
                        'image_sty_feat': model_out['image_sty_feat'],
                        'features2': model_out['text_features'],
                        'text_sty_feat': model_out['text_sty_feat'],
                    }
                    logit_scale = model_out["logit_scale"]
                    logit_bias = model_out.get("logit_bias", None)
                    style_logit_scale = model_out["style_logit_scale"]
                    style_logit_bias = model_out.get("style_logit_bias", None)

                    styled_features = feat_dict['features1'][:bsz]
                    # style_token = feat_dict['style_token'][:bsz]
                    image_sty_feat = feat_dict['image_sty_feat']
                    text_sty_feat = feat_dict['text_sty_feat'][:bsz]

                    batch_size = images.shape[0]
                    # all_styled_features.append(styled_features.cpu())
                    # all_style_features.append(style_features.cpu())
                    all_im_sty_feats.append(image_sty_feat.cpu())
                    all_txt_sty_feats.append(text_sty_feat.cpu())
                    all_image_features.append(feat_dict['features1'].cpu())
                    all_text_features.append(feat_dict['features2'].cpu())
                    all_classes.append(classes.cpu())
                    style_classes.append(classes[:bsz].cpu())

                    losses = {}
                    losses['clip_loss'] = clip_loss(
                        features1=feat_dict['features1'], features2=feat_dict['features2'],
                        # image_features=feat_dict['features1'], text_features=feat_dict['features2'],
                        logit_scale=logit_scale, logit_bias=logit_bias, 
                        # s2c_tag_ratio=s2c_tag_ratio, c2s_tag_ratio=c2s_tag_ratio, batch_size=len(images),
                        full_tags=tags, sel_tags=keep_tags,
                        classes=classes,
                        output_dict=False
                    )

                    losses['image_style_loss'] = style_loss(
                        features1=feat_dict['image_sty_feat'], features2=feat_dict['image_sty_feat'],
                        logit_scale=style_logit_scale, logit_bias=style_logit_bias,
                        classes=classes[:bsz], same_class_weight=args.same_class_weight,
                        times=times[:bsz], time_range=args.time_range, time_weight=args.time_weight,
                        output_dict=False,
                    )

                    losses['text_style_loss'] = style_loss(
                        features1=feat_dict['text_sty_feat'], features2=feat_dict['text_sty_feat'],
                        logit_scale=style_logit_scale, logit_bias=style_logit_bias,
                        classes=classes, same_class_weight=args.same_class_weight,
                        times=times, time_range=args.time_range, time_weight=args.time_weight,
                        output_dict=False,
                    )
                    
                    losses['style_loss'] = style_loss(
                        features1=feat_dict['image_sty_feat'], features2=feat_dict['text_sty_feat'][:bsz],
                        logit_scale=style_logit_scale, logit_bias=style_logit_bias,
                        classes=classes[:bsz], same_class_weight=args.same_class_weight,
                        times=times[:bsz], time_range=args.time_range, time_weight=args.time_weight,
                        output_dict=False,
                    )

                for key, val in losses.items():
                    if key not in losses_m:
                        losses_m[key] = AverageMeter()
                    losses_m[key].update(val.item(), batch_size)

                loss_log = " ".join(
                    [
                        f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
                        for loss_name, loss_m in losses_m.items()
                    ]
                )

                # cumulative_loss += losses * batch_size
                num_samples += batch_size // 2

                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t" + loss_log
                        # f"Clip Loss: {losses_m['contrastive_loss'].avg:.6f}\t"
                        # f"Style Loss: {losses_m['style_contrastive_loss'].avg:.6f}\t"
                        # f"Proj Style Loss: {losses_m['proj_style_loss'].avg:.6f}\t"
                    )
            
            if logit_bias:
                logit_bias = logit_bias.cpu()

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
                classes=torch.cat(all_classes),
            )

            style_metrics = get_style_clip_metrics(
                image_style=torch.cat(all_im_sty_feats),
                text_style=torch.cat(all_txt_sty_feats),
                logit_scale=style_logit_scale.cpu(),
                classes=torch.cat(style_classes),
            )
            style_metrics = {f"style_{k}":v for k,v in style_metrics.items()}

            metrics.update({
                **val_metrics, 
                **style_metrics,
                "clip_loss": losses_m['clip_loss'].avg, 
                "style_loss": losses_m['style_loss'].avg,
                "text_style_loss": losses_m['text_style_loss'].avg, 
                "image_style_loss": losses_m['image_style_loss'].avg, 
                "epoch": epoch, "num_samples": num_samples
            })
            
            # if args.proj_style_loss:
            #     metrics.update({"val_proj_style_loss": losses_m['proj_style_loss'].avg})

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.6f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if tb_writer is not None:
        for name, val in log_data.items():
            tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics
       
def main(args):
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)

    t_now = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    args.logs = "style_log"
    args.output_path = os.path.join(args.logs, f"{t_now}_{args.model_name[:50]}")
    args.log_name = os.path.basename(args.output_path)
    args.checkpoint_path = os.path.join(args.output_path, "checkpoint")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    args.config_path = os.path.join(args.output_path, os.path.basename(args.config))
    if args.config: shutil.copy(args.config, args.config_path)

    if is_master(args, local=args.log_local):
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(args.output_path, log_filename)
        if os.path.exists(args.log_path):
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)
    
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    writer = None
    if is_master(args):
        args.tensorboard_path = os.path.join(args.output_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    random_seed(args.seed)
    # autodtype = get_cast_dtype(args.precision)

    scaler = None
    if args.precision == "amp":
        try:
            scaler = torch.amp.GradScaler(device=device)
        except (AttributeError, TypeError) as e:
            scaler = torch.cuda.amp.GradScaler()

    print("## Load Dataset")
    train_loader, valid_loader = make_style_dataset(args)

    print("## Model Create\n")
    model = create_clip(
        args.model_name, args.model_kargs, args.vision_kargs, args.text_kargs, args.pretrain, 
        output_dict=True, compute_style=True, 
    )
    model.to(device=device)

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")

    params_group = make_clip_param_groups(model, args.image_lr, args.text_lr, args.lr_decay, args.patch_embed_lr_mult)

    if args.opt_name == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(params_group, lr=args.lr, weight_decay=args.wd, betas=(args.beta1, args.beta2), eps=1e-08)    # float16 학습을 위함

    elif args.opt_name == "adamw":
        from torch.optim import AdamW
        optimizer = AdamW(params_group, lr=args.lr, weight_decay=args.wd, betas=(args.beta1, args.beta2), eps=1e-08)    # float16으로 학습하지 말것! 모델이 nan을 출력함.

    clip_loss = SigLipLoss2(
        rank=args.rank,
        world_size=args.world_size,
        dist_impl=args.loss_dist_impl,  # siglip has multiple distributed implementations to choose from
    )

    style_loss = StyleSigLip(
        rank=args.rank,
        world_size=args.world_size,
        dist_impl=args.loss_dist_impl,  # siglip has multiple distributed implementations to choose from
    )

    cos_sim_loss = CosineSimilarityLoss(dim=-1)

    if is_master(args):
        defaults = copy.deepcopy(optimizer.defaults)
        defaults['weight_decay'] = args.wd
        defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
        logging.info(
            f'Created {type(optimizer).__name__} ({args.opt_name}) optimizer: {defaults}'
        )

    start_epoch = 0
    if args.resume is not None:
        logging.info(f"Resume training")
        logging.info(f"Resume checkpoint: {args.resume}")
        logging.info(f"wandb run_id: {args.run_id}")

        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            load_random_state(checkpoint)
            model.load_state_dict(checkpoint["state_dict"])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
        
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)

        logging.info(f"Resume checkpoint is loaded successfully")
        logging.info(f"Training begins at epoch {start_epoch}.")


    if args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')

        wandb.init(
            dir="logs",
            project=args.wandb_project_name,
            name=None,
            id=args.run_id,
            notes=None,
            tags=[],
            resume='must' if args.resume or args.run_id else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')

        wandb.save(args.config_path)
        logging.info(f'Finished loading wandb. run_id: {wandb.run.id}')


    total_steps = (train_loader.num_batches // args.accum_freq) * args.epochs

    warmup_steps = total_steps * args.warmup_ratio if args.warmup_ratio is not None else args.warmup
    scheduler = CosineScheduler(args.lr, args.min_lr, total_steps, warmup_steps)
    print(f"## Using CosineScheduler \nwarmup steps: {warmup_steps} \ntotal steps: {total_steps}\n")

    # args.step = 0
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, device, train_loader, clip_loss, style_loss, epoch, optimizer, scaler, scheduler, writer, args)
        completed_epoch = epoch + 1
        metrics = evaluate(model, device, valid_loader, clip_loss, style_loss, completed_epoch, writer, args.step, args)
        # breakpoint()
        # args.step += 1

        checkpoint_dict = {
            "epoch": completed_epoch,
            "name": args.model_name,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            'torch_rng_state': torch.random.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'numpy_rng_state': np.random.get_state(),
            'random_state': random.getstate()
        }
        if scaler is not None:
            checkpoint_dict["scaler"] = scaler.state_dict()

        if completed_epoch == args.epochs or (
            args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
        ):
            model_save_path = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt")
            
            torch.save(
                checkpoint_dict,
                model_save_path,
            )

            saved_models = [os.path.join(args.checkpoint_path, file) for file in os.listdir(args.checkpoint_path) if file.endswith(".pt")]
            saved_models.sort(key=lambda x: int(re.search(r"epoch_(\d+).pt", os.path.basename(x)).group(1)))
            if len(saved_models) > args.max_save_count:
                remove_paths = saved_models[:-args.max_save_count]
                for remove_path in remove_paths:
                    if os.path.exists(remove_path):
                        os.remove(remove_path)
            
    
    if args.wandb and is_master(args):
        wandb.finish()

        logging.info('Finished.')


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = read_config(args, parser)
    main(args)