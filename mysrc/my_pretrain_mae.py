import json
import re
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

import torch.utils.tensorboard as tensorboard

from open_clip_train.distributed import is_master, init_distributed_device
from open_clip_train.logger import setup_logging

from mysrc.utils.pretrain_params import setup_parser
from mysrc.utils.config import read_config
from mysrc.utils.dataset import make_pretrain_dataset, make_pretrain_dataset02, make_pretrain_dataset03
from mysrc.utils.optim_utils import make_mae_param_groups
from mysrc.utils.loss import SigLipLoss, CosineSimilarityLoss, mae_loss
from mysrc.utils.scheduler import CosineScheduler
from mysrc.utils.precision import get_autocast, get_input_dtype
from mysrc.model import CustomMaeVisual
from mysrc.utils.file_utils import pt_load
from mysrc.utils.metrics import get_embed_metrics


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

def load_pretrain(model, path):
    if path is not None and os.path.exists(path):
        checkpoint = torch.load(path, weights_only=False)

        if 'state_dict' in checkpoint:
            missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            
        logging.info(f"Loaded pretrain mode from {path}")
        logging.info(f"missing_key: {len(missing)}: {missing[:10]}")
        logging.info(f"unexpected_key: {len(unexpected)}: {unexpected[:10]}")

def train_one_epoch(model, device, dataloader, clip_loss, cosine_loss, epoch, optimizer, scaler, scheduler, tb_writer, args):
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    model.train()

    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images, accum_classes, accum_times = [], [], []
        accum_crop_masks, accum_mask_indices, accum_nm_indices, accum_mask_samples = [], [], [], []
        accum_features = {}

    losses_m = {}
    losses_log = {}
    # total_m = AverageMeter()
    # clip_loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        lr = scheduler[step]
        apply_optim_scheduler(optimizer, lr)

        images, classes, times = batch['image'], batch['label'], batch['time']
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        classes = classes.to(device=device, dtype=input_dtype, non_blocking=True)
        times = times.to(device=device, dtype=input_dtype, non_blocking=True)

        crop_masks, mask_indices, none_mask_indices, mask_samples = batch['crop_masks'], batch['mask_indices'], batch['none_mask_indices'], batch['mask_samples']
        crop_masks = crop_masks.to(device=device, non_blocking=True)
        mask_indices = mask_indices.to(device=device, non_blocking=True)
        none_mask_indices = none_mask_indices.to(device=device, non_blocking=True)
        mask_samples = mask_samples.to(device=device, non_blocking=True)

        bsz = len(images) // 2

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(
                    images, 
                    crop_masks, mask_indices, none_mask_indices, mask_samples,
                )
                cls_features = {
                    'style_feat1': model_out['style_cls'][:bsz],
                    'style_feat2': model_out['style_cls'][bsz:],
                    'content_feat1': model_out['content_cls'][:bsz],
                    'content_feat2': model_out['content_cls'][bsz:],
                }
                logit_scale = model_out["logit_scale"]
                logit_bias = model_out.get("logit_bias", None)
                
                losses = {}
                losses['style_loss'] = clip_loss(
                    cls_features['style_feat1'], cls_features['style_feat2'], 
                    logit_scale, logit_bias, 
                    classes, same_class_weight=args.same_class_weight, 
                    times=times, time_range=10, time_weight=0.3,
                    output_dict=False
                )
                losses['content_loss'] = clip_loss(
                    cls_features['content_feat1'], cls_features['content_feat2'], 
                    logit_scale, logit_bias,
                    output_dict=False
                )

                ## Reconstruction loss
                pred = model_out['im_pred']
                masks = model_out['masks']
                label = model.decoder.patchify(images[mask_samples])

                loss = mae_loss(pred, label, masks, norm_pix_loss=args.norm_pix_loss)
                # loss = loss * args.mae_weight
                losses['mae_loss'] = loss * 0.5

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
            losses_log.update(losses)

        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images, crop_masks)
                    cls_features = {
                        'style_feat1': model_out['style_cls'][:bsz],
                        'style_feat2': model_out['style_cls'][bsz:],
                        'content_feat1': model_out['content_cls'][:bsz],
                        'content_feat2': model_out['content_cls'][bsz:],
                    }

                    for key, val in cls_features.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]
                    
                accum_images.append(images)
                accum_classes.append(classes)
                accum_times.append(times)

                accum_crop_masks.append(crop_masks)
                accum_mask_indices.append(mask_indices)
                accum_nm_indices.append(none_mask_indices)
                accum_mask_samples.append(mask_samples)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            accum_classes = torch.concat(accum_classes, dim=-1)
            accum_times = torch.concat(accum_times, dim=-1)
            for j in range(args.accum_freq):
                images = accum_images[j]
                crop_masks = accum_crop_masks[j]
                mask_indices = accum_mask_indices[j]
                none_mask_indices = accum_nm_indices[j]
                mask_samples = accum_mask_samples[j]

                with autocast():
                    model_out = model(
                        images, 
                        crop_masks, mask_indices, none_mask_indices, mask_samples,
                    )
                    cls_features = {
                        'style_feat1': model_out['style_cls'][:bsz],
                        'style_feat2': model_out['style_cls'][bsz:],
                        'content_feat1': model_out['content_cls'][:bsz],
                        'content_feat2': model_out['content_cls'][bsz:],
                    }
                    logit_scale = model_out["logit_scale"]
                    logit_bias = model_out.get("logit_bias", None)

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [cls_features[key]] + accumulated[j + 1:])
                    
                    losses = {}
                    losses['style_loss'] = clip_loss(
                        inputs['style_feat1'], inputs['style_feat2'], 
                        logit_scale, logit_bias, 
                        accum_classes, same_class_weight=args.same_class_weight, 
                        times=accum_times, time_range=10, time_weight=0.3,
                        output_dict=False
                    )
                    losses['content_loss'] = clip_loss(
                        inputs['content_feat1'], inputs['content_feat2'], 
                        logit_scale, logit_bias,
                        output_dict=False
                    )

                    del inputs

                    ## Reconstruction loss
                    pred = model_out['im_pred']
                    masks = model_out['masks']
                    label = model.decoder.patchify(images[mask_samples])

                    loss = mae_loss(pred, label, masks, norm_pix_loss=args.norm_pix_loss)
                    loss /= args.accum_freq
                    # loss = loss * args.mae_weight
                    losses['mae_loss'] = loss

                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)
                
                # accum_freq 때문에 mae_lossr의 기록이 왜곡되는 것 방지
                losses_log.update(losses)
                del losses_log["loss"]
                losses_log['mae_loss'] = losses_log['mae_loss'] * args.accum_freq
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
            accum_images, accum_classes, accum_times = [], [], []
            accum_crop_masks, accum_mask_indices, accum_nm_indices, accum_mask_samples = [], [], [], []
            accum_features = {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

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


def evaluate(model, device, dataloader, clip_loss, cosine_loss, epoch, tb_writer, step, args):
    metrics = {}
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    model.eval()

    if args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs):
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # cumulative_loss = {}
        losses_m = {}
        all_image_features, all_classes = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, image_eval, classes, times = batch['image'], batch['image_eval'], batch['label'], batch['time']
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)   # aug1
                image_eval = image_eval.to(device=device, dtype=input_dtype, non_blocking=True)   # non aug
                classes = classes.to(device=device, dtype=input_dtype, non_blocking=True)
                times = times.to(device=device, dtype=input_dtype, non_blocking=True)

                crop_masks, mask_indices, none_mask_indices, mask_samples = batch['crop_masks'], batch['mask_indices'], batch['none_mask_indices'], batch['mask_samples']
                crop_masks = crop_masks.to(device=device, non_blocking=True)
                mask_indices = mask_indices.to(device=device, non_blocking=True)
                none_mask_indices = none_mask_indices.to(device=device, non_blocking=True)
                mask_samples = mask_samples.to(device=device, non_blocking=True)

                batch_size = images.shape[0] // 2
                with autocast():
                    model_out = model.evaluate(
                        images, image_eval,
                        crop_masks, mask_indices, none_mask_indices, mask_samples,
                    )
                    cls_features = {
                        'style_feat1': model_out['style_cls'][:batch_size],
                        'style_feat2': model_out['style_cls'][batch_size:],
                        'content_feat1': model_out['content_cls'][:batch_size],
                        'content_feat2': model_out['content_cls'][batch_size:],
                    }
                    eval_features = model_out["eval_style_cls"]
                    logit_scale = model_out["logit_scale"]
                    logit_bias = model_out.get('logit_bias', None)
                    
                    all_image_features.append(eval_features.cpu())
                    all_classes.append(classes.cpu())

                    losses = {}
                    losses['style_loss'] = clip_loss(
                        cls_features['style_feat1'], cls_features['style_feat2'], 
                        logit_scale, logit_bias, 
                        classes=classes, same_class_weight=args.same_class_weight,
                        times=times, time_range=args.time_range, time_weight=args.time_weight,
                        output_dict=False
                    )

                    losses['content_loss'] = clip_loss(
                        cls_features['content_feat1'], cls_features['content_feat2'], 
                        logit_scale, logit_bias,
                        output_dict=False
                    )

                    ## Reconstruction loss
                    pred = model_out['im_pred']
                    masks = model_out['masks']
                    label = model.decoder.patchify(images[mask_samples])

                    loss = mae_loss(pred, label, masks, norm_pix_loss=args.norm_pix_loss)
                    # loss = loss * args.mae_weight
                    losses['mae_loss'] = loss

                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

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
                num_samples += batch_size

                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t" + loss_log
                        # f"Clip Loss: {losses_m['contrastive_loss'].avg:.6f}\t"
                        # f"Recon Loss: {losses_m['mae_loss'].avg:.6f}\t"
                    )
            
            if logit_bias:
                logit_bias = logit_bias.cpu()

            val_metrics = get_embed_metrics(
                image_features=torch.cat(all_image_features),
                classes=torch.cat(all_classes),
            )

            metrics.update({
                    **val_metrics, 
                    "style_loss": losses_m['style_loss'].avg, 
                    "content_loss": losses_m['content_loss'].avg, 
                    "mae_loss": losses_m['mae_loss'].avg, 
                    "epoch": epoch, "num_samples": num_samples
                 })

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
    args.logs = "./pretrain_log"
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
    # train_loader, valid_loader = make_pretrain_dataset(args)
    train_loader, valid_loader = make_pretrain_dataset02(args)
    # train_loader, valid_loader = make_pretrain_dataset03(args)

    print("## Model Create\n")
    model = CustomMaeVisual(
        model_name=args.model_name, 
        decoder_name=args.decoder_name, 
        model_kargs=args.model_kargs,
        init_logit_scale=math.log(10), 
        init_logit_bias=-10, 
        output_dict=True, 
    ).to(device=device)
    load_pretrain(model, args.pretrain)

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")

    params_group = make_mae_param_groups(model, args.lr_decay, args.patch_embed_lr_mult)
    if args.opt_name == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(params_group, lr=args.lr, weight_decay=args.wd, betas=(args.beta1, args.beta2), eps=1e-08)    # float16 학습을 위함

    elif args.opt_name == "adamw":
        from torch.optim import AdamW
        optimizer = AdamW(params_group, lr=args.lr, weight_decay=args.wd, betas=(args.beta1, args.beta2), eps=1e-08)    # float16으로 학습하지 말것! 모델이 nan을 출력함.

    clip_loss = SigLipLoss(
        rank=args.rank,
        world_size=args.world_size,
        dist_impl=args.loss_dist_impl,  # siglip has multiple distributed implementations to choose from
    )

    cosine_loss = CosineSimilarityLoss(dim=-1)

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
        train_one_epoch(model, device, train_loader, clip_loss, cosine_loss, epoch, optimizer, scaler, scheduler, writer, args)
        completed_epoch = epoch + 1
        # args.step += 1
        metrics = evaluate(model, device, valid_loader, clip_loss, cosine_loss, completed_epoch, writer, args.step, args)

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