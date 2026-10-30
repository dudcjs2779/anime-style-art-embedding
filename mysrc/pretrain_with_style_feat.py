import json
import re
import logging
import os
import shutil
from datetime import datetime
import time
import math
import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
import wandb
from dotenv import load_dotenv
load_dotenv()

import torch.utils.tensorboard as tensorboard
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# from open_clip_train.distributed import is_master, init_distributed_device
# from open_clip_train.logger import setup_logging

from mysrc.utils.distributed import is_master
from mysrc.utils.logger import setup_logging
from mysrc.utils.pretrain_params import setup_parser
from mysrc.utils.config import read_config
from mysrc.utils.dataset import make_pretrain_dataset04
from mysrc.utils.optim_utils import make_mae_param_groups
from mysrc.utils.loss import RelationPrototypeContrastiveLoss, RelationSigLipLoss, similarity_mse_loss, content_style_sim_loss, BCELoss
from mysrc.utils.scheduler import CosineScheduler, StepGenerator
from mysrc.utils.precision import get_autocast, get_input_dtype
from mysrc.model import StyleFeatVisual
from mysrc.utils.file_utils import pt_load
from mysrc.utils.metrics import get_embed_metrics, get_proto_metrics, get_class_topk, get_sfg_metrics
from mysrc.utils.pos_embed import interpolate_pos_embed


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

class SampleManager:
    def __init__(self, n_samples, momentum=0.9):
        self.loss = torch.zeros(n_samples)
        self.count = torch.zeros(n_samples)
        self.seen = torch.zeros(n_samples, dtype=torch.bool)
        self.pending_sum = torch.zeros(n_samples)
        self.pending_count = torch.zeros(n_samples)
        self.momentum = momentum
        self.total_count = 0

    @torch.no_grad()
    def update(self, indices, loss):
        indices = indices.detach().to(device="cpu", dtype=torch.long).reshape(-1)
        values = loss.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        if indices.numel() != values.numel():
            raise ValueError("sample indices and losses must have the same length")
        self.pending_sum.index_add_(0, indices, values)
        self.pending_count.index_add_(0, indices, torch.ones_like(values))

    @torch.no_grad()
    def _synchronize_pending(self):
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            device = torch.device("cpu")
            if dist.get_backend() == "nccl":
                device = torch.device("cuda", torch.cuda.current_device())
            pending_sum = self.pending_sum.to(device=device)
            pending_count = self.pending_count.to(device=device)
            dist.all_reduce(pending_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(pending_count, op=dist.ReduceOp.SUM)
            self.pending_sum.copy_(pending_sum.cpu())
            self.pending_count.copy_(pending_count.cpu())

        observed = self.pending_count > 0
        if observed.any():
            new = self.pending_sum[observed] / self.pending_count[observed]
            first = ~self.seen[observed]
            old = self.loss[observed]
            self.loss[observed] = torch.where(
                first,
                new,
                self.momentum * old + (1 - self.momentum) * new,
            )
            self.count[observed] += self.pending_count[observed]
            self.seen[observed] = True
            self.total_count += int(self.pending_count.sum().item())

        self.pending_sum.zero_()
        self.pending_count.zero_()

    @torch.no_grad()
    def get_score(self, alpha=0.9):
        # All ranks call this at the epoch boundary; synchronize observations
        # before computing the score vector used by the global batch sampler.
        self._synchronize_pending()
        mean_count = self.total_count / len(self.loss)
        hardness = self.loss
        exploration = (
            np.sqrt(mean_count + 1)
            / (self.count + 1).sqrt()
        )
        return alpha * hardness + (1 - alpha) * exploration


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
            if checkpoint['state_dict']['prototypes'].shape != model.prototypes.shape:
                logging.info(f"The model of the prototpye shape not matches pretrain prototype shape. {model.prototypes.shape}(model) != {checkpoint['state_dict']['prototypes'].shape}(pretrained). Therefore, the prototype vector is excluded.")
                del checkpoint['state_dict']['prototypes']

            if checkpoint['state_dict']['self_style_head.weight'].shape != model.self_style_head.weight.shape:
                del checkpoint['state_dict']['self_style_head.weight']
                del checkpoint['state_dict']['self_style_head.bias']

            if checkpoint['state_dict']['content_head.weight'].shape != model.content_head.weight.shape:
                del checkpoint['state_dict']['content_head.weight']
                del checkpoint['state_dict']['content_head.bias']

            if model.visual.pos_embed.shape != checkpoint['state_dict']['visual.pos_embed'].shape:
                model.visual.pos_embed, checkpoint['state_dict']['visual.pos_embed']
                interpolate_pos_embed(model, checkpoint['state_dict'])

            missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            if checkpoint.get('prototypes') and checkpoint['prototypes'].shape != model.prototypes.shape:
                logging.info(f"The model of the prototpye shape not matches pretrain prototype shape. {model.prototypes.shape}(model) != {checkpoint['prototypes'].shape}(pretrained). Therefore, the prototype vector is excluded.")
                del checkpoint['state_dict']['prototypes']

            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            
        logging.info(f"Loaded pretrain model from {path}")
        logging.info(f"missing_key: {len(missing)}: {missing[:10]}")
        logging.info(f"unexpected_key: {len(unexpected)}: {unexpected[:10]}")

def train_one_epoch(
        model, device, dataloader, 
        clip_loss, proto_loss, feat_loss, self_clip_loss, self_proto_loss, 
        epoch, optimizer, scaler, 
        scheduler, clip_scheduler, proto_scheduler, proto_momentum_scheduler, batch_sampler_scheduler, 
        tb_writer, sample_manager, args
    ):
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    model.train()

    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if epoch >= 1:
        scores = sample_manager.get_score(alpha=0.9)
        dataloader.batch_sampler.update_score(scores, top_p=10)

    if args.accum_freq > 1:
        accum_images, accum_classes = [], []
        accum_sfg_label = []
        accum_crop_masks = []
        accum_not_fars = []
        accum_index = []
        # accum_mask_indices, accum_nm_indices, accum_mask_samples = [], [], []
        accum_features = {}

    losses_m = {}
    # total_m = AverageMeter()
    # clip_loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    # if args.local_rank == 0:
    #     breakpoint()

    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        if batch_sampler_scheduler.is_update_step(step):
            sim_matrix = model.prototypes @ model.prototypes.T
            sim_matrix = torch.clip(sim_matrix, 0, 1.0)
            sim_matrix.fill_diagonal_(1.0)

            dataloader.batch_sampler.rel_pseudo = sim_matrix.detach().cpu().numpy()
            dataloader.batch_sampler.update_weights()
            
        lr = scheduler[step]
        proto_loss.momentum = proto_momentum_scheduler[step]
        apply_optim_scheduler(optimizer, lr)

        index, images, classes, sfg_label, hards, centers, not_fars = batch['index'], batch['image'], batch['label'], batch['feat_label'], batch['hard'], batch['center'], batch['not_far']
        index = index.to(device='cpu', dtype=torch.long, non_blocking=True)
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        classes = classes.to(device=device, dtype=torch.long, non_blocking=True)
        sfg_label = sfg_label.to(device=device, dtype=torch.float32, non_blocking=True)
        hards = hards.to(device=device, dtype=torch.bool, non_blocking=True)
        centers = centers.to(device=device, dtype=torch.bool, non_blocking=True)
        not_fars = not_fars.to(device=device, dtype=torch.bool, non_blocking=True)

        crop_masks, mask_indices, none_mask_indices, mask_samples = batch['crop_masks'], batch['mask_indices'], batch['none_mask_indices'], batch['mask_samples']
        crop_masks = crop_masks.to(device=device, non_blocking=True)
        # mask_indices = mask_indices.to(device=device, non_blocking=True)
        # none_mask_indices = none_mask_indices.to(device=device, non_blocking=True)
        # mask_samples = mask_samples.to(device=device, non_blocking=True)

        # dataloader.batch_sampler.all_batches[-1]
        # dataloader.batch_sampler.all_classes[-1]
        # dataloader.batch_sampler.sample_diff_weights

        self_prototypes = model.module.self_prototypes if args.distributed else model.self_prototypes
        prototypes = model.module.prototypes if args.distributed else model.prototypes

        bsz = len(images) // 2

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()
        
        if args.accum_freq == 1:
            with autocast():
                model_out = model(
                    images, 
                    crop_masks, 
                    # mask_indices, none_mask_indices, mask_samples,
                )
                cls_features = {
                    'style_feat1': model_out['style_cls'][:bsz],
                    'style_feat2': model_out['style_cls'][bsz:],
                    'content_feat1': model_out['content_cls'][:bsz],
                    'content_feat2': model_out['content_cls'][bsz:],
                    'self_style_feat1': model_out['self_style_cls'][bsz:],
                    'self_style_feat2': model_out['self_style_cls'][:bsz],
                    'sfg_feat': model_out['sfg_cls'],
                }
                logit_scale = model_out["logit_scale"]
                logit_bias = model_out.get("logit_bias", None)

                losses = {}

                losses['self_style_loss'] = self_clip_loss(
                    cls_features['self_style_feat1'], cls_features['self_style_feat2'], 
                    logit_scale, logit_bias, 
                    classes, 
                    not_fars=not_fars,
                    output_dict=False
                )

                losses['self_proto_loss'] = self_proto_loss(
                    cls_features['self_style_feat1'], self_prototypes,
                    logit_scale, logit_bias,
                    classes,
                    not_fars=not_fars,
                    index=index,
                    sample_manager=sample_manager,
                    sample_manager_update=False,
                    train=True,
                ) * 0.5

                losses['self_proto_loss'] +=  self_proto_loss(
                    cls_features['self_style_feat2'], self_prototypes,
                    logit_scale, logit_bias,
                    classes,
                    not_fars=not_fars,
                    index=index,
                    sample_manager=sample_manager,
                    sample_manager_update=False,
                    train=True,
                ) * 0.5

                losses['style_loss'] = clip_loss(
                    cls_features['style_feat1'], cls_features['style_feat2'], 
                    logit_scale, logit_bias, 
                    classes, 
                    not_fars=not_fars,
                    output_dict=False
                )

                # proto_loss를 따로 계산하는 이유는 accum 사용시 꼬일 수 있기 때문에 일관성을 유지하기 위함
                losses['proto_loss'] = proto_loss(
                    cls_features['style_feat1'], prototypes,
                    logit_scale, logit_bias,
                    classes,
                    hards=hards, centers=centers, not_fars=not_fars,
                    index=index, sample_manager=sample_manager,
                    train=True,
                ) * 0.5

                losses['proto_loss'] +=  proto_loss(
                    cls_features['style_feat2'], prototypes,
                    logit_scale, logit_bias, 
                    classes,
                    hards=hards, centers=centers, not_fars=not_fars,
                    index=index, sample_manager=sample_manager,
                    train=True, 
                ) * 0.5

                losses['content_loss'] = clip_loss(
                    cls_features['content_feat1'], cls_features['content_feat2'], 
                    logit_scale, logit_bias,
                    output_dict=False
                )

                losses['sfg_loss'] = feat_loss(cls_features['sfg_feat'], torch.concat([sfg_label, sfg_label], dim=0))

                # contetn style sim loss
                losses['content_style_cosine_loss'] = content_style_sim_loss(
                    torch.concat((cls_features['content_feat1'], cls_features['content_feat2']), dim=0),
                    torch.concat((cls_features['style_feat1'], cls_features['style_feat2']), dim=0),
                )

                # # similarity mse loss
                # losses['mse_loss'] = similarity_mse_loss(
                #     torch.cat((cls_features['style_feat1'][~not_fars], cls_features['style_feat2'][~not_fars]), dim=0),
                #     torch.cat((classes[~not_fars], classes[~not_fars]), dim=0).long(),
                #     model.prototypes, args.rel_real,
                # )
                
                # ## Reconstruction loss
                # pred = model_out['im_pred']
                # masks = model_out['masks']
                # label = model.decoder.patchify(images[mask_samples])

                # loss = mae_loss(pred, label, masks, norm_pix_loss=args.norm_pix_loss)
                # losses['mae_loss'] = loss

                losses["raw_loss"] = sum(losses.values())
                losses["loss"] = (
                    losses['style_loss'] * clip_scheduler[step] 
                    + losses['proto_loss'] * proto_scheduler[step] 
                    + losses['content_loss'] 
                    + losses['content_style_cosine_loss'] 
                    + losses['sfg_loss']
                    + losses['self_style_loss'] * clip_scheduler[step] 
                    + losses['self_proto_loss'] * proto_scheduler[step] 
                    # + losses['mse_loss'] * 1.0 
                    # + losses['mae_loss'] * 0.5

                )

            backward(losses["loss"], scaler)

            if args.world_size > 1:
                for key in losses.keys():
                    dist.all_reduce(losses[key], op=dist.ReduceOp.SUM)
                    losses[key] /= args.world_size

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
                        'self_style_feat1': model_out['self_style_cls'][bsz:],
                        'self_style_feat2': model_out['self_style_cls'][:bsz],
                    }

                    for key, val in cls_features.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]
                    
                accum_images.append(images)
                accum_classes.append(classes)
                accum_sfg_label.append(sfg_label)
                accum_index.append(index)
                accum_not_fars.append(not_fars)
                accum_crop_masks.append(crop_masks)
                # accum_mask_indices.append(mask_indices)
                # accum_nm_indices.append(none_mask_indices)
                # accum_mask_samples.append(mask_samples)

            # If (i + 1) % accum_freq is not zero, move on to the next batch.
            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            optimizer.zero_grad()
            accum_classes = torch.concat(accum_classes, dim=-1)
            accum_not_fars = torch.concat(accum_not_fars, dim=-1)
            accum_index = torch.concat(accum_index, dim=-1)

            losses = {}
            losses['content_style_cosine_loss_log'] = 0
            losses['sfg_loss_log'] = 0
            for j in range(args.accum_freq):
                images = accum_images[j]
                classes = accum_classes[j * bsz:(j + 1) * bsz]
                sfg_label = accum_sfg_label[j]
                crop_masks = accum_crop_masks[j]
                # mask_indices = accum_mask_indices[j]
                # none_mask_indices = accum_nm_indices[j]
                # mask_samples = accum_mask_samples[j]

                with autocast():
                    model_out = model(
                        images, 
                        crop_masks, 
                        # mask_indices, none_mask_indices, mask_samples,
                    )
                    cls_features = {
                        'style_feat1': model_out['style_cls'][:bsz],
                        'style_feat2': model_out['style_cls'][bsz:],
                        'content_feat1': model_out['content_cls'][:bsz],
                        'content_feat2': model_out['content_cls'][bsz:],
                        'self_style_feat1': model_out['self_style_cls'][bsz:],
                        'self_style_feat2': model_out['self_style_cls'][:bsz],
                    }
                    sfg_feat = model_out['sfg_cls']
                    logit_scale = model_out["logit_scale"]
                    logit_bias = model_out.get("logit_bias", None)

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [cls_features[key]] + accumulated[j + 1:])

                    losses['self_style_loss'] = self_clip_loss(
                        inputs['self_style_feat1'], inputs['self_style_feat2'], 
                        logit_scale, logit_bias, 
                        accum_classes, 
                        not_fars=accum_not_fars,
                        output_dict=False
                    )

                    losses['self_proto_loss'] = self_proto_loss(
                        inputs['self_style_feat1'], self_prototypes,
                        logit_scale, logit_bias,
                        accum_classes,
                        not_fars=accum_not_fars,
                        index=accum_index,
                        sample_manager=sample_manager,
                        sample_manager_update=False,
                        train=(j==0),
                    ) * 0.5

                    losses['self_proto_loss'] +=  self_proto_loss(
                        inputs['self_style_feat2'], self_prototypes,
                        logit_scale, logit_bias,
                        accum_classes,
                        not_fars=accum_not_fars,
                        index=accum_index,
                        sample_manager=sample_manager,
                        sample_manager_update=False,
                        train=(j==0),
                    ) * 0.5

                    losses['style_loss'] = clip_loss(
                        inputs['style_feat1'], inputs['style_feat2'], 
                        logit_scale, logit_bias, 
                        accum_classes, 
                        not_fars=accum_not_fars,
                        output_dict=False
                    )

                    losses['proto_loss'] = proto_loss(
                        inputs['style_feat1'], prototypes,
                        logit_scale, logit_bias,
                        accum_classes,
                        not_fars=accum_not_fars,
                        index=accum_index,
                        sample_manager=sample_manager,
                        sample_manager_update=(j==0),
                        train=(j==0),
                    ) * 0.5

                    losses['proto_loss'] +=  proto_loss(
                        inputs['style_feat2'], prototypes,
                        logit_scale, logit_bias,
                        accum_classes,
                        not_fars=accum_not_fars,
                        index=accum_index,
                        sample_manager=sample_manager,
                        sample_manager_update=(j==0),
                        train=(j==0),
                    ) * 0.5

                    losses['content_loss'] = clip_loss(
                        inputs['content_feat1'], inputs['content_feat2'], 
                        logit_scale, logit_bias,
                        output_dict=False
                    )

                    # no need concat
                    losses['content_style_cosine_loss'] = content_style_sim_loss(
                        torch.concat((cls_features['content_feat1'], cls_features['content_feat2']), dim=0),
                        torch.concat((cls_features['style_feat1'], cls_features['style_feat2']), dim=0),
                    ) / args.accum_freq

                    losses['content_style_cosine_loss_log'] += losses['content_style_cosine_loss'].detach()
                    
                    losses['sfg_loss'] = feat_loss(sfg_feat, torch.cat((sfg_label, sfg_label), dim=0)) / args.accum_freq
                    losses['sfg_loss_log'] += losses['sfg_loss'].detach()

                    # # similarity mse loss
                    # losses['mse_loss'] = similarity_mse_loss(
                    #     torch.cat((cls_features['style_feat1'], cls_features['style_feat2']), dim=0),
                    #     torch.cat((classes, classes), dim=0).long(),
                    #     prototypes, args.rel_real
                    # ) / args.accum_freq

                    del inputs

                    losses["loss"] = (
                        losses['style_loss'] * clip_scheduler[step] 
                        + losses['proto_loss'] * proto_scheduler[step] 
                        + losses['content_loss'] 
                        + losses['content_style_cosine_loss'] 
                        + losses['sfg_loss']
                        + losses['self_style_loss'] * clip_scheduler[step] 
                        + losses['self_proto_loss'] * proto_scheduler[step] 
                        # losses['mse_loss'] * 1.0 + 
                    )

                backward(losses["loss"], scaler)

            # accum_freq 일부 loss가 왜곡되는 부분 대응
            losses['content_style_cosine_loss'] = losses['content_style_cosine_loss_log']
            losses['sfg_loss'] = losses['sfg_loss_log']
            del losses['content_style_cosine_loss_log']
            del losses['sfg_loss_log']
            del losses['loss']

            losses['raw_loss'] = sum(losses.values())
            losses["loss"] = (
                losses['style_loss'] * clip_scheduler[step] 
                + losses['proto_loss'] * proto_scheduler[step] 
                + losses['content_loss'] 
                + losses['content_style_cosine_loss']
                + losses['sfg_loss']
                + losses['self_style_loss'] * clip_scheduler[step] 
                + losses['self_proto_loss'] * proto_scheduler[step] 
                # losses['mse_loss'] * 1.0 + 
            )

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
            accum_images, accum_classes  = [], []
            accum_sfg_label = []
            accum_crop_masks = []
            accum_index = []
            accum_not_fars = []
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
            for key, val in losses.items():
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


def evaluate(
        model, device, dataloader, 
        clip_loss, proto_loss, feat_loss, self_clip_loss, self_proto_loss,
        epoch, tb_writer, step, args):
    metrics = {}
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    model.eval()

    if args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs):
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # cumulative_loss = {}
        losses_m = {}
        all_image_features, all_self_image_features, all_classes = [], [], []
        all_sfg_features, all_sfg_labels = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, image_eval, classes, sfg_label = batch['image'], batch['image_eval'], batch['label'], batch['feat_label']
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)   # aug1
                image_eval = image_eval.to(device=device, dtype=input_dtype, non_blocking=True)   # non aug
                classes = classes.to(device=device, dtype=torch.long, non_blocking=True)
                sfg_label = sfg_label.to(device=device, dtype=torch.float32, non_blocking=True)

                crop_masks, mask_indices, none_mask_indices, mask_samples = batch['crop_masks'], batch['mask_indices'], batch['none_mask_indices'], batch['mask_samples']
                crop_masks = crop_masks.to(device=device, non_blocking=True)
                mask_indices = mask_indices.to(device=device, non_blocking=True)
                none_mask_indices = none_mask_indices.to(device=device, non_blocking=True)
                mask_samples = mask_samples.to(device=device, non_blocking=True)

                self_prototypes = model.module.self_prototypes if args.distributed else model.self_prototypes
                prototypes = model.module.prototypes if args.distributed else model.prototypes

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
                        'self_style_feat1': model_out['self_style_cls'][batch_size:],
                        'self_style_feat2': model_out['self_style_cls'][:batch_size],
                    }
                    eval_features = model_out["eval_style_cls"]
                    # eval_self_features = model_out["eval_self_style_cls"]
                    eval_sfg_feat = model_out["eval_sfg_cls"]
                    logit_scale = model_out["logit_scale"]
                    logit_bias = model_out.get('logit_bias', None)
                    
                    all_image_features.append(eval_features.cpu())
                    # all_self_image_features.append(eval_self_features.cpu())
                    all_classes.append(classes.cpu())
                    all_sfg_features.append(eval_sfg_feat.cpu())
                    all_sfg_labels.append(sfg_label.cpu())

                    losses = {}
                    losses['self_style_loss'] = self_clip_loss(
                        cls_features['self_style_feat1'], cls_features['self_style_feat2'], 
                        logit_scale, logit_bias, 
                        classes=classes, 
                        output_dict=False
                    )

                    losses['self_proto_loss'] = self_proto_loss(
                        cls_features['self_style_feat1'], self_prototypes,
                        logit_scale, logit_bias,
                        classes,
                    ) * 0.5

                    losses['self_proto_loss'] +=  self_proto_loss(
                        cls_features['self_style_feat2'], self_prototypes,
                        logit_scale, logit_bias,
                        classes,
                    ) * 0.5


                    losses['style_loss'] = clip_loss(
                        cls_features['style_feat1'], cls_features['style_feat2'], 
                        logit_scale, logit_bias, 
                        classes=classes,
                        output_dict=False
                    )
                    losses['proto_loss'] = proto_loss(
                        cls_features['style_feat1'], prototypes,
                        logit_scale, logit_bias,
                        classes, 
                    ) * 0.5

                    losses['proto_loss'] +=  proto_loss(
                        cls_features['style_feat2'], prototypes,
                        logit_scale, logit_bias,
                        classes,
                    ) * 0.5

                    losses['content_loss'] = clip_loss(
                        cls_features['content_feat1'], cls_features['content_feat2'], 
                        logit_scale, logit_bias,
                        output_dict=False
                    )

                    # contetn style sim loss
                    losses['content_style_cosine_loss'] = content_style_sim_loss(
                        torch.concat((cls_features['content_feat1'], cls_features['content_feat2']), dim=0),
                        torch.concat((cls_features['style_feat1'], cls_features['style_feat2']), dim=0),
                    )

                    losses['sfg_loss'] = feat_loss(eval_sfg_feat, sfg_label)

                    losses["loss"] = sum(losses.values())

                if args.world_size > 1:
                    for key in losses.keys():
                        dist.all_reduce(losses[key], op=dist.ReduceOp.SUM)
                        losses[key] /= args.world_size

                if is_master(args):
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
                    num_samples += batch_size * args.world_size

                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t" + loss_log
                        # f"Clip Loss: {losses_m['contrastive_loss'].avg:.6f}\t"
                        # f"Recon Loss: {losses_m['mae_loss'].avg:.6f}\t"
                    )

            all_image_features = torch.cat(all_image_features)
            all_classes = torch.cat(all_classes)
                    
            if is_master(args):
                if args.world_size > 1:
                    gathered = [torch.empty_like(all_image_features) for _ in range(args.world_size)]
                    dist.all_gather(gathered, all_image_features)
                    all_image_features = torch.cat(gathered, dim=0)

                    gathered = [torch.empty_like(all_classes) for _ in range(args.world_size)]
                    dist.all_gather(gathered, all_classes)
                    all_classes = torch.cat(gathered, dim=0)

                val_metrics = get_class_topk(
                    image_features=all_image_features,
                    prototypes=prototypes.cpu(),
                    classes=all_classes,
                    rel_real=args.rel_real,
                )

                val_metrics.update(
                    get_sfg_metrics(torch.cat(all_sfg_features), torch.cat(all_sfg_labels), threshold=0.4)
                )

                metrics.update({
                    **val_metrics, 
                    "style_loss": losses_m['style_loss'].avg, 
                    "proto_loss": losses_m['proto_loss'].avg, 
                    "content_loss": losses_m['content_loss'].avg, 
                    "self_style_loss": losses_m['self_style_loss'].avg, 
                    "self_proto_loss": losses_m['self_proto_loss'].avg, 
                    "content_style_cosine_loss": losses_m['content_style_cosine_loss'].avg, 
                    "epoch": epoch, "num_samples": num_samples
                })

    if is_master(args):
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

def test_evaluate(
        model, device, dataloader, 
        clip_loss, proto_loss, self_clip_loss, self_proto_loss,
        epoch, tb_writer, step, args):
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
                images, image_eval, classes = batch['image'], batch['image_eval'], batch['label']
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)   # aug1
                image_eval = image_eval.to(device=device, dtype=input_dtype, non_blocking=True)   # non aug
                classes = classes.to(device=device, dtype=torch.long, non_blocking=True)

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
                        'self_style_feat1': model_out['self_style_cls'][batch_size:],
                        'self_style_feat2': model_out['self_style_cls'][:batch_size],

                    }
                    eval_features = model_out["eval_style_cls"]
                    # eval_self_features = model_out["eval_self_style_cls"]
                    logit_scale = model_out["logit_scale"]
                    logit_bias = model_out.get('logit_bias', None)
                    
                    all_image_features.append(eval_features.cpu())
                    all_classes.append(classes.cpu())

                    losses = {}
                    losses['self_style_loss'] = self_clip_loss(
                        cls_features['self_style_feat1'], cls_features['self_style_feat2'], 
                        logit_scale, logit_bias, 
                        classes=classes, 
                        output_dict=False
                    )

                    # losses['self_proto_loss'] = self_proto_loss(
                    #     cls_features['self_style_feat1'], self_prototypes,
                    #     logit_scale, logit_bias,
                    #     classes,
                    # ) * 0.5

                    # losses['self_proto_loss'] +=  self_proto_loss(
                    #     cls_features['self_style_feat2'], self_prototypes,
                    #     logit_scale, logit_bias,
                    #     classes,
                    # ) * 0.5


                    losses['style_loss'] = clip_loss(
                        cls_features['style_feat1'], cls_features['style_feat2'], 
                        logit_scale, logit_bias, 
                        classes=classes,
                        output_dict=False
                    )
                    # losses['proto_loss'] = proto_loss(
                    #     cls_features['style_feat1'], prototypes,
                    #     logit_scale, logit_bias,
                    #     classes, 
                    # ) * 0.5

                    # losses['proto_loss'] +=  proto_loss(
                    #     cls_features['style_feat2'], prototypes,
                    #     logit_scale, logit_bias,
                    #     classes,
                    # ) * 0.5

                    losses['content_loss'] = clip_loss(
                        cls_features['content_feat1'], cls_features['content_feat2'], 
                        logit_scale, logit_bias,
                        output_dict=False
                    )

                    # contetn style sim loss
                    losses['content_style_cosine_loss'] = content_style_sim_loss(
                        torch.concat((cls_features['content_feat1'], cls_features['content_feat2']), dim=0),
                        torch.concat((cls_features['style_feat1'], cls_features['style_feat2']), dim=0),
                    )

                    losses["loss"] = sum(losses.values())

                if args.world_size > 1:
                    for key in losses.keys():
                        dist.all_reduce(losses[key], op=dist.ReduceOp.SUM)
                        losses[key] /= args.world_size
                        
                if is_master(args):

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
                num_samples += batch_size * args.world_size

                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Test Epoch: {epoch} [{num_samples} / {samples_per_val}]\t" + loss_log
                        # f"Clip Loss: {losses_m['contrastive_loss'].avg:.6f}\t"
                        # f"Recon Loss: {losses_m['mae_loss'].avg:.6f}\t"
                    )
            
            all_image_features = torch.cat(all_image_features)
            all_classes = torch.cat(all_classes)

            if is_master(args) and args.world_size > 1:
                gathered = [torch.empty_like(all_image_features) for _ in range(args.world_size)]
                dist.all_gather(gathered, all_image_features)
                all_image_features = torch.cat(gathered, dim=0)

                gathered = [torch.empty_like(all_classes) for _ in range(args.world_size)]
                dist.all_gather(gathered, all_classes)
                all_classes = torch.cat(gathered, dim=0)

            if is_master(args):
                centers = []
                for i in range(args.text_num_classes):
                    indices = torch.where(all_classes == i)
                    class_features = all_image_features[indices]
                    center = class_features.median(dim=0)[0]

                    centers.append(center)

                centers = torch.stack(centers)
                centers = torch.nn.functional.normalize(centers, dim=-1)

                val_metrics = get_class_topk(
                    image_features=all_image_features,
                    prototypes=centers,
                    classes=all_classes,
                )

                metrics.update({
                    **val_metrics, 
                    "style_loss": losses_m['style_loss'].avg, 
                    # "proto_loss": losses_m['proto_loss'].avg, 
                    "content_loss": losses_m['content_loss'].avg, 
                    "self_style_loss": losses_m['self_style_loss'].avg, 
                    # "self_proto_loss": losses_m['self_proto_loss'].avg, 
                    "content_style_cosine_loss": losses_m['content_style_cosine_loss'].avg, 
                    "epoch": epoch, "num_samples": num_samples
                })

    if is_master(args):
        logging.info(
            f"Test Epoch: {epoch} "
            + "\t".join([f"{k}: {round(v, 4):.6f}" for k, v in metrics.items()])
        )

        log_data = {"test/" + name: val for name, val in metrics.items()}

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
    if "RANK" in os.environ:
        args.distributed = True
        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        assert args.world_size > 1 and args.accum_freq == 1, ("Not implemented accum with distributed training!")
    else:
        args.distributed = False

        # print("rank_test: ", args.rank, args.local_rank, args.world_size)

    if args.device == "cpu":
        backend = "gloo"
        device = torch.device("cpu")
    else:
        backend = "nccl"
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)

    if args.distributed:
        dist.init_process_group(backend=backend)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # device = init_distributed_device(args)

    t_now = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    args.logs = "./pretrain_log"
    args.output_path = os.path.join(args.logs, f"{t_now}_{args.model_name[:50]}")
    args.log_name = os.path.basename(args.output_path)
    args.checkpoint_path = os.path.join(args.output_path, "checkpoint")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    args.config_path = os.path.join(args.output_path, os.path.basename(args.config))
    if args.config: shutil.copy(args.config, args.config_path)

    args.log_path = None
    if is_master(args, local=args.log_local):
        log_filename = f'out-{args.local_rank}' if args.log_local else 'out.log'
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
    # train_loader, valid_loader = make_pretrain_dataset02(args)
    dataset = make_pretrain_dataset04(args)

    print("## Model Create\n")
    model = StyleFeatVisual(
        model_name=args.model_name, 
        model_kargs=args.model_kargs,
        feat_head = {"num_classes": args.feat_len},
        init_logit_scale=math.log(10), 
        init_logit_bias=-8, 
        output_dict=True,
        num_classes=args.num_classes,
    ).to(device=device)
    load_pretrain(model, args.pretrain)

    # 대표벡터 불러오기
    if args.prototype_path is not None:
        prototypes = torch.load(args.prototype_path)

        if prototypes.shape == model.prototypes.shape:
            model.prototypes = prototypes.to(device=device, dtype=torch.float32)
            logging.info(f"Loaded prototypes from {args.prototype_path} with shape {prototypes.shape}")

        elif prototypes.size(0) == model.prototypes.size(0) and prototypes.size(1) != model.prototypes.size(1):
            logging.info(f"Prototpyes will scaled to fit model.prototpyes. {prototypes.shape} => {model.prototypes.shape}")

            diff = model.prototypes.size(1) - prototypes.size(1)
            if diff > 0:
                prototypes = F.pad(prototypes, (0, diff), mode="constant", value=0.0)
            else:
                prototypes = F.interpolate(prototypes.unsqueeze(1), size=model.prototypes.size(1), mode='linear', align_corners=False).squeeze(1)

            model.prototypes = prototypes.to(device=device, dtype=torch.float32)
            logging.info(f"Loaded prototypes from {args.prototype_path}")

        else:
            logging.warning(f"Loaded prototypes shape {prototypes.shape} does not match model prototypes shape {model.prototypes.shape}. Skipping prototype loading.")

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


    clip_loss = RelationSigLipLoss(
        rel_matrix=args.rel_real,
        rel_real=args.rel_real,
        rank=args.rank,
        world_size=args.world_size,
        device=device,
    )

    proto_loss = RelationPrototypeContrastiveLoss(
        num_classes=args.num_classes,
        momentum=0.99,
        rel_matrix=args.rel_real, 
        rel_real=args.rel_real,
        topk_rate=args.topk_rate,
        rank=args.rank,
        world_size=args.world_size,
        device=device,
    )

    feat_loss = BCELoss(
        args.feat_class_weight,
        args.feat_pos_weight,
        device=device
    )

    self_clip_loss = RelationSigLipLoss(
        rank=args.rank,
        world_size=args.world_size,
        device=device,
    )

    self_proto_loss = RelationPrototypeContrastiveLoss(
        num_classes=args.num_classes,
        momentum=0.99,
        topk_rate=args.topk_rate,
        rank=args.rank,
        world_size=args.world_size,
        device=device,
    )


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
        breakpoint()
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
        
        if 'proto_mean' in checkpoint:
            proto_loss.statics.mean = checkpoint["proto_mean"].to(device=device)

        if 'sample_manager' in checkpoint:
            sample_manager.loss = checkpoint['sample_manager']['loss']
            sample_manager.count = checkpoint['sample_manager']['count']
            sample_manager.seen = checkpoint['sample_manager']['seen']
            sample_manager.total_count = checkpoint['sample_manager']['total_count']
        

        logging.info(f"Resume checkpoint is loaded successfully")
        logging.info(f"Training begins at epoch {start_epoch}.")

        logging.info(f"Resume checkpoint is loaded successfully")
        logging.info(f"Training begins at epoch {start_epoch}.")

    if args.distributed:
        if device.type == "cuda":
            model = DDP(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
            )
        else:
            model = DDP(model)

    if args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')

        if os.getenv("WANDB_API"):
            wandb.login(key=os.getenv("WANDB_API"))
            
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

    
    total_steps = (dataset['train'].num_batches // args.accum_freq) * args.epochs

    warmup_steps = total_steps * args.warmup_ratio if args.warmup_ratio is not None else args.warmup
    scheduler = CosineScheduler(args.lr, args.min_lr, total_steps, warmup_steps)
    clip_scheduler = CosineScheduler(*args.style_weight, total_steps // 2, 0)    # clip_loss weight scheduler
    proto_scheduler = CosineScheduler(*args.proto_weight, total_steps // 2, 0)    # clip_loss weight scheduler
    proto_momentum_scheduler = CosineScheduler(*args.proto_momentum, total_steps, 0)    # relation momentum
    batch_sampler_scheduler = StepGenerator(total_steps, args.BS_update_freq, args.BS_update_shape, args.BS_update_start)
    sample_manager = SampleManager(len(dataset['train'].dataset), 0.7)
    logging.info(f"## Using CosineScheduler \nwarmup steps: {warmup_steps} \ntotal steps: {total_steps}\n")
    logging.info(f"## batch_sampler_scheduler \nstep_list: {batch_sampler_scheduler.step_list[:10]} ... {batch_sampler_scheduler.step_list[-10:]}\n")

    # args.step = 0
    for epoch in range(start_epoch, args.epochs):
        if args.proto_ema_threshold_start_epoch <= epoch:
            if not proto_loss.ema_threshold_on:
                logging.info("The proto_loss_ema_threshold feature is turned on.")
            proto_loss.ema_threshold_on = True

        train_one_epoch(
            model, device, dataset['train'], 
            clip_loss, proto_loss, feat_loss, self_clip_loss, self_proto_loss,
            epoch, optimizer, scaler, scheduler, clip_scheduler, 
            proto_scheduler, proto_momentum_scheduler, batch_sampler_scheduler,
            writer, sample_manager, args
        )

        completed_epoch = epoch + 1
        # args.step += 1

        if args.valid:
            metrics = evaluate(
                model, device, dataset['valid'], 
                clip_loss, proto_loss, feat_loss, self_clip_loss, self_proto_loss,
                completed_epoch, writer, args.step, args
            )

        if args.test:
            metrics = test_evaluate(
                model, device, dataset['test'], 
                clip_loss, proto_loss, self_clip_loss, self_proto_loss,
                completed_epoch, writer, args.step, args
            )

        checkpoint_dict = {
            "epoch": completed_epoch,
            "name": args.model_name,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "proto_mean": proto_loss.statics.mean,
            "sample_manager": {"loss": sample_manager.loss, "count": sample_manager.count, "seen": sample_manager.seen, "total_count": sample_manager.total_count},
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
