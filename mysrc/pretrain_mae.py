import os
import re
import numpy as np
from datetime import datetime
import shutil
import timm
from timm.models.vision_transformer import Block
import random
import math
import time
import wandb
import logging
import copy

import torch
import torch.nn as nn

from mysrc.utils.mae_params import setup_parser
from mysrc.utils.config import read_config
from mysrc.utils.pos_embed import get_2d_sincos_pos_embed
from mysrc.utils.logger import setup_logging
from mysrc.utils.precision import get_autocast, get_input_dtype
from mysrc.utils.scheduler import CosineScheduler
from mysrc.utils.optim_utils import make_param_groups
from mysrc.utils.dataset import make_mae_dataset
from mysrc.model_mae import MAE

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

def load_pretrain(model, path):
    if args.pretrain is not None and os.path.exists(path):
        checkpoint = torch.load(path, weights_only=False)

        if isinstance(checkpoint, dict):
            missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            
        logging.info(f"Loaded pretrain mode from {path}")
        logging.info(f"missing_key: {len(missing)}: {missing[:10]}")
        logging.info(f"unexpected_key: {len(unexpected)}: {unexpected[:10]}")



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

def train_one_epoch(model, device, dataloader, epoch, optimizer, scaler, scheduler, args):
    autocast = get_autocast(args.precision, device_type=device)
    input_dtype = get_input_dtype(args.precision)
    model.train()
    losses_m = {}

    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, batch in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum

        lr = scheduler[step]
        apply_optim_scheduler(optimizer, lr)

        imgs = batch
        imgs = imgs.to(device=device, dtype=input_dtype, non_blocking=True)

        data_time_m.update(time.time() - end)

        if (i % args.accum_freq) == 0:
            optimizer.zero_grad()
        
        with autocast():
            pred, mask = model(imgs)

            target = model.patchify(imgs)

            if args.norm_pix_loss:
                mean = target.mean(dim=-1, keepdim=True)
                var = target.var(dim=-1, keepdim=True)
                target = (target - mean) / (var + 1.e-6)**.5

            loss = ((pred - target) ** 2).mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
            loss = loss / args.accum_freq
            losses = {'reconstruction_loss': loss * args.accum_freq}
    
        backward(loss, scaler)


        if ((i+1) % args.accum_freq) == 0:
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

            batch_time_m.update(time.time() - end)
            end = time.time()
            batch_count = i_accum + 1

            if (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
                batch_size = len(imgs)
                num_samples = batch_count * batch_size * args.accum_freq
                samples_per_epoch = dataloader.num_samples
                percent_complete = 100.0 * batch_count / num_batches_per_epoch

                # NOTE loss is coarsely sampled, just master node and per log update
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
                samples_per_second = args.accum_freq * args.batch_size / batch_time_m.val
                samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} " + loss_log
                )

                # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "lr": optimizer.param_groups[0]["lr"],
                }            
                log_data.update({name:val.val for name, val in losses_m.items()})
                log_data.update({f"{name}_avg":val.avg for name, val in losses_m.items()})

                log_data = {"train/" + name: val for name, val in log_data.items()}

                
                if args.wandb:
                    assert wandb is not None, 'Please install wandb.'
                    log_data['step'] = step  # for backwards compatibility
                    args.step = step
                    wandb.log(log_data, step=step)
                
                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()

def evaluate(model, device, dataloader, epoch, step, args):
    metrics = {}
    autocast = get_autocast(args.precision, device_type=device)
    input_dtype = get_input_dtype(args.precision)
    model.eval()

    if args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs):
        num_samples = 0
        samples_per_val = dataloader.num_samples

        losses_m = {}
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                imgs = batch
                imgs = imgs.to(device=device, dtype=input_dtype, non_blocking=True)

                with autocast():
                    pred, mask = model(imgs)

                    target = model.patchify(imgs)

                    if args.norm_pix_loss:
                        mean = target.mean(dim=-1, keepdim=True)
                        var = target.var(dim=-1, keepdim=True)
                        target = (target - mean) / (var + 1.e-6)**.5

                    loss = ((pred - target) ** 2).mean(dim=-1)
                    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
                    loss = loss / args.accum_freq
                    losses = {'reconstruction_loss': loss * args.accum_freq}

                batch_size = imgs.shape[0]
                for key, val in losses.items():
                    if key not in losses_m:
                        losses_m[key] = AverageMeter()
                    losses_m[key].update(val.item(), batch_size)

                num_samples += batch_size

                if (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Recon Loss: {losses_m['reconstruction_loss'].avg:.6f}\t")
                    
                metrics.update({
                    "recon_loss": losses_m['reconstruction_loss'].avg, 
                    "epoch": epoch, "num_samples": num_samples
                 })
                
    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.6f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)



def main(args):
    t_now = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    args.logs = "mae_logs"
    args.output_path = os.path.join(args.logs, f"{t_now}_{args.model_name[:50]}")
    args.log_name = os.path.basename(args.output_path)
    args.checkpoint_path = os.path.join(args.output_path, "checkpoint")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    args.config_path = os.path.join(args.output_path, os.path.basename(args.config))
    if args.config: shutil.copy(args.config, args.config_path)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.log_path = os.path.join(args.output_path, 'out.log')
    if os.path.exists(args.log_path):
        print(
            "Error. Experiment already exists. Use --name {} to specify a new experiment."
        )
        return -1

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    random_seed(args.seed)

    scaler = None
    if args.precision == "amp":
        try:
            scaler = torch.amp.GradScaler(device=device)
        except (AttributeError, TypeError) as e:
            scaler = torch.cuda.amp.GradScaler()
    
    backbonre = timm.create_model(args.model_name, pretrained=True)
    model = MAE(backbonre).to(device)
    load_pretrain(model, args.pretrain)

    logging.info("Model:")
    logging.info(f"{str(model)}")
    logging.info("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        logging.info(f"  {name}: {val}")

    train_loader, valid_loader = make_mae_dataset(args)

    # Optimizer
    
    params_group = make_param_groups(model, len(backbonre.blocks), args.lr_decay, args.patch_embed_lr_mult)
    optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.wd, betas=(args.beta1, args.beta2), eps=1e-08)
    defaults = copy.deepcopy(optimizer.defaults)
    defaults['weight_decay'] = args.wd
    defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
    logging.info(
        f'Created {type(optimizer).__name__} ({args.opt_name}) optimizer: {defaults}'
    )
    logging.info(f"{optimizer}")

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


    logging.debug('Starting wandb.')

    if 'wandb' in args.report_to:
        args.wandb = True

    if args.wandb:
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

    scheduler = CosineScheduler(args.lr, args.min_lr, total_steps, args.warmup)
    print(f"## Using CosineScheduler \nwarmup steps: {args.warmup} \ntotal steps: {total_steps}\n")

    # Train
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, device, train_loader, epoch, optimizer, scaler, scheduler, args)
        completed_epoch = epoch + 1
        evaluate(model, device, valid_loader, completed_epoch, args.step, args)

        # print(f"Epoch {epoch}, Loss: {loss:.4f}")

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


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = read_config(args, parser)
    main(args)