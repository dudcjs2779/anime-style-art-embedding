import os
import random
import torch
import timm
import pandas as pd
import wandb
import time
import numpy as np
import shutil

from PIL import Image
from tqdm import tqdm
from datetime import datetime

from torch.cuda.amp import GradScaler
from torch.nn import MSELoss
from accelerate import Accelerator
from pyiqa.losses.iqa_losses import NiNLoss
from sklearn.metrics import f1_score

from mysrc.utils.data import tensor_to_pil_image
from mysrc.utils.quality_params import setup_parser
from mysrc.utils.config import read_config
from mysrc.utils.optim_utils import make_lwld_params_group
from mysrc.utils.scheduler import GroupCosineLR, CosineAnnealingWarmUpRestarts
from mysrc.utils.precision import get_autocast, get_cast_dtype, get_input_dtype
from mysrc.utils.dataset import make_mse_dataset02
from mysrc.utils.file_utils import pt_load

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

def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()

def clip_preds(preds, ranges):
    clip_preds = []
    for i in range(preds.shape[-1]):
        preds[:, i] = np.clip(preds[:, i], 0, ranges[i])
        clip_preds.append(np.eye(ranges[i]+1)[preds[:, i].reshape(-1)])

    return clip_preds

def calc_range_accn(preds, labels, split_times, acc_range=0.05):
    bins_dict = {}
    split_step = 1 / split_times
    for i in range(split_times):
        start = i * split_step
        end = start + split_step

        mask = (labels >= start) & (labels < end)
        error = np.abs(labels[mask] - preds[mask])

        # indices = np.where((labels >= start) & (labels < end))
        # error = np.abs(labels[indices] - preds[indices])
        
        if error.shape[0] == 0:
            acc = -1 # -1은 데이터가 없음을 의미 
        else:
            # acc = (error < acc_range).sum(axis=0) / error.shape[0]
            acc = np.mean(error < acc_range)

        bins_dict[f"{start:.2f}-{end:.2f}_acc"] = acc

    return bins_dict


def train_one_epoch(model, device, dataloader, accelerator, mse_loss, nin_loss, epoch, optimizer, scheduler, args):
    # autocast = get_autocast(args.precision)
    # input_dtype = get_input_dtype(args.precision)

    model.train()
    num_batches_per_epoch = len(dataloader)
    
    mse_m = AverageMeter()
    # nin_m = AverageMeter()
    total_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    pbar = tqdm(total=num_batches_per_epoch, desc=f"Training Epoch:{epoch+1}")
    for i, batch in enumerate(dataloader):
        step = epoch * num_batches_per_epoch + i

        if scheduler is not None:
            scheduler(step)

        images, labels = batch
        # images = images.to(device, non_blocking=True, dtype= torch.float16)
        # labels = labels.to(device, non_blocking=True, dtype= torch.float16)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        data_time_m.update(time.time() - end)

        # with autocast():
        with accelerator.accumulate(model):
            # breakpoint()
            outputs = model(images)

            if args.count_weight:
                bin_indices = torch.bucketize(labels * 100, args.bin_edges, right=True) - 1
                sample_weights = args.count_weights[bin_indices]

                mse_los = mse_loss(outputs, labels) * sample_weights
            else:
                # mse_cl_weight = torch.exp(args.mse_cl_weight * labels)
                # mse_los = mse_cl_weight * mse_loss(outputs, labels)
                mse_los = mse_loss(outputs, labels)

            # nin_los = nin_loss(outputs, labels)
            total_loss = mse_los.mean()
            # total_loss = nin_los
            # total_loss = args.mse_weight * mse_los.mean() + args.nin_weight * nin_los


            # if i == 0:
            #     for j in range(5):
            #         save_path = f"./train_sample/train_{j}.png"
            #         tensor_to_pil_image(images[j].to('cpu')).save(save_path)
            #         print(save_path)
            #         print(f"output({j}): {outputs[j].round(decimals=4)}")
            #         print(f"labels({j}): {labels[j].round(decimals=4)}")
            #         print(f"loss({j}): {mse_los[j].round(decimals=4)}")
            #         print()

            accelerator.backward(total_loss)
            # backward(total_loss, scaler)

            # if accelerator is None and scaler is not None:
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
            optimizer.step()
            optimizer.zero_grad()


        batch_time_m.update(time.time() - end)

        bsz = len(images)
        mse_m.update(mse_los.mean().item(), bsz)
        # nin_m.update(nin_los.item(), bsz)
        total_m.update(total_loss.item(), bsz)

        log_data = {
            "data_time": data_time_m.val,
            "batch_time": batch_time_m.val,
            "total_loss" : total_m.val,
            "total_avg" : total_m.avg,
            "mse_loss" : mse_m.val,
            "mse_avg" : mse_m.avg,
            # "nin_loss" : nin_m.val,
            # "nin_avg" : nin_m.avg,
            "lr": optimizer.param_groups[0]['lr'],
        }

        if optimizer.param_groups[0].get('d') is not None:
            log_data['lr_d'] = optimizer.param_groups[0].get('d')

        # for i, group in enumerate(optimizer.param_groups):
        #     key = f"lr/optim_group{i}"
        #     log_data[key] = group['lr']

        log_data['step'] = step
        log_data = {"train/" + name: val for name, val in log_data.items()}
        args.step = step
        wandb.log(log_data, step=step)
        end = time.time()

        pbar.postfix = f"Loss avg:{total_m.avg:.5f}, Loss batch:{total_m.val:.5f}" 
        pbar.update(1)
    pbar.close()

def evaluate(model, device, valid_loader, accelerator, mse_loss, nin_loss, epoch, step, args):
    # autocast = get_autocast(args.precision)
    # input_dtype = get_input_dtype(args.precision)

    model.eval()
    mse_m = AverageMeter()
    # nin_m = AverageMeter()
    total_m = AverageMeter()

    all_preds = []
    all_labels = []
    with torch.inference_mode():
        pbar = tqdm(total=len(valid_loader), desc=f"Validation Current Epoch:{epoch}")
        for i, batch in enumerate(valid_loader):
            images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # with autocast():
            with torch.inference_mode(), accelerator.autocast(): 
                outputs = model(images)

                mse_los = mse_loss(outputs, labels)
                # nin_los = nin_loss(outputs, labels)
                total_loss = mse_los.mean()
                # total_loss = mse_los.mean() + nin_los
            
            # if i == 0:
            #     for j in range(5):
            #         save_path = f"./train_sample/valid_{j}.png"
            #         tensor_to_pil_image(images[j].to('cpu')).save(save_path)
            #         print(save_path)
            #         print(f"output({j}): {outputs[j].round(decimals=4)}")
            #         print(f"labels({j}): {labels[j].round(decimals=4)}")
            #         print(f"loss({j}): {mse_los[j].round(decimals=4)}")
            #         print()

            bsz = len(images)
            mse_m.update(mse_los.mean().item(), bsz)
            # nin_m.update(nin_los.item(), bsz)
            total_m.update(total_loss.item(), bsz)

            preds = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

            pbar.update(1)
        pbar.close()

        all_preds = np.stack(all_preds)
        all_labels = np.stack(all_labels)

        error = np.abs(all_labels - all_preds)
        acc5 = (error < 0.05).sum(axis=0) / all_labels.shape[0]     # 5% 에러, 100점이라면 5점까지 에러허용범위
        acc10 = (error < 0.1).sum(axis=0) / all_labels.shape[0]
        acc15 = (error < 0.15).sum(axis=0) / all_labels.shape[0]

        # accs = (all_preds == all_labels).sum(axis=0) / all_labels.shape[0]
        # f1 = f1_score(all_labels, all_preds, average=None)
        print(f"## Evaluate\n acc5: {acc5.mean():.5f}, acc10: {acc10.mean():.5f}, acc15: {acc15.mean():.5f}, error: {error.mean():.5f}, loss: {total_m.avg:.5f}\n")

        bins_dict = calc_range_accn(all_preds, all_labels, split_times=10, acc_range=0.05)

        log_data = {
            "total_avg" : total_m.avg,
            "mse_avg" : mse_m.avg,
            # "nin_avg" : nin_m.avg,
            "mae_avg" : error.mean(),
            "acc5": acc5.mean(),
            "acc10": acc10.mean(),
            "acc15": acc15.mean(),
            # "f1": f1.mean(),
        }
        # log_data.update({f"acc_{i}": val for i, val in enumerate(accs)})
        # log_data.update({f"f1_{i}": val for i, val in enumerate(f1)})
        log_data.update(bins_dict)
        log_data['epoch'] = epoch
        log_data = {"valid/" + name: val for name, val in log_data.items()}

        wandb.log(log_data, step=step)

def main(args):
    t_now = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    log_path = "./logs"
    output_path = os.path.join(log_path, f"{t_now}_{args.model_name[:50]}")
    args.log_name = os.path.basename(output_path)
    checkpoint_path = os.path.join(output_path, "checkpoint")
    os.makedirs(checkpoint_path, exist_ok=True)
    if args.config: shutil.copy(args.config, os.path.join(output_path, os.path.basename(args.config)))

    random_seed(args.seed)
    # autodtype = get_cast_dtype(args.precision)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("## Load Dataset")
    train_loader, valid_loader = make_mse_dataset02(args, device)

    print("## Model Create\n")
    model = timm.create_model(args.model_name, pretrained=True, num_classes=args.num_classes, img_size=(args.image_size), drop_path_rate=args.drop_path)
    model.to(device=device)
    # breakpoint()

    if args.pretrain is not None:
        checkpoint = pt_load(args.pretrain, map_location='cpu')
        model.load_state_dict(checkpoint["state_dict"])
        print(f"## Pretrained model is loaded: {args.pretrain}\n")
        del checkpoint

    # args.network.pop('type')
    # # model = MANIQA(**args.network)
    # model = CFANet(**args.network)
    # model.to(device=device, dtype=autodtype)

    # params_group = make_lwld_params_group(model, args.lr, args.lr_decay, args.wd)
    params_group = [p for p in model.parameters() if p.requires_grad]

    if args.opt_name == "adamw8bit":
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(params_group, lr=args.lr, weight_decay=args.wd, betas=(args.beta1, args.beta2), eps=1e-08)    # float16 학습을 위함

    elif args.opt_name == "adamw":
        from torch.optim import AdamW
        optimizer = AdamW(params_group, lr=args.lr, weight_decay=args.wd, betas=(args.beta1, args.beta2), eps=1e-08)    # float16으로 학습하지 말것! 모델이 nan을 출력함.

    elif args.opt_name == "dadaptation":
        from dadaptation import DAdaptAdam
        optimizer = DAdaptAdam(params_group, lr=1.0)

    elif args.opt_name == "prodigy":
        from prodigyopt import Prodigy
        optimizer = Prodigy(params_group, lr=1., weight_decay=args.wd, slice_p=11)

    print(f"## Selected Opimizer: {args.opt_name}\n")
    
    mse_loss = MSELoss(reduction='none')
    nin_loss = NiNLoss(loss_weight=1.0, p=1, q=2)
    # scaler = GradScaler() if args.precision == "amp" else None

    total_steps = (train_loader.num_batches) * args.epochs

    ## Learning Rate Scheduler
    if args.opt_name == "dadaptation":
        scheduler = None
        print(f"## Using DAdaptation optimizer, {args.sch_name} scheduler is ignored and scheduler is replaced by None\n")
    else:
        if args.sch_name == "cos":
            warmup_steps = total_steps * args.warmup_ratio if args.warmup_ratio is not None else args.warmup
            scheduler = GroupCosineLR(optimizer, args.lr, args.min_lr, warmup_steps, total_steps)
            print(f"## Using GroupCosineLR \nwarmup steps: {warmup_steps} \ntotal steps: {total_steps}\n")
        elif args.sch_name == "cos_restart":
            cycle_step =  total_steps // args.cycles
            warmup_steps = int(total_steps // args.cycles * args.warmup_ratio) if args.warmup_ratio is not None else args.warmup
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, cycle_step=cycle_step, cy_mult=args.cycle_multiplier, lr_max=args.lr, lr_min=args.min_lr,  warmup=warmup_steps, gamma=args.re_g)
            print(f"## Using CosineAnnealingWarmUpRestarts \nwarmup steps: {warmup_steps} \ncycle_steps: {cycle_step} \ntotal steps: {total_steps} \ncycles: {args.cycles} \ncycle_multiplier: {args.cycle_multiplier}, \nre_g: {args.re_g}")
        
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            load_random_state(checkpoint)
            model.load_state_dict(checkpoint["state_dict"])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            # if scaler is not None and 'scaler' in checkpoint:
            #     scaler.load_state_dict(checkpoint['scaler'])
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)

    # breakpoint()
    accelerator = Accelerator(gradient_accumulation_steps=args.accum_freq, mixed_precision=args.precision)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(model, optimizer, train_loader, valid_loader)

    print("\n## Config info:")
    for k, v in vars(args).items(): print(f"{k}: {v}")
    print()

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

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, device, train_loader, accelerator, mse_loss, nin_loss, epoch, optimizer, scheduler, args)
        completed_epoch = epoch + 1
        # args.step = 1
        evaluate(model, device, valid_loader, accelerator, mse_loss, nin_loss, completed_epoch, args.step, args)

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

        # if scaler is not None:
        #     checkpoint_dict["scaler"] = scaler.state_dict()

        if completed_epoch == args.epochs or (
            args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
        ):
            torch.save(
                checkpoint_dict,
                os.path.join(checkpoint_path, f"epoch_{completed_epoch}.pt"),
            )

    wandb.finish()


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    args = read_config(args, parser)
    main(args)
