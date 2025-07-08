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
# from torch.optim import AdamW
from bitsandbytes.optim import AdamW8bit
from torch.nn import MSELoss
from pyiqa.losses.iqa_losses import NiNLoss
from sklearn.metrics import f1_score

from pyiqa.archs.topiq_arch import CFANet
from pyiqa.archs.maniqa_arch import MANIQA

from mysrc.utils.data import tensor_to_pil_image
from mysrc.utils.quality_params import setup_parser
from mysrc.utils.config import read_config
from mysrc.utils.optim_utils import make_lwld_params_group
from mysrc.utils.scheduler import GroupCosineLR
from mysrc.utils.precision import get_autocast, get_cast_dtype, get_input_dtype
from mysrc.utils.dataset import make_prefer_dataset
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

        # Quality
        qual_mask = (labels[:, 0] >= start) & (labels[:, 0] < end)
        qual_error = np.abs(labels[:, 0][qual_mask] - preds[:, 0][qual_mask])

        if qual_error.shape[0] == 0:
            qual_acc = -1 # -1은 데이터가 없음을 의미 
        else:
            qual_acc = np.mean(qual_error < acc_range)

        # Preference
        prefer_mask = (labels[:, 1] >= start) & (labels[:, 1] < end)
        prefer_error = np.abs(labels[:, 1][prefer_mask] - preds[:, 1][prefer_mask])

        if prefer_error.shape[0] == 0:
            prefer_acc = -1 # -1은 데이터가 없음을 의미 
        else:
            prefer_acc = np.mean(prefer_error < acc_range)

        bins_dict[f"{start:.2f}-{end:.2f}_qual_acc"] = qual_acc
        bins_dict[f"{start:.2f}-{end:.2f}_prefer_acc"] = prefer_acc

    return bins_dict


def train_one_epoch(model, device, dataloader, mse_loss, nin_loss, epoch, optimizer, scaler, scheduler, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.train()
    num_batches_per_epoch = len(dataloader)
    
    mse_m = AverageMeter()
    nin_m = AverageMeter()
    total_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    pbar = tqdm(total=num_batches_per_epoch, desc=f"Training Epoch:{epoch+1}")
    for i, batch in enumerate(dataloader):
        step = epoch * num_batches_per_epoch + i
        scheduler(step)

        images, labels = batch
        images = images.to(device, dtype=input_dtype, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        breakpoint
        with autocast():
            # outputs = model(images, return_mos=False, return_dist=True)
            outputs = model(images)

        mse_cl_weight = torch.exp(args.mse_cl_weight * labels)
        mse_los = mse_cl_weight * mse_loss(outputs, labels)

        nin_los = nin_loss(outputs, labels)
        # total_loss = mse_los.mean()
        # total_loss = nin_los
        total_loss = args.mse_weight * mse_los.mean() + args.nin_weight * nin_los

        # if i == 0:
        #     for j in range(5):
        #         save_path = f"./train_sample/train_{j}.png"
        #         tensor_to_pil_image(images[j].to('cpu')).save(save_path)
        #         print(save_path)
        #         print(f"output({j}): {outputs[j].round(decimals=4)}")
        #         print(f"labels({j}): {torch.cat((labels[j], mse_labels[j]), dim=0).round(decimals=4)}")
        #         print(f"loss({j}): {torch.cat((bce_los[j], mse_los[j]), dim=0).round(decimals=4)}")
        #         print()

        backward(total_loss, scaler)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        batch_time_m.update(time.time() - end)

        bsz = len(images)
        mse_m.update(mse_los.mean().item(), bsz)
        nin_m.update(nin_los.item(), bsz)
        total_m.update(total_loss.item(), bsz)

        log_data = {
            "data_time": data_time_m.val,
            "batch_time": batch_time_m.val,
            "total_loss" : total_m.val,
            "total_loss" : total_m.avg,
            "mse_loss" : mse_m.val,
            "mse_avg" : mse_m.avg,
            "nin_loss" : nin_m.val,
            "nin_avg" : nin_m.avg,
            "lr": optimizer.param_groups[0]['lr'],
        }

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

def evaluate(model, device, valid_loader, mse_loss, nin_loss, epoch, step, args):
    autocast = get_autocast(args.precision)
    input_dtype = get_input_dtype(args.precision)

    model.eval()
    mse_m = AverageMeter()
    nin_m = AverageMeter()
    total_m = AverageMeter()

    all_preds = []
    all_labels = []
    with torch.inference_mode():
        pbar = tqdm(total=len(valid_loader), desc=f"Validation Current Epoch:{epoch}")
        for i, batch in enumerate(valid_loader):
            images, labels = batch
            images = images.to(device, dtype=input_dtype, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                # outputs = model(images, return_mos=False, return_dist=True)
                outputs = model(images)


            mse_los = mse_loss(outputs, labels)
            nin_los = nin_loss(outputs, labels)
            total_loss = mse_los.mean() + nin_los
            
            # if i == 0:
            #     for j in range(5):
            #         save_path = f"./train_sample/valid_{j}.png"
            #         tensor_to_pil_image(images[j].to('cpu')).save(save_path)
            #         print(save_path)
            #         print(f"output({j}): {outputs[j].round(decimals=4)}")
            #         print(f"labels({j}): {torch.cat((labels[j], mse_labels[j]), dim=0).round(decimals=4)}")
            #         print(f"loss({j}): {torch.cat((bce_los[j], mse_los[j]), dim=0).round(decimals=4)}")
            #         print()

            bsz = len(images)
            mse_m.update(mse_los.mean().item(), bsz)
            nin_m.update(nin_los.item(), bsz)
            total_m.update(total_loss.item(), bsz)

            preds = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

            # preds = (outputs.detach().cpu().numpy() * args.max_mos).round().astype(int)
            # preds = clip_preds(preds, args.max_mos)
            # preds = np.concatenate(preds, axis=1)
            # all_preds.extend(preds)

            # labels = (labels.detach().cpu().numpy() * args.max_mos).round().astype(int)
            # labels = [np.eye(args.max_mos[i]+1)[labels[:, i].reshape(-1)] for i in range(labels.shape[-1])]
            # labels = np.concatenate(labels, axis=1)
            # all_labels.extend(labels)
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
        print(f"## Evaluate")
        print(f"[Quality] acc5: {acc5[0]:.5f}, acc10: {acc10[0]:.5f}, acc15: {acc15[0]:.5f}, mae: {error.mean(axis=0)[0]:.5f}\n")
        print(f"[Preference] acc5: {acc5[1]:.5f}, acc10: {acc10[1]:.5f}, acc15: {acc15[1]:.5f}, mae: {error.mean(axis=0)[1]:.5f}\n")

        bins_dict = calc_range_accn(all_preds, all_labels, split_times=10, acc_range=0.05)

        log_data = {
            "total_avg" : total_m.avg,
            "mse_avg" : mse_m.avg,
            "nin_avg" : nin_m.avg,
            "quality_error": error.mean(axis=0)[0],
            "quality_acc5": acc5[0],
            "quality_acc10": acc10[0],
            "quality_acc15": acc15[0],
            "preference_error": error.mean(axis=0)[1],
            "preference_acc5": acc5[1],
            "preference_acc10": acc10[1],
            "preference_acc15": acc15[1],
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
    autodtype = get_cast_dtype(args.precision)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("## Load Dataset")
    train_loader, valid_loader = make_prefer_dataset(args, autodtype, device)

    print("## Model Create")
    model = timm.create_model(args.model_name, pretrained=True, num_classes=args.num_classes, img_size=(args.image_size), drop_path_rate=args.drop_path)

    # args.network.pop('type')
    # model = MANIQA(**args.network)
    # model = CFANet(**args.network)
    model.to(device=device, dtype=autodtype)

    # model = timm.create_model(args.model_name, pretrained=True, num_classes=args.num_classes, img_size=(args.image_size), drop_path_rate=args.drop_path)
    # model.to(device=device, dtype=autodtype)

    # params_group = make_lwld_params_group(model, args.lr, args.lr_decay, args.wd)
    params_group = [p for p in model.parameters() if p.requires_grad]

    optimizer = AdamW8bit(params_group, lr=args.lr, weight_decay=args.wd, betas=(args.beta1, args.beta2), eps=1e-08)
    # optimizer = AdamW(params_group, lr=args.lr, weight_decay=args.wd, betas=(args.beta1, args.beta2), eps=1e-08)
    # optimizer = AdamW(params_group, lr=args.lr, weight_decay=args.wd, betas=(args.beta1, args.beta2), eps=1e-08)
    mse_loss = MSELoss(reduction='none')
    nin_loss = NiNLoss(loss_weight=1.0, p=1, q=2)
    scaler = GradScaler() if args.precision == "amp" else None

    total_steps = (train_loader.num_batches) * args.epochs
    scheduler = GroupCosineLR(optimizer, args.lr, args.warmup, total_steps)

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
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)

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
        train_one_epoch(model, device, train_loader, mse_loss, nin_loss, epoch, optimizer, scaler, scheduler, args)
        completed_epoch = epoch + 1
        # args.step = 1
        evaluate(model, device, valid_loader, mse_loss, nin_loss, completed_epoch, args.step, args)

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
