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

import torch.utils.tensorboard as tensorboard

from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from mysrc.utils.pretrain_params import setup_parser
from mysrc.utils.config import read_config
from mysrc.utils.dataset import make_pretrain_dataset
from mysrc.utils.optim_utils import make_param_groups
from mysrc.utils.loss import SigLipLoss
from mysrc.utils.scheduler import CosineScheduler
from mysrc.utils.precision import get_autocast, get_input_dtype
from mysrc.model import CustomVisual
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


def train_one_epoch(model, device, dataloader, clip_loss, epoch, optimizer, scaler, scheduler, tb_writer, args):
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    model.train()

    num_batches_per_epoch = dataloader.num_batches // args.accum_freq
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    if args.accum_freq > 1:
        accum_images1, accum_images2, accum_classes, accum_times, accum_features = [], [], [], [], {}

    losses_m = {}
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

        images1, images2, classes, times = batch['image1'], batch['image2'], batch['label'], batch['time']
        images1 = images1.to(device=device, dtype=input_dtype, non_blocking=True)
        images2 = images2.to(device=device, dtype=input_dtype, non_blocking=True)
        classes = classes.to(device=device, dtype=input_dtype, non_blocking=True)
        times = times.to(device=device, dtype=input_dtype, non_blocking=True)

        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                model_out = model(images1, images2)
                logit_scale = model_out["logit_scale"]
                losses = clip_loss(**model_out, classes=classes, same_class_weight=args.same_class_weight, 
                                   times=times, time_range=args.time_range, time_weight=args.time_weight,
                                   output_dict=True)

                total_loss = sum(losses.values())
                losses["loss"] = total_loss

            backward(total_loss, scaler)
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast():
                    model_out = model(images1, images2)

                    for f in ("logit_scale", "logit_bias"):
                        model_out.pop(f, None)

                    for key, val in model_out.items():
                        if key in accum_features:
                            accum_features[key].append(val)
                        else:
                            accum_features[key] = [val]
                    
                accum_images1.append(images1)
                accum_images2.append(images2)
                accum_classes.append(classes)
                accum_times.append(times)

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
                images1 = accum_images1[j]
                images2 = accum_images2[j]
                with autocast():
                    model_out = model(images1, images2)

                    inputs_no_accum = {}
                    inputs_no_accum["logit_scale"] = logit_scale = model_out.pop("logit_scale")
                    if "logit_bias" in model_out:
                        inputs_no_accum["logit_bias"] = model_out.pop("logit_bias")

                    inputs = {}
                    for key, val in accum_features.items():
                        accumulated = accum_features[key]
                        inputs[key] = torch.cat(accumulated[:j] + [model_out[key]] + accumulated[j + 1:])

                    losses = clip_loss(**inputs, **inputs_no_accum, classes=accum_classes, same_class_weight=args.same_class_weight, 
                                       times=accum_times, time_range=args.time_range, time_weight=args.time_weight,
                                       output_dict=True)

                    del inputs
                    del inputs_no_accum
                    total_loss = sum(losses.values())
                    losses["loss"] = total_loss

                backward(total_loss, scaler)


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
            accum_images1, accum_images2, accum_classes, accum_times, accum_features = [], [], [], [], {}

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i_accum + 1

        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            batch_size = len(images1)
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


def evaluate(model, device, dataloader, clip_loss, epoch, tb_writer, step, args):
    metrics = {}
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)
    model.eval()

    if args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs):
        num_samples = 0
        samples_per_val = dataloader.num_samples

        cumulative_loss = 0.0
        all_image_features, all_classes = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images1, images2, images3, classes, times = batch['image1'], batch['image2'], batch['image3'], batch['label'], batch['time']
                images1 = images1.to(device=device, dtype=input_dtype, non_blocking=True)   # aug1
                images2 = images2.to(device=device, dtype=input_dtype, non_blocking=True)   # aug2
                images3 = images3.to(device=device, dtype=input_dtype, non_blocking=True)   # non aug
                classes = classes.to(device=device, dtype=input_dtype, non_blocking=True)
                times = times.to(device=device, dtype=input_dtype, non_blocking=True)

                with autocast():
                    model_out = model.evaluate(images1, images2, images3)
                    image_features3 = model_out["image_features3"].clone()
                    logit_scale = model_out["logit_scale"]
                    logit_bias = model_out.get('logti_bias', None)
                    
                    batch_size = images1.shape[0]
                    all_image_features.append(image_features3.cpu())
                    all_classes.append(classes.cpu())
                    losses = clip_loss(
                        image_features1=model_out["image_features1"], 
                        image_features2=model_out["image_features2"],
                        logit_scale=model_out["logit_scale"], 
                        logit_bias=model_out.get('logit_bias', None), 
                        classes=classes, 
                        times=times, time_range=args.time_range, time_weight=args.time_weight,
                        same_class_weight=args.same_class_weight, 
                    )

                cumulative_loss += losses * batch_size
                num_samples += batch_size

                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")
            
            if logit_bias:
                logit_bias = logit_bias.cpu()

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                classes=torch.cat(all_classes),
                logit_scale=logit_scale.cpu(),
                logit_bias=logit_bias,
            )

            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
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
       
            
def get_clip_metrics(image_features, classes, logit_scale, logit_bias=None):
    metrics = {}

    ## 분류 accuracy
    # logits_image = (logit_scale * image_features @ image_features.t()).detach().cpu()
    logits_image = (image_features @ image_features.t()).detach().cpu()
    # if logit_bias:
    #     logits_image += logit_bias

    ground_truth = (classes.unsqueeze(0) == classes.unsqueeze(1)).int()
    N = logits_image.size(0)

    # 자기 자신 제외
    logits_no_diag = logits_image.clone()
    logits_no_diag.fill_diagonal_(-float("inf"))

    ## 클래스 별 총 데이터수 만큼의 top-k 랭크에서 얼마나 해당 class를 상위에 랭크시켰는지 accuracy 계산
    sorted_idx = logits_no_diag.argsort(dim=1, descending=True)  # 각 row에서 similarity 순으로 정렬한 index
    gt_mask = ground_truth.gather(1, sorted_idx)  # ground_truth 요소를 sorted_idx의 순서대로 재배열 즉, 높은 sim에 해당하는 ground_truth값이 앞으로 배치됨

    num_gt_per_row = ground_truth.sum(dim=1) # 각 샘플별 gt의 갯수
    ranks = torch.arange(N, device=logits_image.device).unsqueeze(0)

    rank_range = (ranks < num_gt_per_row.unsqueeze(1)) # (존재하는 클래스 갯수 만큼 앞부분을 True)
    rank_range_x1_5 = (ranks < (num_gt_per_row * 1.5).to(dtype=int).unsqueeze(1))
    rank_range_x3 = (ranks < (num_gt_per_row * 3).to(dtype=int).unsqueeze(1))
    rank_range_x10 = (ranks < (num_gt_per_row * 10).to(dtype=int).unsqueeze(1))

    acc_per_row = rank_range & (gt_mask.bool()) # (측정 랭킹 구간) & (sim이 높은 순으로 정렬한 gt_mask)
    acc_per_row_x1_5 = rank_range_x1_5 & (gt_mask.bool())
    acc_per_row_x3 = rank_range_x3 & (gt_mask.bool())
    acc_per_row_x10 = rank_range_x10 & (gt_mask.bool())

    accuracy = acc_per_row.float().sum(dim=1) / num_gt_per_row
    accuracy_x1_5 = acc_per_row_x1_5.float().sum(dim=1) / num_gt_per_row
    accuracy_x3 = acc_per_row_x3.float().sum(dim=1) / num_gt_per_row
    accuracy_x10 = acc_per_row_x10.float().sum(dim=1) / num_gt_per_row

    accuracy = accuracy.mean().item()
    accuracy_x1_5 = accuracy_x1_5.mean().item()
    accuracy_x3 = accuracy_x3.mean().item()
    accuracy_x10 = accuracy_x10.mean().item()

    metrics[f"accuracy"] = accuracy
    metrics[f"accuracy_x1_5"] = accuracy_x1_5
    metrics[f"accuracy_x3"] = accuracy_x3
    metrics[f"accuracy_x10"] = accuracy_x10

    ## 클래스별 평균 유사도 및 표준편차
    image_features = image_features.numpy()
    classes = classes.numpy()
    unique_classes = np.unique(classes)

    mean_list = []
    std_list = []
    for i in unique_classes:
        class_features = image_features[classes == i]
        class_sims = class_features @ class_features.T
        np.fill_diagonal(class_sims, np.nan)

        sims_mean = np.nanmean(class_sims)
        mean_list.append(sims_mean)

        stds = np.nanstd(class_sims, axis=-1)
        stds_mean = np.nanmean(stds)
        std_list.append(stds_mean)

    metrics[f"sim_mean_by_class"] = np.mean(mean_list)
    metrics[f"sim_stu_mean_by_class"] = np.mean(std_list)

    ## silhouette_score
    metrics[f"silhouette_score_real_class"] = silhouette_score(image_features, classes, metric='cosine')

    kmeans = KMeans(n_clusters=len(unique_classes), random_state=42)
    labels = kmeans.fit_predict(image_features)
    metrics[f"silhouette_score_kmeans"] = silhouette_score(image_features, labels, metric='cosine')

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
    args.logs = "./logs"
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
    train_loader, valid_loader = make_pretrain_dataset(args)

    print("## Model Create\n")
    kargs = {
        'num_classes': args.embed_dim,
        "global_pool": args.global_pool,
    }
    model = CustomVisual(model_name="eva02_base_patch14_224.mim_in22k", init_logit_scale=math.log(10), init_logit_bias=-10, output_dict=True, model_kargs=kargs)
    model.to(device=device)

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")

    params_group = make_param_groups(model, args.lr_decay, args.patch_embed_lr_mult)
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

    if is_master(args):
            defaults = copy.deepcopy(optimizer.defaults)
            defaults['weight_decay'] = args.wd
            defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
            logging.info(
                f'Created {type(optimizer).__name__} ({args.opt_name}) optimizer: {defaults}'
            )

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
        logging.debug('Finished loading wandb.')


    total_steps = (train_loader.num_batches // args.accum_freq) * args.epochs

    warmup_steps = total_steps * args.warmup_ratio if args.warmup_ratio is not None else args.warmup
    scheduler = CosineScheduler(args.lr, args.min_lr, total_steps, warmup_steps)
    print(f"## Using CosineScheduler \nwarmup steps: {warmup_steps} \ntotal steps: {total_steps}\n")

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, device, train_loader, clip_loss, epoch, optimizer, scaler, scheduler, writer, args)
        completed_epoch = epoch + 1
        metrics = evaluate(model, device, valid_loader, clip_loss, completed_epoch, writer, args.step, args)

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