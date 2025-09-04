import logging
import pandas as pd
import cv2
import math
import json
import torch
import numpy as np
import random

from collections import defaultdict
from copy import deepcopy
import albumentations as A
from functools import partial
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mysrc.utils.aug_preset import get_preset

def balanced_sampleing(df, target_key):
    print("## Balanced sampling")
    df['score'] = df[target_key].apply(lambda x: x[0])

    # 각 점수대별 데이터 개수 확인
    bins = list(range(0, 111, 10))
    df['score_bin'] = pd.cut(df['score'], bins=bins, right=False)
    count_by_bin = df['score_bin'].value_counts().sort_index()
    max_count = count_by_bin.max()
    
    print(f"Max count: {max_count}\n")
    print(f"## Before count by score: \n {count_by_bin}")
    balanced_dfs = []

    # 각 점수대별로 max_count에 맞춰서 샘플링
    for s_range, count in count_by_bin.items():

        range_df = df[(df['score'] >= s_range.left) & (df['score'] < s_range.right)]
        num_needed = max_count - count

        if num_needed == 0:
            balanced_dfs.append(range_df)
            continue

        n_repeat = num_needed // count
        n_extra = num_needed % count

        df_repeated = pd.concat([range_df] * n_repeat, ignore_index=True) if n_repeat > 0 else pd.DataFrame()
        df_extra = range_df.sample(n=n_extra, replace=False, random_state=42)

        balanced_range_df = pd.concat([range_df, df_repeated, df_extra], ignore_index=True)
        balanced_dfs.append(balanced_range_df)

    df_balanced = pd.concat(balanced_dfs, ignore_index=True)

    count_by_bin = df_balanced['score_bin'].value_counts().sort_index()
    del df_balanced['score_bin']

    print(f"## After count by score: \n {count_by_bin}")

    return df_balanced

def pretrain_collate(batch, mask_ratio_tuple, mask_probability, n_tokens=None, mask_generator=None):
    output = {}
    for key in batch[0].keys():
        output[key] = torch.stack([data[key] for data in batch])

    B = len(output['image1'])
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)
    
    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    output['collated_masks'] = collated_masks
    output['mask_indices_list'] = mask_indices_list
    output['masks_weight'] = masks_weight
    output['upperbound'] = upperbound
    output['n_masked_patches'] = torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),

    return output


def make_pretrain_dataset(args):
    train_df = pd.read_parquet(args.train)
    valid_df = pd.read_parquet(args.valid)
    print(f"Total Data: {len(train_df) + len(valid_df)}, Train: {len(train_df)}, Valid: {len(valid_df)}")

    transforms = get_preset(args.train_aug, image_size=args.image_size)
    valid_trans = get_preset(args.valid_aug, image_size=args.image_size)

    logging.info(f"Train Image Processor:")
    logging.info(f"{transforms}")
    logging.info(f"Valid Image Processor: ")
    logging.info(f"{valid_trans}")

    img_size = args.image_size[0]
    patch_size = args.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )

    collate_fn = partial(
        pretrain_collate,
        mask_ratio_tuple=(0.1, 0.5),
        mask_probability=0.5,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
    )

    train_labels = torch.tensor(train_df.target.tolist(), dtype=torch.float32, device='cpu')
    train_times = torch.tensor(train_df.years_passed.tolist(), dtype=torch.float32, device='cpu')
    train_dataset = PretrainDataset(train_df.image.tolist(), train_labels, transforms, times=train_times)
    sampler = CustomBatchSampler(train_dataset, batch_size=args.batch_size, num_postive=args.num_postive, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=collate_fn, pin_memory=True, num_workers=args.workers)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fn, pin_memory=True, num_workers=args.workers)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    valid_labels = torch.tensor(valid_df.target.tolist(), dtype=torch.float32, device='cpu')
    valid_times = torch.tensor(valid_df.years_passed.tolist(), dtype=torch.float32, device='cpu')
    valid_dataset = PretrainValidDataset(valid_df.image.tolist(), valid_labels, transforms, valid_trans, times=valid_times)
    valid_loader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    valid_loader.num_samples = len(valid_dataset)
    valid_loader.num_batches = len(valid_loader)

    return train_loader, valid_loader


def make_dataset(args):
    df = pd.read_parquet(args.train)

    encoder = LabelEncoder()
    df['target'] = encoder.fit_transform(df[args.label_key])

    train_df, valid_df = train_test_split(df, test_size=args.valid_size, stratify=df[args.label_key], random_state=42)

    print(f"Total Data: {len(df)}, Train: {len(train_df)}, Valid: {len(valid_df)}")

    args.num_classes = df['target'].nunique()

    transforms = get_preset(args.train_aug, image_size=args.image_size)
    valid_trans = get_preset(args.valid_aug, image_size=args.image_size)
    
    train_dataset = TrainDataset(train_df.image.tolist(), train_df.target.tolist(), transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    valid_dataset = TrainDataset(valid_df.image.tolist(), valid_df.target.tolist(), valid_trans)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    valid_loader.num_samples = len(valid_dataset)
    valid_loader.num_batches = len(valid_loader)

    return train_loader, valid_loader

def make_multilabel_dataset(args, autodtype, device, set_df=False):
    df = pd.read_parquet(args.train)

    # train_test_split 함수의 stratify인자를 활용하기 위해 빈도수 1개인 클래스를 제외
    df['target_str'] = df[args.target_key].astype(str)
    value_counts = df.value_counts(subset='target_str')

    single_sample_classes = value_counts[value_counts == 1].index
    multi_sample_classes = value_counts[value_counts > 1].index

    single_sample_df = df[df['target_str'].isin(single_sample_classes)]
    multi_sample_df = df[df['target_str'].isin(multi_sample_classes)]
    train_df, valid_df = train_test_split(multi_sample_df, test_size=args.valid_size, stratify=multi_sample_df['target_str'], random_state=42)
    train_df = pd.concat([train_df, single_sample_df])  # 빈도수가 1인 라벨은 train에 포함
    del df['target_str'], train_df['target_str'], valid_df['target_str']

    print(f"Total Data: {len(df)}, Train: {len(train_df)}, Valid: {len(valid_df)}\n")
    args.num_classes = len(df.loc[0, args.target_key])

    # Loss function weight
    # class weight
    class_counts = torch.tensor(train_df[args.target_key].sum(), dtype=autodtype, device=device)
    class_weights = 1.0 / torch.log1p(class_counts + 2)
    # class_weights = 1.0 / class_counts
    args.class_weight_val = (class_weights / class_weights.mean()) ** args.class_weight
    if args.label_json:
        with open(args.label_json, 'r') as f:
            class_list = json.load(f)

    # pos weight
    label_count = torch.tensor(len(train_df) * args.num_classes, dtype=autodtype, device=device)
    args.pos_weight_val = (label_count - class_counts.sum()) / class_counts.sum() * args.pos_weight

    print(f"## Class weight")
    if args.label_json: print(f"Class: {class_list}")
    print(f"Weight: {[weight.item() for weight in args.class_weight_val]}")
    print(f"Pos Weight: {[args.pos_weight_val.item()]}")


    transforms = get_preset(args.train_aug)
    valid_trans = get_preset(args.valid_aug)
    
    train_labels = torch.tensor(train_df[args.target_key].tolist(), dtype=autodtype, device='cpu')
    train_dataset = MLDataset(train_df[args.image_key].tolist(), train_labels, transforms, train_df if set_df else None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    valid_labels = torch.tensor(valid_df[args.target_key].tolist(), dtype=autodtype, device='cpu')
    valid_dataset = MLDataset(valid_df[args.image_key].tolist(), valid_labels, valid_trans, valid_df if set_df else None)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    valid_loader.num_samples = len(valid_dataset)
    valid_loader.num_batches = len(valid_loader)

    return train_loader, valid_loader

def mse_label_to_binary(mse_labels:np.array) -> np.array:
    # 이진 라벨과 연속형 라벨을 이진 형식으로 합칩니다.
    # mse 라벨은 정수형 라벨이여야합니다.
    max_ranges = mse_labels.max(axis=0).astype(int)

    bin_mse_labels = [[0] * mse_labels.shape[0] for _ in range(mse_labels.shape[-1])]
    for i, mse_label in enumerate(mse_labels):
        for j, label_num in enumerate(mse_label):
            label_num = label_num.astype(int)
            bin_label = np.zeros(max_ranges[j] + 1, dtype=np.float32)
            bin_label[label_num] = 1

            bin_mse_labels[j][i] = bin_label
            # print(np.arange(max_num))

    for i, bin_mse_label in enumerate(bin_mse_labels):
        bin_mse_labels[i] = np.stack(bin_mse_label)

    bin_mse_labels = np.concatenate(bin_mse_labels, axis=-1)

    return bin_mse_labels

def combine_binary_and_mse(bin_labels:np.array, mse_labels:np.array) -> np.array:
    # 이진 라벨과 연속형 라벨을 이진 형식으로 합칩니다.
    # mse 라벨은 정수형 라벨이여야합니다.
    bin_mse_labels = mse_label_to_binary(mse_labels)
    combined = np.concatenate([bin_labels, bin_mse_labels], axis=-1)

    return combined

def make_mse_dataset(args, autodtype, device, set_df=False):
    df = pd.read_parquet(args.train)
    args.min_mos = min(args.mos_range)
    args.max_mos = max(args.mos_range)

    mse_labels = np.stack(df[args.target_key].values)

    bin_mse_label = mse_label_to_binary(mse_labels)
    df['bin_mse_label'] = bin_mse_label.tolist()

    # 점수를 기반으로 split
    df['target_str'] = df['bin_mse_label'].astype(str)
    value_counts = df.value_counts(subset='target_str')

    # train_test_split 함수의 stratify인자를 활용하기 위해 빈도수 1개인 클래스를 제외
    single_sample_classes = value_counts[value_counts == 1].index
    multi_sample_classes = value_counts[value_counts > 1].index

    single_sample_df = df[df['target_str'].isin(single_sample_classes)]
    multi_sample_df = df[df['target_str'].isin(multi_sample_classes)]
    train_df, valid_df = train_test_split(multi_sample_df, test_size=args.valid_size, stratify=multi_sample_df['target_str'], random_state=42)
    train_df = pd.concat([train_df, single_sample_df])  # 빈도수가 1인 라벨은 train에 포함
    del df['target_str'], train_df['target_str'], valid_df['target_str'], df['bin_mse_label'], train_df['bin_mse_label'], valid_df['bin_mse_label']

    print(f"Total Data: {len(df)}, Train: {len(train_df)}, Valid: {len(valid_df)}\n")
    args.num_classes = len(df.loc[0, args.target_key])

    transforms = get_preset(args.train_aug, image_size=args.image_size)
    valid_trans = get_preset(args.valid_aug, image_size=args.image_size)
    
    train_labels = (torch.tensor(train_df[args.target_key].tolist(), dtype=autodtype, device='cpu') / args.max_mos).to(dtype=autodtype)
    train_dataset = MSEDataset(train_df[args.image_key].tolist(), train_labels, transforms, train_df if set_df else None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    valid_labels = (torch.tensor(valid_df[args.target_key].tolist(), dtype=autodtype, device='cpu') / args.max_mos).to(dtype=autodtype)
    valid_dataset = MSEDataset(valid_df[args.image_key].tolist(), valid_labels, valid_trans, valid_df if set_df else None)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    valid_loader.num_samples = len(valid_dataset)
    valid_loader.num_batches = len(valid_loader)

    return train_loader, valid_loader

def make_mse_dataset02(args, device, set_df=False):
    df = pd.read_parquet(args.train)

    args.min_mos = args.mos_range[0]
    args.max_mos = args.mos_range[1]

    # count_weights 계산
    if args.count_weight:
        all_labels = torch.tensor(df['mse_label'].apply(lambda x: x[0]).values)

        # 구간별 빈도 계산
        args.bin_edges = torch.arange(args.min_mos, args.max_mos+10+1, (args.max_mos+10) // args.cw_bins).to(device)
        bin_counts = torch.histc(all_labels.float(), bins=args.cw_bins, min=args.min_mos, max=args.max_mos+10)  # +10은 max값이 100일 때 마지막 구간을 포함하기 위함
        bin_counts[bin_counts < (bin_counts.mean() * 0.1)] = bin_counts.mean() * 0.1    # 빈도수가 너무 적은 구간은 평균의 10%로 대체

        # inverse frequency weight (예: 빈도수가 적으면 weight가 크도록)
        bin_weights = 1.0 / (bin_counts)
        bin_weights = bin_weights ** args.cw_alpha
        bin_weights = bin_weights / bin_weights.mean()
        args.count_weights = bin_weights.to(device=device)

        print("## Count Weights")
        print("Bin counts:", bin_counts)
        print("count_weights:", args.count_weights)

    # tag를 기반으로 split(ex: 작가별)
    value_counts = df.value_counts(subset='tag')

    # train_test_split 함수의 stratify인자를 활용하기 위해 빈도수 1개인 클래스를 제외
    single_sample_classes = value_counts[value_counts == 1].index
    multi_sample_classes = value_counts[value_counts > 1].index

    single_sample_df = df[df['tag'].isin(single_sample_classes)]
    multi_sample_df = df[df['tag'].isin(multi_sample_classes)]
    train_df, valid_df = train_test_split(multi_sample_df, test_size=args.valid_size, stratify=multi_sample_df['tag'], random_state=42)
    train_df = pd.concat([train_df, single_sample_df])  # 빈도수가 1인 라벨은 train에 포함

    print(f"Total Data: {len(df)}, Train: {len(train_df)}, Valid: {len(valid_df)}\n")
    args.num_classes = len(df.loc[0, args.target_key])

    if args.balanced_sample:
        train_df = balanced_sampleing(train_df, args.target_key)

    print(f"After resample total Data: {len(df)}, Train: {len(train_df)}, Valid: {len(valid_df)}\n")
    
    transforms = get_preset(args.train_aug, image_size=args.image_size)
    valid_trans = get_preset(args.valid_aug, image_size=args.image_size)

    # train_df[args.target_key].tolist() 이 코드가 여러개의 np.array로 이루어진 list인데 이걸 먼저 np로 합쳐준뒤 tensor로 변환해야 빠른 변환이 가능하다고 함. 수정필요. 
    train_labels = torch.tensor(train_df[args.target_key].tolist(), dtype=torch.float32)
    train_labels = (train_labels - args.min_mos) / (args.max_mos - args.min_mos)

    train_dataset = MSEDataset(train_df[args.image_key].tolist(), train_labels, transforms, train_df if set_df else None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    valid_labels = torch.tensor(valid_df[args.target_key].tolist(), dtype=torch.float32)
    valid_labels = (valid_labels - args.min_mos) / (args.max_mos - args.min_mos)
    
    valid_dataset = MSEDataset(valid_df[args.image_key].tolist(), valid_labels, valid_trans, valid_df if set_df else None)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    valid_loader.num_samples = len(valid_dataset)
    valid_loader.num_batches = len(valid_loader)

    return train_loader, valid_loader

def make_prefer_dataset(args, autodtype, device, set_df=False):
    df = pd.read_parquet(args.train)

    # args.min_mos = np.array(args.mos_range)[:, 0]
    # args.max_mos = np.array(args.mos_range)[:, 1]
    args.mos_range = [[0, 100], [25, 75]]
    args.min_mos = np.array(args.mos_range)[:, 0]
    args.max_mos = np.array(args.mos_range)[:, 1]

    # tag를 기반으로 split(ex: 작가별)
    value_counts = df.value_counts(subset='tag')

    # train_test_split 함수의 stratify인자를 활용하기 위해 빈도수 1개인 클래스를 제외
    single_sample_classes = value_counts[value_counts == 1].index
    multi_sample_classes = value_counts[value_counts > 1].index

    single_sample_df = df[df['tag'].isin(single_sample_classes)]
    multi_sample_df = df[df['tag'].isin(multi_sample_classes)]
    train_df, valid_df = train_test_split(multi_sample_df, test_size=args.valid_size, stratify=multi_sample_df['tag'], random_state=42)
    train_df = pd.concat([train_df, single_sample_df])  # 빈도수가 1인 라벨은 train에 포함

    print(f"Total Data: {len(df)}, Train: {len(train_df)}, Valid: {len(valid_df)}\n")
    args.num_classes = len(df.loc[0, args.target_key])

    transforms = get_preset(args.train_aug, image_size=args.image_size)
    valid_trans = get_preset(args.valid_aug, image_size=args.image_size)

    # Train dataset
    tr_labels = torch.tensor(train_df[args.target_key].tolist(), dtype=autodtype)
    tr_labels = (tr_labels - args.min_mos) / (args.max_mos - args.min_mos)
    tr_labels = tr_labels.to(dtype=autodtype)

    train_dataset = MSEDataset(train_df[args.image_key].tolist(), tr_labels, transforms, train_df if set_df else None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    # Valid dataset
    val_labels = torch.tensor(valid_df[args.target_key].tolist(), dtype=autodtype)
    val_labels = (val_labels - args.min_mos) / (args.max_mos - args.min_mos)
    val_labels = val_labels.to(dtype=autodtype)

    valid_dataset = MSEDataset(valid_df[args.image_key].tolist(), val_labels, valid_trans, valid_df if set_df else None)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    valid_loader.num_samples = len(valid_dataset)
    valid_loader.num_batches = len(valid_loader)

    return train_loader, valid_loader


def make_ml_mse_dataset(args, autodtype, device, set_df=False):
    df = pd.read_parquet(args.train)
    args.mse_ranges = np.stack(df["mse_label"].values).max(axis=0).astype(int)

    # 이진 라벨과 연속형 라벨을 이진 형식으로 합칩니다.
    bin_labels = np.stack(df['label'].values)
    mse_labels = np.stack(df['mse_label'].values)

    combined_label = combine_binary_and_mse(bin_labels, mse_labels)
    df['combined_label'] = combined_label.tolist()

    # train_test_split 함수의 stratify인자를 활용하기 위해 빈도수 1개인 클래스를 제외
    df['target_str'] = df['combined_label'].astype(str)
    value_counts = df.value_counts(subset='target_str')

    single_sample_classes = value_counts[value_counts == 1].index
    multi_sample_classes = value_counts[value_counts > 1].index

    single_sample_df = df[df['target_str'].isin(single_sample_classes)]
    multi_sample_df = df[df['target_str'].isin(multi_sample_classes)]
    train_df, valid_df = train_test_split(multi_sample_df, test_size=args.valid_size, stratify=multi_sample_df['target_str'], random_state=42)
    train_df = pd.concat([train_df, single_sample_df])  # 빈도수가 1인 라벨은 train에 포함
    del df['target_str'], train_df['target_str'], valid_df['target_str']

    print(f"Total Data: {len(df)}, Train: {len(train_df)}, Valid: {len(valid_df)}\n")
    args.num_classes = len(df.loc[0, args.target_key]) + len(df.loc[0, args.mse_target_key])

    # Loss function weight
    # class weight
    class_counts = torch.tensor(train_df[args.target_key].sum(), dtype=autodtype, device=device)
    class_weights = 1.0 / torch.log1p(class_counts + 2)
    # class_weights = 1.0 / class_counts
    args.class_weight_val = (class_weights / class_weights.mean()) ** args.class_weight
    if args.label_json:
        with open(args.label_json, 'r') as f:
            class_list = json.load(f)

    # pos weight
    bce_num_classes = len(df.loc[0, args.target_key])
    label_count = torch.tensor(len(train_df) * bce_num_classes, dtype=autodtype, device=device)
    args.pos_weight_val = (label_count - class_counts.sum()) / class_counts.sum() * args.pos_weight

    print(f"## Class weight")
    if args.label_json: print(f"Class: {class_list}")
    print(f"Weight: {[weight.item() for weight in args.class_weight_val]}")
    print(f"Pos Weight: {[args.pos_weight_val.item()]}")


    transforms = get_preset(args.train_aug)
    valid_trans = get_preset(args.valid_aug)
    
    train_labels = torch.tensor(train_df[args.target_key].tolist(), dtype=autodtype, device='cpu')
    train_mse_labels = (torch.tensor(train_df[args.mse_target_key].tolist(), dtype=autodtype, device='cpu') / args.mse_ranges).to(dtype=autodtype)
    train_dataset = MLMSEDataset(train_df[args.image_key].tolist(), train_labels, train_mse_labels, transforms, train_df if set_df else None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    valid_labels = torch.tensor(valid_df[args.target_key].tolist(), dtype=autodtype, device='cpu')
    valid_mse_labels = (torch.tensor(valid_df[args.mse_target_key].tolist(), dtype=autodtype, device='cpu') / args.mse_ranges).to(dtype=autodtype)
    valid_dataset = MLMSEDataset(valid_df[args.image_key].tolist(), valid_labels, valid_mse_labels, valid_trans, valid_df if set_df else None)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    valid_loader.num_samples = len(valid_dataset)
    valid_loader.num_batches = len(valid_loader)

    return train_loader, valid_loader


def exclude_norm_totensor(transforms):
    transforms = [trans for trans in transforms if not isinstance(trans, (A.augmentations.transforms.Normalize, A.pytorch.transforms.ToTensorV2))]
    return A.Compose(transforms)

def remove_normalize_and_totensor(transforms: A.Compose) -> A.Compose:
    def _filter(t):
        # 만약 내부에 또 Compose가 있으면 재귀적으로 처리
        if isinstance(t, A.core.composition.Compose):
            return A.Compose([_filter(inner) for inner in t.transforms 
                              if not isinstance(inner, (A.augmentations.transforms.Normalize, A.pytorch.transforms.ToTensorV2))],
                             p=t.p)
        else:
            return t

    filtered = [
        _filter(t) for t in transforms.transforms
        if not isinstance(t, (A.augmentations.transforms.Normalize, A.pytorch.transforms.ToTensorV2))
    ]
    return A.Compose(filtered, p=transforms.p)


class TrainDataset(Dataset):
    def __init__(self, images, labels, transforms):
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.test_transforms = exclude_norm_totensor(transforms)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        label = self.labels[idx]
        
        return image, label
    
    def get_pil(self, idx, pixels=None):
        image =Image.open(self.images[idx])
        if pixels:
            width, height = image.size

            aspect_ratio = width / height
            resize_h  = math.ceil(math.sqrt(pixels**2 / aspect_ratio))
            resize_w = math.ceil(resize_h * aspect_ratio)

            image = image.resize((resize_w, resize_h), Image.Resampling.BICUBIC)

        return image
    
    def get_trnasformed(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.test_transforms(image=image)['image']

        return Image.fromarray(image)
    

    

class MLDataset(Dataset):
    def __init__(self, images, labels, transforms, df=None):
        self.df = df
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.test_transforms = exclude_norm_totensor(transforms)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        label = self.labels[idx]
        
        return image, label
    
    def get_pil(self, idx, pixels=None):
        image =Image.open(self.images[idx])
        if pixels:
            width, height = image.size

            aspect_ratio = width / height
            resize_h  = math.ceil(math.sqrt(pixels**2 / aspect_ratio))
            resize_w = math.ceil(resize_h * aspect_ratio)

            image = image.resize((resize_w, resize_h), Image.Resampling.BICUBIC)

        return image
    
    def get_trnasformed(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.test_transforms(image=image)['image']

        return Image.fromarray(image)


class MSEDataset(Dataset):
    def __init__(self, images, labels, transforms, df=None):
        self.df = df
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.test_transforms = exclude_norm_totensor(transforms)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        label = self.labels[idx]
        
        return image, label
    
    def get_pil(self, idx, pixels=None):
        image =Image.open(self.images[idx])
        if pixels:
            width, height = image.size

            aspect_ratio = width / height
            resize_h  = math.ceil(math.sqrt(pixels**2 / aspect_ratio))
            resize_w = math.ceil(resize_h * aspect_ratio)

            image = image.resize((resize_w, resize_h), Image.Resampling.BICUBIC)

        return image
    
    def get_trnasformed(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.test_transforms(image=image)['image']

        return Image.fromarray(image)
    
class MLMSEDataset(Dataset):
    def __init__(self, images, labels, mse_labels, transforms, df=None):
        self.df = df
        self.images = images
        self.labels = labels
        self.mse_labels = mse_labels
        self.transforms = transforms
        self.test_transforms = exclude_norm_totensor(transforms)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        label = self.labels[idx]
        mse_label = self.mse_labels[idx]
        
        return image, label, mse_label
    
    def get_pil(self, idx, pixels=None):
        image =Image.open(self.images[idx])
        if pixels:
            width, height = image.size

            aspect_ratio = width / height
            resize_h  = math.ceil(math.sqrt(pixels**2 / aspect_ratio))
            resize_w = math.ceil(resize_h * aspect_ratio)

            image = image.resize((resize_w, resize_h), Image.Resampling.BICUBIC)

        return image
    
class PretrainDataset(Dataset):
    def __init__(self, images, labels, transforms, times=None, df=None):
        self.df = df
        self.images = images
        self.labels = labels
        self.times = times
        self.trans1 = transforms['trans1']
        self.trans2 = transforms['trans2']
        self.test_trans1 = remove_normalize_and_totensor(self.trans1)
        self.test_trans2 = remove_normalize_and_totensor(self.trans2)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image1 = self.trans1(image=image)['image']
        image2 = self.trans2(image=image)['image']

        label = self.labels[idx]
        output = {'image1': image1, 'image2': image2, 'label': label}

        if self.times is not None:
            output['time'] = self.times[idx]

        return output
    
    def get_pil(self, idx, pixels=None):
        image =Image.open(self.images[idx])
        if pixels:
            width, height = image.size

            aspect_ratio = width / height
            resize_h  = math.ceil(math.sqrt(pixels**2 / aspect_ratio))
            resize_w = math.ceil(resize_h * aspect_ratio)

            image = image.resize((resize_w, resize_h), Image.Resampling.BICUBIC)

        return image
    
    def get_transform1(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.test_trans1(image=image)['image']

        return Image.fromarray(image)
    
    def get_transform2(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.test_trans2(image=image)['image']

        return Image.fromarray(image)
    
class PretrainValidDataset(PretrainDataset):
    def __init__(self, images, labels, transforms, valid_transforms, times=None, df=None):
        super().__init__(images, labels, transforms, times, df)
        self.valid_transforms = valid_transforms

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image1 = self.trans1(image=image)['image']
        image2 = self.trans2(image=image)['image']
        image3 = self.valid_transforms(image=image)['image']

        label = self.labels[idx]

        output= {'image1': image1, 'image2': image2, 'image3':image3, 'label': label}

        if self.times is not None:
            output['time'] = self.times[idx]

        return output

class CustomBatchSampler(Sampler[int]):
    r"""
    class를 랜덤으로 하나 샘플링한뒤 해당 클래스와 같은 클래스의 샘플을 num_postive 갯수만큼 
    1개 미니배치에 포함시킵니다. 나머지 는 선택된 class의 샘플을 제외하고 무작위로 채워집니다.

    Args:
        num_postive: 1개 미니배치에 포함하고 싶은 postive sample의 갯수 (min_num, max_num):Tuple
            ex: (2, 4)로 설정할 경우 샘플링된 클래스가 미내배치내에 2~4개 존재합니다.
    """

    def __init__(self, dataset, batch_size, num_postive=[2, 2], drop_last=True) -> None:
        self.dataset = dataset
        self.indices = [i for i in range(len(self.dataset))]
        self.labels = dataset.labels.to(dtype=int).tolist()
        self.batch_size = batch_size
        self.num_postive = num_postive
        self.drop_last = drop_last

        data_by_class = defaultdict(set)
        for idx, label in zip(self.indices, self.labels):
            data_by_class[label].add(idx)

        self.data_by_class = data_by_class
        self.idx_to_class = {idx: label for idx, label in zip(self.indices, self.labels)}

    def __iter__(self):
        self.all_batches = self.make_batches(self.data_by_class, self.idx_to_class, self.batch_size, self.num_postive, self.drop_last)
        for idxs in self.all_batches:
            yield idxs
        # return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return math.ceil(len(self.dataset) / self.batch_size)
    
    def make_batches(self, data_by_class, idx_to_class, batch_size, num_postive, drop_last):
        data_by_class = deepcopy(data_by_class)
        all_batches = []
        while True:
            available_classes = [k for k, v in data_by_class.items() if v]  # 남은 클래스만
            if not available_classes:
                break

            # 2. 랜덤 클래스 선택
            cls = random.choice(available_classes)

            # 3. 최소 2~4개 뽑기
            n_samples = min(random.randint(*num_postive), len(data_by_class[cls]))
            selected = random.sample(list(data_by_class[cls]), n_samples)
            data_by_class[cls].difference_update(selected)  # O(1) 제거

            # 4. 남은 부분 채우기
            remaining = batch_size - n_samples
            other_classes = [k for k in available_classes if k != cls]
            other_candidates = [i for k in other_classes for i in data_by_class[k]]

            if remaining > len(other_candidates):
                if drop_last:
                    break
                else:
                    selected += other_candidates
                    for k in other_classes:
                        data_by_class[k].clear()
                    random.shuffle(selected)
                    all_batches.append(selected)
                    break
            else:
                other_selected = random.sample(other_candidates, remaining)
                selected += other_selected
                for idx in other_selected:
                    data_by_class[idx_to_class[idx]].remove(idx)

                random.shuffle(selected)
                all_batches.append(selected)

        return all_batches
    
    
class InferDataset(Dataset):
    def __init__(self, images, transforms, df=None):
        self.df = df
        self.images = images
        self.transforms = transforms
        self.test_transforms = exclude_norm_totensor(transforms)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        
        return image
    
    def get_pil(self, idx, pixels=None):
        image =Image.open(self.images[idx])
        if pixels:
            width, height = image.size

            aspect_ratio = width / height
            resize_h  = math.ceil(math.sqrt(pixels**2 / aspect_ratio))
            resize_w = math.ceil(resize_h * aspect_ratio)

            image = image.resize((resize_w, resize_h), Image.Resampling.BICUBIC)

        return image
    
    def get_trnasformed(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.test_transforms(image=image)['image']

        return Image.fromarray(image)
    
class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top : top + h, left : left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask
    
