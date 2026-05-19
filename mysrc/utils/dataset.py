import logging
import pandas as pd
import cv2
import math
import json
import torch
import numpy as np
import random
import re
import yaml

from collections import defaultdict, Counter
from copy import deepcopy
import albumentations as A
from functools import partial
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from open_clip import get_tokenizer

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

def style_collate(batch):
    output = {}
    styled = torch.stack([data['styled'] for data in batch])
    content = torch.stack([data['content'] for data in batch])
    output['image'] = torch.concat((styled, content), dim=0)

    styled_tokens = torch.stack([data['styled_tokens'] for data in batch])
    content_tokens = torch.stack([data['content_tokens'] for data in batch])
    output['text'] = torch.concat((styled_tokens, content_tokens), dim=0)

    styled_tags_onehot = torch.stack([data['styled_tags_onehot'] for data in batch])
    content_tags_onehot = torch.stack([data['content_tags_onehot'] for data in batch])
    output['tags'] = torch.concat((styled_tags_onehot, content_tags_onehot), dim=0)

    styled_keep_tags_onehot = torch.stack([data['styled_keep_tags_onehot'] for data in batch])
    content_keep_tags_onehot = torch.stack([data['content_keep_tags_onehot'] for data in batch])
    output['keep_tags'] = torch.concat((styled_keep_tags_onehot, content_keep_tags_onehot), dim=0)

    styled_label = torch.stack([data['label'] for data in batch])
    content_label = torch.tensor([0.0] * len(batch), dtype=styled_label.dtype)
    output['label'] = torch.cat((styled_label, content_label), dim=0)
    output['artist'] = [data['artist'] for data in batch]

    output['styled_label_token'] = torch.concat([data['styled_label_token'] for data in batch], dim=0)
    output['content_label_token'] = torch.concat([data['content_label_token'] for data in batch], dim=0)

    if 'time' in batch[0]:
        styled_times = torch.stack([data['time'] for data in batch])
        content_times = torch.tensor([0.0] * len(batch), dtype=styled_times.dtype)
        output['time'] = torch.cat((styled_times, content_times), dim=0)
    
    return output

def style_collate02(batch):
    output = {}
    styled1 = torch.stack([data['styled1'] for data in batch])
    content1 = torch.stack([data['content1'] for data in batch])
    output['image1'] = torch.concatenate((styled1, content1), dim=0)

    styled2 = torch.stack([data['styled2'] for data in batch])
    content2 = torch.stack([data['content2'] for data in batch])
    output['image2'] = torch.concatenate((styled2, content2), dim=0)

    output['label'] = torch.stack([data['label'] for data in batch])

    if 'time' in batch[0]:
        output['time'] = torch.stack([data['time'] for data in batch])

    if 'styled_valid' in batch[0]:
        styled_valid = torch.stack([data['styled_valid'] for data in batch])
        content_valid = torch.stack([data['content_valid'] for data in batch])
        output['image_valid'] = torch.concatenate((styled_valid, content_valid), dim=0)
    
    return output

def pretrain_collate(batch, mask_ratio_tuple, mask_probability, n_tokens=None, mask_generator=None):
    output = {}
    for key in batch[0].keys():
        output[key] = torch.stack([data[key] for data in batch])

    B = len(output['image1'])
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    
    masks_list1 = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list1.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list1.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list1)

    masks_list2 = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list2.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list2.append(torch.BoolTensor(mask_generator(0)))
        
    random.shuffle(masks_list2)

    masks_list = masks_list1 + masks_list2

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    output['collated_masks'] = collated_masks
    output['mask_indices_list'] = mask_indices_list
    output['masks_weight'] = masks_weight
    output['upperbound'] = upperbound
    output['n_masked_patches'] = torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),

    return output

def pretrain_collate02(batch, patch_size, size_per_mask_range, mask_probability, mask_generator=None):
    output = {}
    image1 = torch.stack([data['image1'] for data in batch])
    image2 = torch.stack([data['image2'] for data in batch])
    output['image'] = torch.concat((image1, image2), dim=0)

    output['label'] = torch.stack([data['label'] for data in batch])
    if 'time' in batch[0]:
        output['time'] = torch.stack([data['time'] for data in batch])

    if 'image3' in batch[0]:
        output['image_eval'] = torch.stack([data['image3'] for data in batch])

    B, C, H, W = image1.shape
    n_samples_masked = int(B * mask_probability)

    global_crop_masks = []
    for i in range(0, n_samples_masked):
        global_crop_masks.append(torch.BoolTensor(mask_generator(H, patch_size, size_per_mask_range)))
    for i in range(n_samples_masked, B):
        global_crop_masks.append(torch.BoolTensor(mask_generator(H, patch_size, 0)))

    random.shuffle(global_crop_masks)

    B, C, H, W = image2.shape
    n_samples_masked = int(B * mask_probability)

    local_crop_masks = []
    for i in range(0, n_samples_masked):
        local_crop_masks.append(mask_generator(H, patch_size, size_per_mask_range))
    for i in range(n_samples_masked, B):
        local_crop_masks.append(torch.BoolTensor(mask_generator(H, patch_size, 0)))

    random.shuffle(local_crop_masks)

    # masks_list = global_crop_masks + local_crop_masks

    global_crop_masks = torch.stack(global_crop_masks).flatten(1)
    global_mask_indices = global_crop_masks.flatten().nonzero().flatten()
    global_none_mask_indices = (global_crop_masks == False).flatten().nonzero().flatten()
    global_masks_weight = global_crop_masks.sum(-1)
    global_mask_samples = global_masks_weight.nonzero().flatten()
    global_masks_weight = (1 / global_masks_weight.clamp(min=1.0)).unsqueeze(-1).expand_as(global_crop_masks)[global_crop_masks]
    
    local_crop_masks = torch.stack(local_crop_masks).flatten(1)
    local_mask_indices = local_crop_masks.flatten().nonzero().flatten()
    local_none_mask_indices = (local_crop_masks == False).flatten().nonzero().flatten()
    local_masks_weight = local_crop_masks.sum(-1)
    local_mask_samples = local_masks_weight.nonzero().flatten()
    local_masks_weight = (1 / local_masks_weight.clamp(min=1.0)).unsqueeze(-1).expand_as(local_crop_masks)[local_crop_masks]

    output['crop_masks'] = torch.concat((global_crop_masks, local_crop_masks), dim=0)
    output['mask_indices'] = torch.concat((global_mask_indices, local_mask_indices), dim=0)
    output['none_mask_indices'] = torch.concat((global_none_mask_indices, local_none_mask_indices), dim=0)
    output['mask_samples'] = torch.concat((global_mask_samples, local_mask_samples + len(image1)), dim=0)
    output['masks_weight'] = torch.concat((global_masks_weight, local_masks_weight), dim=0)

    return output

def make_mae_dataset(args):
    train_df = pd.read_parquet(args.train)
    valid_df = pd.read_parquet(args.valid)

    transform = get_preset(args.train_aug, image_size=[224, 224], mean=[0.70843786, 0.65038896, 0.6473464 ], std=[0.25763473, 0.26254627, 0.25369588])
    valid_trans = get_preset(args.valid_aug, image_size=[224, 224], mean=[0.70843786, 0.65038896, 0.6473464 ], std=[0.25763473, 0.26254627, 0.25369588])

    logging.info(f"Train Image Processor:")
    logging.info(f"{transform}")
    logging.info(f"Valid Image Processor: ")
    logging.info(f"{valid_trans}")


    train_dataset = NoLabelDataset(train_df.image.tolist(), transforms=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    valid_dataset = NoLabelDataset(valid_df.image.tolist(), transforms=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    valid_loader.num_samples = len(valid_dataset)
    valid_loader.num_batches = len(valid_loader)

    return train_loader, valid_loader

def make_style_dataset(args):
    train_df = pd.read_parquet(args.train)
    valid_df = pd.read_parquet(args.valid)

    transforms = get_preset(args.train_aug, image_size=args.image_size, mean=[0.70843786, 0.65038896, 0.6473464], std=[0.25763473, 0.26254627, 0.25369588])
    # artist_tokens = [f"<artist_{target}>" for target in train_df.target.unique()]
    artist_tokens = ["<artist_0>"] + [f"<artist_{target}>" for target in train_df.target.unique()]
    tokenizer = get_tokenizer('hf-hub:timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k', additional_special_tokens=artist_tokens)
    args.tokenizer = tokenizer

    with open(args.tag2id, 'r', encoding='utf-8') as f:
        tag2id = json.load(f)

    with open(args.token2tag, 'r', encoding='utf-8') as f:
        token2tag = json.load(f)
 
    # train_df = train_df[0:100]
    # valid_df = valid_df[0:100]
    # train_df = train_df[train_df['target'].isin([i for i in range(80)])]
    # valid_df = valid_df[valid_df['target'].isin([i for i in range(80)])]

    train_dataset = StyleClipDataset(
        styleds=train_df.image.tolist(),
        contents=train_df.im_content.tolist(),
        labels=train_df.target.tolist(),
        artists=train_df.artist.tolist(),
        styled_tags=train_df.tags.tolist(),
        styled_grey_tags=train_df.grey_tags.tolist(),
        content_tags=train_df.content_tags.tolist(),
        transforms=transforms,
        tokenizer=tokenizer,
        tag2id=tag2id,
        token2tag=token2tag,
        grey_p=0.2,
        times=train_df.years_passed.tolist(),
    )
    sampler = CustomBatchSampler(train_dataset, batch_size=args.batch_size, num_postive=args.num_postive, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=style_collate, pin_memory=True, num_workers=args.workers)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    valid_dataset = StyleClipDataset(
        styleds=valid_df.image.tolist(),
        contents=valid_df.im_content.tolist(),
        labels=valid_df.target.tolist(),
        artists=valid_df.artist.tolist(),
        styled_tags=valid_df.tags.tolist(),
        styled_grey_tags=valid_df.grey_tags.tolist(),
        content_tags=valid_df.content_tags.tolist(),
        transforms=transforms,
        tokenizer=tokenizer,
        tag2id=tag2id,
        token2tag=token2tag,
        grey_p=0,
        times=valid_df.years_passed.tolist(),
    )
    valid_loader = DataLoader(valid_dataset, collate_fn=style_collate, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    valid_loader.num_samples = len(valid_dataset)
    valid_loader.num_batches = len(valid_loader)

    return train_loader, valid_loader


def make_style_dataset02(args):
    train_df = pd.read_parquet(args.train)
    valid_df = pd.read_parquet(args.valid)

    train_transforms = get_preset(args.train_aug, image_size=args.image_size, mean=[0.70843786, 0.65038896, 0.6473464], std=[0.25763473, 0.26254627, 0.25369588])

    train_dataset = StyleDataset(
        styleds=train_df.image.tolist(),
        contents=train_df.im_content.tolist(),
        labels=train_df.target.tolist(),
        transforms=train_transforms,
        times=train_df.years_passed.tolist(),
    )
    sampler = CustomBatchSampler(train_dataset, batch_size=args.batch_size, num_postive=args.num_postive, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=style_collate02, pin_memory=True, num_workers=args.workers)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    valid_dataset = StyleDataset(
        styleds=valid_df.image.tolist(),
        contents=valid_df.im_content.tolist(),
        labels=valid_df.target.tolist(),
        transforms=train_transforms,
        times=valid_df.years_passed.tolist(),
        is_valid=True,
    )
    valid_loader = DataLoader(valid_dataset, collate_fn=style_collate02, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    valid_loader.num_samples = len(valid_dataset)
    valid_loader.num_batches = len(valid_loader)

    return train_loader, valid_loader


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

def make_realation_label(path, df, dtype):
    with open(path, 'r', encoding='utf-8') as file:
        sim_labels = yaml.safe_load(file)

    sim_split_labels = []
    for line in sim_labels:
        temp = list(re.search("(.+)<==>(.+):[ ]*([\d].[\d]+)", line).groups())
        temp[2] = float(temp[2])
        sim_split_labels.append(temp)

    num_classes = df['style_group'].nunique()
    rel_matrix = torch.eye(num_classes, dtype=dtype)

    class_to_target = {}
    for class_name, target in df[['style_group', 'target']].values:
        class_to_target[class_name] = target

    for tag1, tag2, weight in sim_split_labels:
        if tag1 in class_to_target and tag2 in class_to_target:
            i, j = class_to_target[tag1], class_to_target[tag2]
            rel_matrix[i, j] = rel_matrix[j, i] = weight

    return rel_matrix



def make_pretrain_dataset02(args):
    train_df = pd.read_parquet(args.train)
    valid_df = pd.read_parquet(args.valid)
    print(f"Total Data: {len(train_df) + len(valid_df)}, Train: {len(train_df)}, Valid: {len(valid_df)}")

    # train_df = train_df[train_df['target'].isin([i for i in range(20)])]
    # valid_df = valid_df[valid_df['target'].isin([i for i in range(20)])]
    args.num_classes = train_df['target'].nunique()
    args.rel_real = make_realation_label(args.relation, train_df, torch.float32)
    args.rel_pseudo = args.rel_real.clone()

    # transforms = get_preset(args.train_aug, image_size=args.image_size, mean=[0.70843786, 0.65038896, 0.6473464], std=[0.25763473, 0.26254627, 0.25369588])
    # valid_trans = get_preset(args.valid_aug, image_size=args.image_size, mean=[0.70843786, 0.65038896, 0.6473464], std=[0.25763473, 0.26254627, 0.25369588])
    transforms = get_preset(args.train_aug, image_size=args.image_size)
    valid_trans = get_preset(args.valid_aug, image_size=args.image_size)

    logging.info(f"Train Image Processor:")
    logging.info(f"{transforms}")
    logging.info(f"Valid Image Processor: ")
    logging.info(f"{valid_trans}")

    mask_generator = CustomMaskingGenerator()

    collate_fn = partial(
        pretrain_collate02,
        patch_size=args.patch_size,
        size_per_mask_range=args.size_per_mask_range,
        mask_probability=0.7,
        mask_generator=mask_generator,
    )

    train_labels = torch.tensor(train_df.target.tolist(), dtype=torch.float32, device='cpu')
    train_times = torch.tensor(train_df.years_passed.tolist(), dtype=torch.float32, device='cpu')
    train_dataset = PretrainDataset(train_df.image.tolist(), train_labels, transforms, times=train_times)
    sampler = CustomBatchSampler(train_dataset, batch_size=args.batch_size, num_postive=args.num_postive, drop_last=True)
    # sampler = ReleationBatchSampler(train_dataset, batch_size=args.batch_size, rel_matrix=args.rel_pseudo.detach().cpu().numpy(), drop_last=True)
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

def make_pretrain_dataset03(args):
    train_df = pd.read_parquet(args.train)
    valid_df = pd.read_parquet(args.valid)
    print(f"Total Data: {len(train_df) + len(valid_df)}, Train: {len(train_df)}, Valid: {len(valid_df)}")

    transforms = get_preset(args.train_aug, image_size=args.image_size, mean=[0.70843786, 0.65038896, 0.6473464 ], std=[0.25763473, 0.26254627, 0.25369588])
    valid_trans = get_preset(args.valid_aug, image_size=args.image_size, mean=[0.70843786, 0.65038896, 0.6473464 ], std=[0.25763473, 0.26254627, 0.25369588])

    logging.info(f"Train Image Processor:")
    logging.info(f"{transforms}")
    logging.info(f"Valid Image Processor: ")
    logging.info(f"{valid_trans}")

    mask_generator = CustomMaskingGenerator(
        input_size=args.image_size[0],
        patch_size=args.patch_size,
    )

    collate_fn = partial(
        pretrain_collate02,
        patch_size=args.patch_size,
        size_per_mask_range=args.size_per_mask_range,
        mask_probability=0.5,
        mask_generator=mask_generator,
    )

    train_labels = torch.tensor(train_df.target.tolist(), dtype=torch.float32, device='cpu')
    train_times = torch.tensor(train_df.years_passed.tolist(), dtype=torch.float32, device='cpu')
    train_dataset = PretrainLocalCrops(train_df.image.tolist(), train_labels, transforms, num_crops=args.num_crops, times=train_times)
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
    transforms = [trans for trans in transforms if not isinstance(trans, (A.augmentations.pixel.transforms.Normalize, A.pytorch.transforms.ToTensorV2))]
    return A.Compose(transforms)

def remove_normalize_and_totensor(transforms: A.Compose) -> A.Compose:
    def _filter(t):
        # 만약 내부에 또 Compose가 있으면 재귀적으로 처리
        if isinstance(t, A.core.composition.Compose):
            return A.Compose([_filter(inner) for inner in t.transforms 
                              if not isinstance(inner, (A.augmentations.pixel.transforms.Normalize, A.pytorch.transforms.ToTensorV2))],
                             p=t.p)
        else:
            return t

    filtered = [
        _filter(t) for t in transforms.transforms
        if not isinstance(t, (A.augmentations.pixel.transforms.Normalize, A.pytorch.transforms.ToTensorV2))
    ]
    return A.Compose(filtered, p=transforms.p)


class TrainDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.images = images
        if labels is None:
            self.labels = None
        self.transforms = transforms
        self.test_transforms = exclude_norm_totensor(transforms)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        if self.labels is None:
            label = None
        
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
    
class NoLabelDataset(TrainDataset):
    def __init__(self, images, transforms=None):
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
    
class PretrainInferDataset(PretrainDataset):
    def __init__(self, images, labels, transforms, times=None, df=None):
        self.df = df
        self.images = images
        self.labels = labels
        self.times = times
        self.transforms = transforms
        self.test_transforms = remove_normalize_and_totensor(self.transforms)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        label = self.labels[idx]

        output= {'image': image, 'label': label}

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
    
    def get_transform(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.test_trans1(image=image)['image']

        return Image.fromarray(image)
    
class PretrainLocalCrops(Dataset):
    def __init__(self, images, labels, transforms, num_crops, times=None, df=None):
        self.df = df
        self.images = images
        self.labels = labels
        self.num_crops = num_crops
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

        crops = []
        for i in range(self.num_crops):
            crops.append(self.trans2(image=image)['image'])

        image2 = torch.stack(crops, dim=0)

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
    
class StyleClipDataset(Dataset):
    def __init__(self, styleds, contents, labels, artists, styled_tags, styled_grey_tags, content_tags, transforms, tokenizer, tag2id=None, token2tag=None, grey_p=0.2, times=None, df=None):
        self.df = df
        self.styleds = styleds
        self.contents = contents
        self.labels = torch.tensor(labels, dtype=torch.float32, device='cpu') if not isinstance(labels, torch.Tensor) else labels
        self.artists = artists

        self.styled_tags = styled_tags
        self.styled_grey_tags = styled_grey_tags
        self.content_tags = content_tags

        self.shared_trans = transforms['shared_trans']
        self.style_trans = transforms['style_trans']
        self.end_trans = transforms['end_trans']

        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.token2tag = token2tag
        self.times = torch.tensor(times, dtype=torch.float32, device='cpu') if not isinstance(times, torch.Tensor) else times
        self.grey_p = grey_p

    def __len__(self):
        return len(self.styleds)
    
    def __getitem__(self, idx):
        if random.random() >= self.grey_p:
            styled = cv2.imread(self.styleds[idx], cv2.IMREAD_COLOR)
            styled = cv2.cvtColor(styled, cv2.COLOR_BGR2RGB)
            styled_tag = self.styled_tags[idx]
        else:
            styled = cv2.imread(self.styleds[idx], cv2.IMREAD_GRAYSCALE)
            styled = cv2.cvtColor(styled, cv2.COLOR_GRAY2BGR)
            styled_tag = self.styled_grey_tags[idx]

        content = cv2.imread(self.contents[idx], cv2.IMREAD_COLOR)
        content_tags = self.content_tags[idx]
        
        # Augmentation
        shared = self.shared_trans(image=styled, image2=content)    # 둘이 완전히 동일한 처리를 적용함. (Random한 Aumgentation이 포함되어 있어도 두 이미지는 완전히 동일한 처리가 적용됨.)
        styled, content = shared['image'], shared['image2']
        styled = self.style_trans(image=styled)['image']
        styled = self.end_trans(image=styled)['image']
        content = self.end_trans(image=content)['image']

        label = self.labels[idx]
        artist = self.artists[idx]

        styled_tokens, styled_tags_onehot, styled_keep_tags_onehot = self.tokenize_tag(styled_tag, int(label.item()))
        content_tokens, content_tags_onehot, content_keep_tags_onehot = self.tokenize_tag(content_tags)
        styled_label_token = self.get_label_token(int(label.item()))
        content_label_token = self.get_label_token()

        output = {
            'styled': styled, 'styled_tag': styled_tag, 
            'styled_tokens':styled_tokens, 'styled_tags_onehot': styled_tags_onehot, 'styled_keep_tags_onehot': styled_keep_tags_onehot,
            'content': content, 'content_tags': content_tags, 
            'content_tokens': content_tokens, 'content_tags_onehot': content_tags_onehot, 'content_keep_tags_onehot': content_keep_tags_onehot,
            'styled_label_token': styled_label_token, 'content_label_token': content_label_token,
            'label': label, 'artist': artist
        }

        if self.times is not None:
            output['time'] = self.times[idx]

        return output
    
    def split_tokens(self, data, sep=267):
        result = []
        current = []
        for num in data:
            if num == sep:
                if current:   # 빈 리스트는 저장하지 않음
                    result.append(current)
                    current = []
            else:
                current.append(num)
        if current:  # 마지막 덩어리 처리
            result.append(current)
        return result
    
    def get_label_token(self, label=0):
        label_tag = f"<artist_{label}>"
        result = self.tokenizer(label_tag)
        return result
        
    def tokenize_tag(self, tags, label=0):
        np.random.shuffle(tags)
        label_tag = f"<artist_{label}>"
        label_token = self.tokenizer.encode(label_tag)[0]

        tag_str = " , ".join(tags) + " ,"
        result = torch.zeros(self.tokenizer.context_length, dtype=torch.long)

        tokens = self.tokenizer.encode(tag_str)
        # tags_encoded = split_tokens(tokens)[1:] if label is not None else self.split_tokens(tokens)  # 라벨태그 제거
        # tags_encoded = self.split_tokens(tokens)
        # tags_encoded = [str(tag) for tag in tags_encoded]
        num_keep = self.tokenizer.context_length - 2 - 1 # sos, eos, style_token

        keep_tokens = tokens[:num_keep+1]
        trunc_id = (len(keep_tokens)-1) - keep_tokens[::-1].index(267)

        keep_tags_encoded = self.split_tokens(keep_tokens[:trunc_id])
        # _keep_tags_encoded = keep_tags_encoded[1:] if label is not None else keep_tags_encoded  # 라벨태그 제거
        keep_tags_encoded = [str(tag) for tag in keep_tags_encoded]
        keep_tags = [self.token2tag[token] for token in keep_tags_encoded]

        result_tokens = [self.tokenizer.sot_token_id] + keep_tokens[:trunc_id] + [label_token, self.tokenizer.eot_token_id]
        result[:len(result_tokens)] = torch.tensor(result_tokens)
        # [tokenizer.decode(tag) for tag in tags_encoded], [tokenizer.decode(tag) for tag in _keep_tags_encoded]

        # Tag One Hot
        tag_vocab_size = len(self.tag2id)
        tag_onehot = torch.zeros(tag_vocab_size)
        tag_keep_onehot = torch.zeros(tag_vocab_size)

        idxs = [self.tag2id[tag] for tag in tags]
        tag_onehot[idxs] = 1.0

        idxs = [self.tag2id[tag] for tag in keep_tags]
        tag_keep_onehot[idxs] = 1.0

        # return result, tags_encoded, _keep_tags_encoded
        return result, tag_onehot, tag_keep_onehot
    

    def get_pil(self, idx, pixels=None):
        image =Image.open(self.images[idx])
        if pixels:
            width, height = image.size

            aspect_ratio = width / height
            resize_h  = math.ceil(math.sqrt(pixels**2 / aspect_ratio))
            resize_w = math.ceil(resize_h * aspect_ratio)

            image = image.resize((resize_w, resize_h), Image.Resampling.BICUBIC)

        return image
    
    def get_transform(self, idx) -> tuple:
        styled = cv2.imread(self.styleds[idx], cv2.IMREAD_COLOR)
        styled = cv2.cvtColor(styled, cv2.COLOR_BGR2RGB)

        content = cv2.imread(self.contents[idx], cv2.IMREAD_COLOR)
        
        shared = self.shared_trans(image=styled, image2=content)    # 둘이 완전히 동일한 처리를 적용함. (Random한 Aumgentation이 포함되어 있어도 두 이미지는 완전히 동일한 처리가 적용됨.)
        styled, content = shared['image'], shared['image2']
        styled = self.style_trans(image=styled)['image']

        return Image.fromarray(styled), Image.fromarray(content)
    

class StyleDataset(Dataset):
    def __init__(self, styleds, contents, labels, transforms, is_valid=False, times=None, df=None):
        self.df = df
        self.is_valid = is_valid
        self.styleds = styleds
        self.contents = contents
        self.labels = torch.tensor(labels, dtype=torch.float32, device='cpu') if not isinstance(labels, torch.Tensor) else labels
        self.times = torch.tensor(times, dtype=torch.float32, device='cpu') if not isinstance(times, torch.Tensor) else times
        self.shared_trans = transforms['shared_trans']
        self.style_trans = transforms['style_trans']
        self.end_trans = transforms['end_trans']
        self.valid_trans = transforms['valid_trans']

    def __len__(self):
        return len(self.styleds)
    
    def __getitem__(self, idx):
        styled = cv2.imread(self.styleds[idx], cv2.IMREAD_COLOR)
        styled = cv2.cvtColor(styled, cv2.COLOR_BGR2RGB)
        content = cv2.imread(self.contents[idx], cv2.IMREAD_COLOR)
        
        # Augmentation
        shared1 = self.shared_trans(image=styled, image2=content)    # 둘이 완전히 동일한 처리를 적용함. (Random한 Aumgentation이 포함되어 있어도 두 이미지는 완전히 동일한 처리가 적용됨.)
        styled1, content1 = shared1['image'], shared1['image2']
        styled1 = self.style_trans(image=styled1)['image']
        styled1 = self.end_trans(image=styled1)['image']
        content1 = self.end_trans(image=content1)['image']

        shared2 = self.shared_trans(image=styled, image2=content)
        styled2, content2 = shared2['image'], shared2['image2']
        styled2 = self.style_trans(image=styled2)['image']
        styled2 = self.end_trans(image=styled2)['image']
        content2 = self.end_trans(image=content2)['image']

        label = self.labels[idx]

        output = {
            'styled1': styled1,
            'content1': content1, 
            'styled2': styled2,
            'content2': content2, 
            'label': label, 
        }

        if self.times is not None:
            output['time'] = self.times[idx]

        if self.is_valid:
            shared_valid = self.valid_trans(image=styled, image2=content)
            styled_valid, content_valid = shared_valid['image'], shared_valid['image2']
            output['styled_valid'] = styled_valid
            output['content_valid'] = content_valid

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
    
    def get_transform(self, idx) -> tuple:
        styled = cv2.imread(self.styleds[idx], cv2.IMREAD_COLOR)
        styled = cv2.cvtColor(styled, cv2.COLOR_BGR2RGB)

        content = cv2.imread(self.contents[idx], cv2.IMREAD_COLOR)
        
        shared = self.shared_trans(image=styled, image2=content)    # 둘이 완전히 동일한 처리를 적용함. (Random한 Aumgentation이 포함되어 있어도 두 이미지는 완전히 동일한 처리가 적용됨.)
        styled, content = shared['image'], shared['image2']
        styled = self.style_trans(image=styled)['image']

        return Image.fromarray(styled), Image.fromarray(content)
    

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
            available_classes = [k for k, v in data_by_class.items() if len(v) >= num_postive[0]]  # 남은 클래스만
            if not available_classes:
                break

            # 2. 랜덤 클래스 선택
            cls = random.choice(available_classes)

            # 3. 최소 n개 뽑기
            pos_n = random.randint(*num_postive) # 뽑을 pos 갯수
            n_samples = min(pos_n, len(data_by_class[cls]))
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

class ReleationBatchSampler(Sampler[int]):
    r"""
    class를 랜덤으로 하나 샘플링한뒤 해당 클래스와 같은 클래스의 샘플을 num_postive 갯수만큼 
    1개 미니배치에 포함시킵니다. 나머지 는 선택된 class의 샘플을 제외하고 무작위로 채워집니다.

    Args:
        num_postive: 1개 미니배치에 포함하고 싶은 postive sample의 갯수 (min_num, max_num):Tuple
            ex: (2, 4)로 설정할 경우 샘플링된 클래스가 미내배치내에 2~4개 존재합니다.
    """

    def __init__(self, dataset, batch_size, rel_matrix: np.ndarray, drop_last=True) -> None:
        self.dataset = dataset
        indices = [i for i in range(len(self.dataset))]
        labels = dataset.labels.to(dtype=int).tolist()
        self.unique_labels = dataset.labels.unique().to(dtype=int).tolist()
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.rel_matrix = rel_matrix

        data_by_class = defaultdict(list)
        for idx, label in zip(indices, labels):
            data_by_class[label].append(idx)

        self.data_by_class = data_by_class

    def __iter__(self):
        self.all_batches = self.make_batches()
        for idxs in self.all_batches:
            yield idxs
        # return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return math.ceil(len(self.dataset) / self.batch_size)

    def relation_weight(self, tau=2.0):
        means = self.rel_matrix.mean(axis=-1)

        # 평균 낮을수록 높은 score
        scores = -means

        # normalize (z-score)
        scores = (scores - scores.mean()) / (scores.std() + 1e-8)

        # softmax
        exp_scores = np.exp(scores / tau)
        weights = exp_scores / exp_scores.sum()

        return weights
        
    def make_batches(self):
        data_by_class_copy = deepcopy(self.data_by_class)

        remain_num = sum(len(v) for v in data_by_class_copy.values())
        available_mask = np.ones(len(self.unique_labels), dtype=bool)
        all_batches = []
        # self.all_classes = []
        # self.remain_class = []

        relation_weights = self.relation_weight(2.5)

        while True:
            class_count_weight = np.array([len(data_by_class_copy[i]) for i in self.unique_labels])
            cls = random.choices(self.unique_labels, weights=class_count_weight * relation_weights, k=1)[0]

            selected_samples = []
            # classes = []

            # 선택된 클래스에서 1개 샘플 보장
            samples = data_by_class_copy[cls]

            idx = random.randrange(len(samples))
            selected_samples.append(samples.pop(idx))
            # classes.append(cls)
            remain_num -= 1

            if not samples:
                available_mask[cls] = False

            # batch 샘플링
            while len(selected_samples) < self.batch_size:
                weights=(self.rel_matrix[cls] + 0.01) * available_mask # 연관성 0인 데이터도 낮은 가중치로 보장
                selected_cls = random.choices(self.unique_labels, weights=weights, k=self.batch_size-1)

                counter = Counter([cls] + selected_cls)
                most_freq = counter.most_common(1)[0][1]

                # 과도한 같은 클래스 중복 및 다양한 클래스 샘플링을 위한 조건문
                if 0.5 < most_freq / self.batch_size or len(counter) <= 4:
                    weights = self.rel_matrix[cls].copy() + 0.01
                    weights[cls] = 0

                    selected_cls = random.choices(self.unique_labels, weights=weights, k=self.batch_size // 2)
                    selected_cls = selected_cls + [cls] * (self.batch_size // 2 - 1) # 전체 batch 수가 너무 늘어나는 것 방지

                    for i in selected_cls:
                        if i == cls:
                            samples = data_by_class_copy[i]
                        else:
                            samples = self.data_by_class[i]  # 전체 데이터에서 샘플링

                        if samples:
                            idx = random.randrange(len(samples))

                            # 샘플링한 데이터가 cls와 같으면 제거
                            if i == cls:
                                selected_samples.append(samples.pop(idx))
                                remain_num -= 1
                                
                                if not samples:
                                    available_mask[i] = False
                            else:
                                selected_samples.append(samples[idx])

                            # classes.append(i)

                            if len(selected_samples) >= self.batch_size:
                                break

                else:
                    for i in selected_cls:
                        samples = data_by_class_copy[i]

                        if samples:
                            # random.choice + remove 대신 pop(idx)
                            idx = random.randrange(len(samples))
                            selected_samples.append(samples.pop(idx))
                            remain_num -= 1
                            # classes.append(i)

                            if not samples:
                                available_mask[i] = False

                            if len(selected_samples) >= self.batch_size:
                                break
            
            random.shuffle(selected_samples)
            all_batches.append(selected_samples)
            # self.all_classes.append(classes)
            # self.remain_class.append(np.array(self.unique_labels)[available_mask])

            if len(all_batches) >= len(self):
                self.data_by_class_copy = data_by_class_copy
                break

            if remain_num < self.batch_size:
                if self.drop_last:
                    break
                else:
                    all_batches.append([v for value in data_by_class_copy.values() for v in value])
                    break
                
        random.shuffle(all_batches)
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

class CustomMaskingGenerator:
    '''
    input_size / patch_size x input_size / patch_size 크기의 유니폼한 마스크 텐서 반환

    Args:
        size_per_mask_range: 개별 마스크의 크기 범위
            ex:(1, 3) => 한번에 마스킹 되는 크기가 1x1, 2x2, 3x3 중 랜덤으로 생성

    '''
    def __init__(
        self,
    ):
        self.patterns = [
                            [[1, 0], [0, 1]],
                            [[1, 0], [0, 0]]
                        ]

    def __call__(self, input_size, patch_size, size_per_mask_range):
        n_patches = int(input_size / patch_size)
        
        if size_per_mask_range == 0:
            return torch.zeros(n_patches, n_patches, dtype=torch.bool)
        
        mask_scale = random.randint(*size_per_mask_range)  # 한번에 마스킹할 최대 patch 크기
        pattern = torch.tensor(random.choices(self.patterns, weights=[1.0, 1.0], k=1)[0])
        mask_unit = torch.kron(pattern, torch.ones((mask_scale, mask_scale), dtype=torch.int))    # mask_scale 기준으로 하나의 pattern의 크기를 키움 최대 3x3패치까지 한번에 마스킹

        unit_size = mask_unit.size(0)
        repeat = math.ceil(n_patches / unit_size) + 1   # mask_unit을 n_patches x n_patches보다 크면서 다양한 패턴으로 crop하기 위해 repeat 수를 +1 해줌.

        repeated_mask = mask_unit.repeat(repeat, repeat)    #  mask_unit을 repeat x repeat으로 확장

        # 최대 1 mask_unit 만큼만 움직이면서 crop => 가능한 패턴에 대해서 유니폼한 확률로 crop할 것을 의도
        left = random.randint(1, unit_size)
        top = random.randint(1, unit_size)

        mask = repeated_mask[top:top+n_patches, left:left+n_patches].to(dtype=bool)    # crop
        return mask
    
