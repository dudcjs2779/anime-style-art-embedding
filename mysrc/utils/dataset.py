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
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from open_clip import get_tokenizer

from mysrc.utils.aug_preset import get_preset


def pretrain_collate03(batch, mask_ratio_tuple, mask_probability, n_tokens=None, mask_generator=None):
    output = {}
    output['index'] = torch.stack([data['index'] for data in batch])

    image1 = torch.stack([data['image1'] for data in batch])
    image2 = torch.stack([data['image2'] for data in batch])
    output['image'] = torch.concat((image1, image2), dim=0)

    output['label'] = torch.stack([data['label'] for data in batch])

    if 'feat_label' in batch[0]:
        output['feat_label'] = torch.stack([data['feat_label'] for data in batch])

    if 'hard' in batch[0]:
        output['hard'] = torch.stack([data['hard'] for data in batch])

    if 'center' in batch[0]:
        output['center'] = torch.stack([data['center'] for data in batch])

    if 'not_far' in batch[0]:
        output['not_far'] = torch.stack([data['not_far'] for data in batch])

    if 'image_eval' in batch[0]:
        output['image_eval'] = torch.stack([data['image_eval'] for data in batch])

    B = len(image1)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0

    global_crop_masks = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        global_crop_masks.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
    for i in range(n_samples_masked, B):
        global_crop_masks.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(global_crop_masks)

    local_crop_masks = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        local_crop_masks.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
    for i in range(n_samples_masked, B):
        local_crop_masks.append(torch.BoolTensor(mask_generator(0)))

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

def pretrain_collate04(batch):
    output = {}
    output['index'] = torch.stack([data['index'] for data in batch])

    output['image'] = torch.stack([data['image'] for data in batch])

    output['label'] = torch.stack([data['label'] for data in batch])

    output['tag'] = torch.stack([data['tags'] for data in batch])
    # output['greytag'] = torch.stack([data['greytags'] for data in batch])
    output['raw_tags'] = [data['raw_tags'] for data in batch]
    # output['raw_greytags'] = [data['raw_greytags'] for data in batch]

    if 'hard' in batch[0]:
        output['hard'] = torch.stack([data['hard'] for data in batch])

    if 'center' in batch[0]:
        output['center'] = torch.stack([data['center'] for data in batch])

    if 'not_far' in batch[0]:
        output['not_far'] = torch.stack([data['not_far'] for data in batch])

    # if 'image_eval' in batch[0]:
    #     output['image_eval'] = torch.stack([data['image_eval'] for data in batch])

    return output


def make_relation_label(path, df, dtype):
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

    def remap(x, src_min, src_max, dst_min, dst_max):
        return dst_min + (x - src_min) * (dst_max - dst_min) / (src_max - src_min)

    # rel_matrix = remap(rel_matrix, 0, 1.0, 0.3, 1.0)
    # rel_matrix[rel_matrix == 0.3] = 0
    # rel_matrix = torch.clamp(rel_matrix, min=0.0, max=0.9)

    self_mask = torch.eye(num_classes, dtype=torch.bool)
    strong_sim_mask = (rel_matrix < 1.0) & (rel_matrix >= 0.7)
    week_sim_mask = (rel_matrix < 0.7) & (rel_matrix >= 0.4)
    except_mask = ~(self_mask | strong_sim_mask | week_sim_mask)

    rel_matrix[self_mask] = 1.0
    rel_matrix[strong_sim_mask] = 0.9
    rel_matrix[week_sim_mask] = 0.0
    rel_matrix[except_mask] = 0.0

    return rel_matrix

def make_pretrain_dataset03(args):
    train_df = pd.read_parquet(args.train)
    valid_df = pd.read_parquet(args.valid)
    
    # train_df = train_df[train_df['target'].isin([i for i in range(100)])]
    # valid_df = valid_df[valid_df['target'].isin([i for i in range(100)])]

    print(f"Total Data: {len(train_df) + len(valid_df)}, Train: {len(train_df)}, Valid: {len(valid_df)}")
    args.num_classes = train_df['target'].nunique()
    args.rel_real = make_relation_label(args.relation, train_df, torch.float32)
    args.rel_pseudo = args.rel_real.clone()


    # transforms = get_preset(args.train_aug, image_size=args.image_size, mean=[0.70843786, 0.65038896, 0.6473464], std=[0.25763473, 0.26254627, 0.25369588])
    # valid_trans = get_preset(args.valid_aug, image_size=args.image_size, mean=[0.70843786, 0.65038896, 0.6473464], std=[0.25763473, 0.26254627, 0.25369588])
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
        pretrain_collate03,
        mask_ratio_tuple=(0.2, 0.5),
        mask_probability=0.7,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
    )
    
    train_labels = torch.tensor(train_df.target.tolist(), dtype=torch.float32, device='cpu')
    train_bboxes = [np.stack(bbox) if len(bbox) > 0 else [] for bbox in train_df.bbox.tolist()]
    train_bb_masks = train_df.bb_bi_mask.tolist()
    train_bg_masks = train_df.bg_mask.tolist()
    train_hards = torch.tensor(train_df.hard_sample.tolist(), dtype=torch.bool, device='cpu')
    train_centers = torch.tensor(train_df.center_sample.tolist(), dtype=torch.bool, device='cpu')
    train_not_fars = torch.tensor(train_df.not_far_sample.tolist(), dtype=torch.bool, device='cpu')

    train_dataset = PretrainBboxDataset(
        train_df.image.tolist(), train_labels,
        train_bboxes, train_bb_masks, train_bg_masks,
        transforms, hards=train_hards, centers=train_centers, not_fars=train_not_fars)
    
    sampler = HardLossSampler(
        train_dataset, batch_size=args.batch_size, 
        rel_real=args.rel_real.detach().cpu().numpy(), rel_pseudo=args.rel_pseudo.detach().cpu().numpy(), 
        hard_num=args.hard_num, same_num=args.same_num, accum=args.accum_freq,
    )
    
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=collate_fn, pin_memory=True, num_workers=args.workers)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fn, pin_memory=True, num_workers=args.workers)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    valid_labels = torch.tensor(valid_df.target.tolist(), dtype=torch.float32, device='cpu')
    valid_dataset = PretrainValidBboxDataset(valid_df.image.tolist(), valid_labels, transforms, valid_trans)
    valid_loader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    valid_loader.num_samples = len(valid_dataset)
    valid_loader.num_batches = len(valid_loader)

    return train_loader, valid_loader


def make_pretrain_dataset04(args):
    dataset = {}
    train_df = pd.read_parquet(args.train)
    train_df = train_df[train_df['target'].isin([i for i in range(20)])]

    args.num_classes = train_df['target'].nunique()
    args.rel_real = make_relation_label(args.relation, train_df, torch.float32)
    args.rel_pseudo = args.rel_real.clone()

    with open(args.feat_label, 'r', encoding='utf-8') as file:
        sty_feat_group = yaml.safe_load(file)
    args.feat_len = len(sty_feat_group)

    args.feat_class_weight = torch.tensor(np.load(args.feat_class_weight_path))
    args.feat_pos_weight = torch.tensor(np.load(args.feat_pos_weight_path))
    # args.tag_pos_weights = None

    # transforms = get_preset(args.train_aug, image_size=args.image_size, mean=[0.70843786, 0.65038896, 0.6473464], std=[0.25763473, 0.26254627, 0.25369588])
    # valid_trans = get_preset(args.valid_aug, image_size=args.image_size, mean=[0.70843786, 0.65038896, 0.6473464], std=[0.25763473, 0.26254627, 0.25369588])
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
        pretrain_collate03,
        mask_ratio_tuple=(0.2, 0.5),
        mask_probability=0.7,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
    )
    
    train_labels = torch.tensor(train_df.target.tolist(), dtype=torch.float32, device='cpu')
    train_sty_feats = [torch.tensor(feats, dtype=torch.long, device='cpu') for feats in train_df.style_feats_encoded.values]
    train_bboxes = [np.stack(bbox) if len(bbox) > 0 else [] for bbox in train_df.bbox.tolist()]
    train_bb_masks = train_df.bb_bi_mask.tolist()
    train_bg_masks = train_df.bg_mask.tolist()
    train_hards = torch.tensor(train_df.hard_sample.tolist(), dtype=torch.bool, device='cpu')
    train_centers = torch.tensor(train_df.center_sample.tolist(), dtype=torch.bool, device='cpu')
    train_not_fars = torch.tensor(train_df.not_far_sample.tolist(), dtype=torch.bool, device='cpu')


    train_dataset = PretrainBboxStyFeatDataset(
        train_df.image.tolist(), train_labels,
        train_sty_feats, args.feat_len,
        train_bboxes, train_bb_masks, train_bg_masks,
        transforms, hards=train_hards, centers=train_centers, not_fars=train_not_fars)
    
    # sampler = HardLossSampler(
    #     train_dataset, batch_size=args.batch_size, 
    #     rel_real=args.rel_real.detach().cpu().numpy(), rel_pseudo=args.rel_pseudo.detach().cpu().numpy(), 
    #     hard_num=args.hard_num, same_num=args.same_num, accum=args.accum_freq,
    # )

    sampler = GHardLossSampler(
        train_dataset, batch_size=args.batch_size, 
        rel_real=args.rel_real.detach().cpu().numpy(), rel_pseudo=args.rel_pseudo.detach().cpu().numpy(), 
        hard_num=args.hard_num, same_num=args.same_num, accum=args.accum_freq,
    )
    
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=collate_fn, pin_memory=True, num_workers=args.workers)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fn, pin_memory=True, num_workers=args.workers)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    dataset['train'] = train_loader

    if args.valid:
        valid_df = pd.read_parquet(args.valid)
        # valid_df = valid_df[valid_df['target'].isin([i for i in range(20)])]

        valid_labels = torch.tensor(valid_df.target.tolist(), dtype=torch.float32, device='cpu')
        valid_sty_feats = [torch.tensor(feats, dtype=torch.long, device='cpu') for feats in valid_df.style_feats_encoded.values]
        valid_dataset = PretrainValidBboxStyFeatDataset(valid_df.image.tolist(), valid_labels, transforms, valid_trans, valid_sty_feats, args.feat_len)
        sampler = DistributedSampler(valid_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed else None
        valid_loader = DataLoader(valid_dataset, collate_fn=collate_fn, sampler=sampler, batch_size=args.valid_batch_size, pin_memory=True, num_workers=args.workers)
        valid_loader.num_samples = len(valid_dataset)
        valid_loader.num_batches = len(valid_loader)

        dataset['valid'] = valid_loader

    if args.test:
        test_df = pd.read_parquet(args.test)
        test_df = test_df[test_df['target'].isin([i for i in range(10)])]

        args.text_num_classes = test_df['target'].nunique()

        test_labels = torch.tensor(test_df.target.tolist(), dtype=torch.float32, device='cpu')
        test_dataset = PretrainValidBboxStyFeatDataset(test_df.image.tolist(), test_labels, transforms, valid_trans)
        sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False) if args.distributed else None
        test_loader = DataLoader(test_dataset, collate_fn=collate_fn, sampler=sampler, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
        test_loader.num_samples = len(test_dataset)
        test_loader.num_batches = len(test_loader)

        dataset['test'] = test_loader


    dataset_info = ""
    total = 0
    for name, loader in dataset.items():
        total += len(loader.dataset)
        dataset_info += f"{name}: {len(loader.dataset)}, "
        
    print(f"Total data: {total}, {dataset_info}")
    
    return dataset

def make_pretrain_tag_dataset(args):
    train_df = pd.read_parquet(args.train)
    valid_df = pd.read_parquet(args.valid)

    args.tag_class_weights = torch.tensor(np.load(args.tag_class_weights_path))
    args.tag_pos_weights = torch.tensor(np.load(args.tag_pos_weights_path))
    # args.tag_pos_weights = None
    args.tag_num_classes = args.tag_class_weights.size(0)

    # train_df = train_df[:2000]
    # valid_df = valid_df[:500]
    # train_df = train_df[train_df['target'].isin([i for i in range(100)])]
    # valid_df = valid_df[valid_df['target'].isin([i for i in range(100)])]

    print(f"Total Data: {len(train_df) + len(valid_df)}, Train: {len(train_df)}, Valid: {len(valid_df)}")
    args.num_classes = train_df['target'].nunique()
    args.rel_real = make_relation_label(args.relation, train_df, torch.float32)
    args.rel_pseudo = args.rel_real.clone()

    # transforms = get_preset(args.train_aug, image_size=args.image_size, mean=[0.70843786, 0.65038896, 0.6473464], std=[0.25763473, 0.26254627, 0.25369588])
    # valid_trans = get_preset(args.valid_aug, image_size=args.image_size, mean=[0.70843786, 0.65038896, 0.6473464], std=[0.25763473, 0.26254627, 0.25369588])
    transforms = get_preset(args.train_aug, image_size=args.image_size)
    valid_trans = get_preset(args.valid_aug, image_size=args.image_size)

    logging.info(f"Train Image Processor:")
    logging.info(f"{transforms}")
    logging.info(f"Valid Image Processor: ")
    logging.info(f"{valid_trans}")
    
    train_labels = torch.tensor(train_df.target.tolist(), dtype=torch.float32, device='cpu')
    train_tags = [torch.tensor(tags, dtype=torch.long, device='cpu') for tags in train_df.encoded_tags.values]
    train_greytags = [torch.tensor(tags, dtype=torch.long, device='cpu') for tags in train_df.encoded_grey_tags.values] 
    train_tags_list = train_df.final_tags.tolist()
    train_greytags_list = train_df.grey_tags.tolist()
    train_hards = torch.tensor(train_df.hard_sample.tolist(), dtype=torch.bool, device='cpu')
    train_centers = torch.tensor(train_df.center_sample.tolist(), dtype=torch.bool, device='cpu')
    train_not_fars = torch.tensor(train_df.not_far_sample.tolist(), dtype=torch.bool, device='cpu')

    train_dataset = PretrainTagDataset(
        train_df.image.tolist(), train_labels,
        train_tags, train_greytags, args.tag_num_classes,
        transforms, raw_tags=train_tags_list, raw_greytags=train_greytags_list,
        hards=train_hards, centers=train_centers, not_fars=train_not_fars)
    
    sampler = HardLossSampler(
        train_dataset, batch_size=args.batch_size, 
        rel_real=args.rel_real.detach().cpu().numpy(), rel_pseudo=args.rel_pseudo.detach().cpu().numpy(), 
        hard_num=args.hard_num, same_num=args.same_num, accum=args.accum_freq,
    )
    
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=pretrain_collate04, pin_memory=True, num_workers=args.workers)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fn, pin_memory=True, num_workers=args.workers)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    valid_labels = torch.tensor(valid_df.target.tolist(), dtype=torch.float32, device='cpu')
    valid_tags = [torch.tensor(tags, dtype=torch.long, device='cpu') for tags in valid_df.encoded_tags.values]
    # valid_greytags = torch.tensor(np.stack(valid_df.encoded_grey_tags.values), dtype=torch.float32, device='cpu')
    valid_tags_list = valid_df.final_tags.tolist()
    # valid_greytags_list = valid_df.grey_tags.tolist()
    valid_dataset = PretrainValidTagDataset(
        valid_df.image.tolist(), valid_labels, valid_tags, args.tag_num_classes, valid_trans,
        raw_tags=valid_tags_list, 
    )
    valid_loader = DataLoader(valid_dataset, collate_fn=pretrain_collate04, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    valid_loader.num_samples = len(valid_dataset)
    valid_loader.num_batches = len(valid_loader)

    return train_loader, valid_loader


def make_pretrain_tag_dataset_pretrain(args):
    train_df = pd.read_parquet(args.train)
    valid_df = pd.read_parquet(args.valid)

    args.tag_class_weights = torch.tensor(np.load(args.tag_class_weights_path))
    args.tag_pos_weights = torch.tensor(np.load(args.tag_pos_weights_path))
    # args.tag_pos_weights = None
    args.tag_num_classes = args.tag_class_weights.size(0)

    # train_df = train_df[:2000]
    # valid_df = valid_df[:500]

    print(f"Total Data: {len(train_df) + len(valid_df)}, Train: {len(train_df)}, Valid: {len(valid_df)}")
    args.num_classes = train_df['target'].nunique()
    # args.rel_real = make_relation_label(args.relation, train_df, torch.float32)
    # args.rel_pseudo = args.rel_real.clone()

    # transforms = get_preset(args.train_aug, image_size=args.image_size, mean=[0.70843786, 0.65038896, 0.6473464], std=[0.25763473, 0.26254627, 0.25369588])
    # valid_trans = get_preset(args.valid_aug, image_size=args.image_size, mean=[0.70843786, 0.65038896, 0.6473464], std=[0.25763473, 0.26254627, 0.25369588])
    transforms = get_preset(args.train_aug, image_size=args.image_size)
    valid_trans = get_preset(args.valid_aug, image_size=args.image_size)

    logging.info(f"Train Image Processor:")
    logging.info(f"{transforms}")
    logging.info(f"Valid Image Processor: ")
    logging.info(f"{valid_trans}")
    
    train_labels = torch.tensor(train_df.target.tolist(), dtype=torch.float32, device='cpu')
    train_tags = [torch.tensor(tags, dtype=torch.long, device='cpu') for tags in train_df.encoded_tags.values]
    train_greytags = [torch.tensor(tags, dtype=torch.long, device='cpu') for tags in train_df.encoded_grey_tags.values] 
    train_tags_list = train_df.final_tags.tolist()
    train_greytags_list = train_df.grey_tags.tolist()
    train_hards = torch.tensor(train_df.hard_sample.tolist(), dtype=torch.bool, device='cpu')
    train_centers = torch.tensor(train_df.center_sample.tolist(), dtype=torch.bool, device='cpu')
    train_not_fars = torch.tensor(train_df.not_far_sample.tolist(), dtype=torch.bool, device='cpu')

    train_dataset = PretrainTagDataset(
        train_df.image.tolist(), train_labels,
        train_tags, train_greytags, args.tag_num_classes,
        transforms, raw_tags=train_tags_list, raw_greytags=train_greytags_list,
        hards=train_hards, centers=train_centers, not_fars=train_not_fars)
    
    sampler = HardLossSampler(
        train_dataset, batch_size=args.batch_size, 
        # rel_real=args.rel_real.detach().cpu().numpy(), rel_pseudo=args.rel_pseudo.detach().cpu().numpy(), 
        rel_real=None, rel_pseudo=None, 
        hard_num=args.hard_num, same_num=args.same_num, accum=args.accum_freq,
    )
    
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, collate_fn=pretrain_collate04, pin_memory=True, num_workers=args.workers)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, collate_fn=collate_fn, pin_memory=True, num_workers=args.workers)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    valid_labels = torch.tensor(valid_df.target.tolist(), dtype=torch.float32, device='cpu')
    valid_tags = [torch.tensor(tags, dtype=torch.long, device='cpu') for tags in valid_df.encoded_tags.values]
    # valid_greytags = torch.tensor(np.stack(valid_df.encoded_grey_tags.values), dtype=torch.float32, device='cpu')
    valid_tags_list = valid_df.final_tags.tolist()
    # valid_greytags_list = valid_df.grey_tags.tolist()
    valid_dataset = PretrainValidTagDataset(
        valid_df.image.tolist(), valid_labels, valid_tags, args.tag_num_classes, valid_trans,
        raw_tags=valid_tags_list, 
    )
    valid_loader = DataLoader(valid_dataset, collate_fn=pretrain_collate04, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
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

class PretrainTagDataset(Dataset):
    def __init__(self, images, labels, tags, greytags, tag_num, transforms, raw_tags=None, raw_greytags=None, hards=None, centers=None, not_fars=None, df=None):
        self.df = df
        self.indices = torch.arange(0, len(images), dtype=torch.long, device='cpu')
        self.images = images
        self.labels = labels
        self.tags = tags
        self.greytags = greytags
        self.tag_num = tag_num
        self.raw_tags = raw_tags
        self.raw_greytags = raw_greytags
        self.hards = hards
        self.centers = centers
        self.not_fars = not_fars
        self.transforms = transforms
        self.test_trans = remove_normalize_and_totensor(self.transforms)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]
        
        hard = self.hards[idx]
        center = self.centers[idx]
        not_far = self.not_fars[idx]

        transformed = self.transforms(image=image)

        applied = transformed["applied_transforms"]
        is_grey = any(name == "ToGray" for name, params in applied)

        tag_indices = self.greytags[idx] if is_grey else self.tags[idx]
        tags = torch.zeros(self.tag_num, dtype=torch.float32)
        tags[tag_indices] = 1.0

        if is_grey:
            if self.raw_greytags is not None:
                raw_tags = self.raw_greytags[idx]
        else:
            if self.raw_tags is not None:
                raw_tags = self.raw_tags[idx]

        output = {
            'index': self.indices[idx], 'image': transformed['image'], 'label': label, 
            'tags': tags, 'raw_tags': raw_tags,
            'hard': hard, 'center': center, 'not_far': not_far}

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
        
        transformed = self.transforms(image=image)

        return Image.fromarray(transformed['image'])

class PretrainValidTagDataset(Dataset):
    def __init__(self, images, labels, tags, tag_num, transforms, raw_tags=None, df=None):
        self.df = df
        self.indices = torch.arange(0, len(images), dtype=torch.long, device='cpu')
        self.images = images
        self.labels = labels
        self.tags = tags
        self.tag_num = tag_num
        self.raw_tags = raw_tags
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transforms(image=image)['image']
        label = self.labels[idx]

        tag_indices = self.tags[idx]
        tags = torch.zeros(self.tag_num, dtype=torch.float32)
        tags[tag_indices] = 1.0

        if self.raw_tags is not None:
            raw_tags = self.raw_tags[idx]

        output= {
            'index': self.indices[idx], 'image': image, 'label': label, 
            'tags': tags, 'raw_tags': raw_tags}
        return output
    


class PretrainBboxDataset(Dataset):
    def __init__(self, images, labels, bboxes, bb_masks, bg_masks, transforms, hards=None, centers=None, not_fars=None, df=None):
        self.df = df
        self.indices = torch.arange(0, len(images), dtype=torch.long, device='cpu')
        self.images = images
        self.labels = labels
        self.bboxes = bboxes
        self.bb_masks = bb_masks
        self.bg_masks = bg_masks
        self.hards = hards
        self.centers = centers
        self.not_fars = not_fars
        self.transforms = transforms
        self.test_trans1 = remove_normalize_and_totensor(self.transforms)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bb_mask = self.bb_masks[idx]
        if bb_mask is not None:
            bb_mask = cv2.imread(self.bb_masks[idx], cv2.IMREAD_GRAYSCALE)

        bg_mask = self.bg_masks[idx]
        if bg_mask is not None:
            bg_mask = cv2.imread(self.bg_masks[idx], cv2.IMREAD_GRAYSCALE)

        label = self.labels[idx]
        bbox = self.bboxes[idx]
        hard = self.hards[idx]
        center = self.centers[idx]
        not_far = self.not_fars[idx]

        transformed1 = self.transforms(
            image=image,
            mask=bb_mask,
            bg_mask=bg_mask,
            bboxes=bbox,
        )

        transformed2 = self.transforms(
            image=image,
            mask=bb_mask,
            bg_mask=bg_mask,
            bboxes=bbox,
        )
        output = {'index': self.indices[idx], 'image1': transformed1['image'], 'image2': transformed2['image'], 'label': label, 'hard': hard, 'center': center, 'not_far': not_far}

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
        bb_mask = cv2.imread(self.bboxes[idx], cv2.IMREAD_GRAYSCALE)
        bg_mask = cv2.imread(self.bg_masks[idx], cv2.IMREAD_GRAYSCALE)
        bbox = self.bboxes[idx]
        
        transformed = self.transforms(
            image=image,
            mask=bb_mask,
            bg_mask=bg_mask,
            bboxes=bbox,
        )

        return Image.fromarray(transformed['image'])

class PretrainValidBboxDataset(Dataset):
    def __init__(self, images, labels, transforms, valid_transforms, df=None):
        self.df = df
        self.indices = torch.arange(0, len(images), dtype=torch.long, device='cpu')
        self.images = images
        self.labels = labels
        self.transforms = transforms
        self.valid_transforms = valid_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image1 = self.transforms(image=image)['image']
        image2 = self.transforms(image=image)['image']
        image_eval = self.valid_transforms(image=image)['image']
        label = self.labels[idx]

        output= {'index': self.indices[idx], 'image1': image1, 'image2': image2, 'image_eval': image_eval, 'label': label}
        return output

class PretrainBboxStyFeatDataset(Dataset):
    def __init__(self, images, labels, feat_labels, feat_labels_len, bboxes, bb_masks, bg_masks, transforms, hards=None, centers=None, not_fars=None, df=None):
        self.df = df
        self.indices = torch.arange(0, len(images), dtype=torch.long, device='cpu')
        self.images = images
        self.labels = labels
        self.feat_labels = feat_labels
        self.feat_labels_len = feat_labels_len
        self.bboxes = bboxes
        self.bb_masks = bb_masks
        self.bg_masks = bg_masks
        self.hards = hards
        self.centers = centers
        self.not_fars = not_fars
        self.transforms = transforms
        self.test_trans1 = remove_normalize_and_totensor(self.transforms)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bb_mask = self.bb_masks[idx]
        if bb_mask is not None:
            bb_mask = cv2.imread(self.bb_masks[idx], cv2.IMREAD_GRAYSCALE)

        bg_mask = self.bg_masks[idx]
        if bg_mask is not None:
            bg_mask = cv2.imread(self.bg_masks[idx], cv2.IMREAD_GRAYSCALE)

        label = self.labels[idx]
        bbox = self.bboxes[idx]
        hard = self.hards[idx]
        center = self.centers[idx]
        not_far = self.not_fars[idx]

        feat_label_indices = self.feat_labels[idx]
        feat_label = torch.zeros(self.feat_labels_len, dtype=torch.float32)
        feat_label[feat_label_indices] = 1.0

        transformed1 = self.transforms(
            image=image,
            mask=bb_mask,
            bg_mask=bg_mask,
            bboxes=bbox,
        )

        transformed2 = self.transforms(
            image=image,
            mask=bb_mask,
            bg_mask=bg_mask,
            bboxes=bbox,
        )
        output = {'index': self.indices[idx], 'image1': transformed1['image'], 'image2': transformed2['image'], 'label': label, 'feat_label': feat_label, 'hard': hard, 'center': center, 'not_far': not_far}

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
        bb_mask = cv2.imread(self.bboxes[idx], cv2.IMREAD_GRAYSCALE)
        bg_mask = cv2.imread(self.bg_masks[idx], cv2.IMREAD_GRAYSCALE)
        bbox = self.bboxes[idx]
        
        transformed = self.transforms(
            image=image,
            mask=bb_mask,
            bg_mask=bg_mask,
            bboxes=bbox,
        )

        return Image.fromarray(transformed['image'])

class PretrainValidBboxStyFeatDataset(Dataset):
    def __init__(self, images, labels, transforms, valid_transforms, feat_labels=None, feat_labels_len=None, df=None):
        self.df = df
        self.indices = torch.arange(0, len(images), dtype=torch.long, device='cpu')
        self.images = images
        self.labels = labels
        self.feat_labels = feat_labels
        self.feat_labels_len = feat_labels_len
        self.transforms = transforms
        self.valid_transforms = valid_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image1 = self.transforms(image=image)['image']
        image2 = self.transforms(image=image)['image']
        image_eval = self.valid_transforms(image=image)['image']
        label = self.labels[idx]

        output= {
            'index': self.indices[idx], 
            'image1': image1, 'image2': image2, 'image_eval': image_eval, 
            'label': label
        }

        if self.feat_labels is not None:
            feat_label_indices = self.feat_labels[idx]
            feat_label = torch.zeros(self.feat_labels_len, dtype=torch.float32)
            feat_label[feat_label_indices] = 1.0
            output['feat_label'] = feat_label

        return output

class PretrainInferBboxStyFeatDataset(Dataset):
    def __init__(self, images, transforms, labels=None, feat_labels=None, feat_labels_len=None, df=None):
        self.df = df
        self.indices = torch.arange(0, len(images), dtype=torch.long, device='cpu')
        self.images = images
        self.labels = labels
        self.feat_labels = feat_labels
        self.feat_labels_len = feat_labels_len
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        output = {'index': self.indices[idx]}

        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        output['image'] = self.transforms(image=image)['image']

        if self.labels is not None:
            output['label'] = self.labels[idx]

        if self.feat_labels is not None:
            feat_label_indices = self.feat_labels[idx]
            feat_label = torch.zeros(self.feat_labels_len, dtype=torch.float32)
            feat_label[feat_label_indices] = 1.0
            output['feat_label'] = feat_label

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

class HardLossSampler(Sampler[list[int]]):
    """
    score가 높은 샘플을 우선적으로 기준 클래스로 설정하며 batch에 반드시 포함됩니다.
    score 샘플링 이후 모델이 잘 맞추지 못하는 클래스 일수록 기준 클래스로 선택됩니다.
    기준 클래스에서 샘플하나, 그리고 hard_num 만큼 잘 못 맞추고 있는 클래스,
    나머지는 유니폼 샘플링을 합니다.

    모델이 잘 맞추지 못하는 클래스 및 샘플에 집중하도록 의도했습니다.
    """

    def __init__(self, dataset, batch_size, rel_real: np.ndarray = None, rel_pseudo: np.ndarray = None, hard_num=[1, 2], same_num=[1, 2], accum=1) -> None:
        self.dataset = dataset
        self.indices = [i for i in range(len(self.dataset))]
        labels = dataset.labels.to(dtype=int).tolist()
        self.unique_labels = dataset.labels.unique().to(dtype=int).tolist()
        n_classes = len(self.unique_labels)
        self.batch_size = batch_size
        self.accum = accum
        self.rel_real = rel_real if rel_real is not None else np.eye(n_classes)
        self.rel_pseudo = rel_pseudo if rel_pseudo is not None else np.eye(n_classes)
        self.gt_mask = self.rel_real != 0
        self.hard_num = hard_num
        self.same_num = same_num
        self.cls_diff_weights = np.ones(n_classes)
        self.sample_diff_weights = np.ones(n_classes)
        self.sample_gauss_weights = np.ones(n_classes)
        self.sample_score = np.zeros(len(self.dataset))
        self.score_sampling_limit = 0

        data_by_class = defaultdict(list)
        for idx, label in zip(self.indices, labels):
            data_by_class[label].append(idx)

        self.data_by_class = data_by_class

        data_to_class = dict()
        for idx, label in zip(self.indices, labels):
            data_to_class[idx] = label

        self.data_to_class = data_to_class

    def __iter__(self):
        self.data_by_class_copy = deepcopy(self.data_by_class)
        # self.remain_num = sum(len(v) for v in self.data_by_class.values())
        self.available_mask = np.ones(len(self.unique_labels), dtype=bool)
        self.sample_available_mask = np.ones(len(self.indices), dtype=bool)

        if self.sample_score.sum() > 0:
            self.score_sampling_limit = len(self) * 0.1 // self.accum

        # self.all_batches = []
        # self.all_classes = []
        # self.remain_class = []

        for i in range(len(self)):
            idxs = self.make_one_batch(i)
            yield idxs
            
        # for idxs in self.all_batches:
        #     yield idxs

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size
        
    def cls_diff_weight(self, rel_real, rel_pseudo, eps, temp):
        # axis=1 기준 평균 절대 오차
        diff_weight = np.abs(rel_real - rel_pseudo).sum(axis=1, where=self.gt_mask) / self.gt_mask.sum(axis=1)

        weights = np.power(diff_weight + eps, temp)
        weights /= weights.sum()

        return weights

    def sample_diff_weight(self, rel_real, rel_pseudo, mask, eps, temp):
        diff_weight = np.abs(rel_real - rel_pseudo)

        weights = np.where(
            mask,
            np.power(diff_weight + eps, temp),
            1e-6
        )

        weights /= weights.sum(axis=1, keepdims=True)
        return weights
        
    def asymmetric_gaussian_weight(self, similarity, center=0.55, left_sigma=0.15, right_sigma=0.1):
        """
        center를 중심으로 한 비대칭 정규분포
        args:
            similarity: 분포
            center: 중심, 최빈값
            left_sigma: 왼쪽 분포의 sigma
            right_sigma: 오른쪽 분포의 sigma
        """
        weights = np.zeros_like(similarity)

        left_mask = similarity < center
        right_mask = ~left_mask

        weights[left_mask] = np.exp(-((similarity[left_mask] - center) ** 2) / (2 * left_sigma ** 2))
        weights[right_mask] = np.exp(-((similarity[right_mask] - center) ** 2) / (2 * right_sigma ** 2))

        return weights

    def update_weights(self):
        self.cls_diff_weights = self.cls_diff_weight(self.rel_real, self.rel_pseudo, 0.1, 1.0)
        self.sample_diff_weights = self.sample_diff_weight(self.rel_real, self.rel_pseudo, self.gt_mask, 0.1, 1.2)
        self.sample_gauss_weights = self.asymmetric_gaussian_weight(self.rel_pseudo, center=0.75, left_sigma=0.20, right_sigma=0.15) + 0.1

    def update_score(self, scores, top_p=10):
        scores = scores.detach().cpu().numpy()

        k = max(1, int(len(scores) * top_p / 100))

        indices = np.argpartition(scores, -k)[-k:]

        result = np.zeros_like(scores)
        result[indices] = scores[indices]

        self.sample_score = result
    
    def make_one_batch(self, step):
        selected_samples = []
        # classes = []

        if step % self.accum == 0:
            same_k = random.randint(*self.same_num)

            if self.score_sampling_limit > 0:
                weights = self.sample_score * self.sample_available_mask

                if not np.any(weights):
                    self.score_sampling_limit = 0
                    weights = self.sample_available_mask

                score_sample = random.choices(self.indices, weights=weights, k=1)[0]
                cls = self.data_to_class[score_sample]
                self.score_sampling_limit -= 1

                self.cls = cls

                samples = self.data_by_class_copy[cls]
                samples.remove(score_sample)
                selected_samples.append(score_sample)
                self.sample_available_mask[score_sample] = False
                # classes.append(cls)

                if not samples:
                    self.available_mask[cls] = False
                    same_k = len(selected_samples)

                for _ in range(same_k-1):
                    samples = self.data_by_class_copy[cls]

                    idx = random.randrange(len(samples))
                    selected_sample = samples.pop(idx)
                    selected_samples.append(selected_sample)
                    self.sample_available_mask[selected_sample] = False
                    # classes.append(cls)

                    if not samples:
                        self.available_mask[cls] = False
                        same_k = len(selected_samples)
                        break

            else:
                cls = random.choices(self.unique_labels, weights=self.cls_diff_weights * self.available_mask, k=1)[0]
                self.cls = cls  

                for _ in range(same_k):
                    samples = self.data_by_class_copy[cls]

                    idx = random.randrange(len(samples))
                    # selected_samples.append(samples.pop(idx))
                    selected_sample = samples.pop(idx)
                    selected_samples.append(selected_sample)
                    self.sample_available_mask[selected_sample] = False
                    # classes.append(cls)

                    if not samples:
                        self.available_mask[cls] = False
                        same_k = len(selected_samples)
                        break
        else:
            same_k = 0
            cls = self.cls


        while len(selected_samples) < self.batch_size:
            hard_k = random.randint(*self.hard_num)
            gaus_k = self.batch_size - same_k - hard_k

            selected_cls = (
                # random.choices(self.unique_labels, weights=self.sample_gauss_weights[cls] * self.available_mask, k=gaus_k)
                random.choices(self.unique_labels, k=gaus_k)
                + random.choices(self.unique_labels, weights=self.sample_diff_weights[cls] * self.available_mask, k=hard_k)
            )

            for sel_cls in selected_cls:
                samples = self.data_by_class_copy[sel_cls]

                if samples:
                    idx = random.randrange(len(samples))
                    # selected_samples.append(samples.pop(idx))
                    selected_sample = samples.pop(idx)
                    selected_samples.append(selected_sample)
                    self.sample_available_mask[selected_sample] = False
                    # classes.append(sel_cls)

                    if not samples:
                        self.available_mask[sel_cls] = False

                    if len(selected_samples) >= self.batch_size:
                        break

        random.shuffle(selected_samples)
        # self.all_batches.append(selected_samples)
        # self.all_classes.append(classes)
        # self.classes = classes
        # self.remain_class.append(np.array(self.unique_labels)[self.available_mask])

        return selected_samples

class GHardLossSampler(Sampler[list[int]]):
    """
    score가 높은 샘플을 우선적으로 기준 클래스로 설정하며 batch에 반드시 포함됩니다.
    score 샘플링 이후 모델이 잘 맞추지 못하는 클래스 일수록 기준 클래스로 선택됩니다.
    기준 클래스에서 샘플하나, 그리고 hard_num 만큼 잘 못 맞추고 있는 클래스,
    나머지는 유니폼 샘플링을 합니다.

    모델이 잘 맞추지 못하는 클래스 및 샘플에 집중하도록 의도했습니다.
    """

    def __init__(
        self,
        dataset,
        batch_size,
        rel_real: np.ndarray = None,
        rel_pseudo: np.ndarray = None,
        hard_num=(1, 2),
        same_num=(1, 2),
        accum=1,
        num_replicas=None,
        rank=None,
    ):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        labels = dataset.labels.to(dtype=int).tolist()
        self.unique_labels = dataset.labels.unique().to(dtype=int).tolist()
        n_classes = len(self.unique_labels)

        # DDP 정보
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.num_replicas = num_replicas
        self.rank = rank

        # batch_size는 GPU당 batch 크기
        self.local_batch_size = batch_size
        self.batch_size = batch_size * num_replicas
        self.accum = accum

        self.rel_real = rel_real if rel_real is not None else np.eye(n_classes)
        self.rel_pseudo = rel_pseudo if rel_pseudo is not None else np.eye(n_classes)
        self.gt_mask = self.rel_real != 0

        self.hard_num = hard_num
        self.same_num = same_num

        self.cls_diff_weights = np.ones(n_classes)
        self.sample_diff_weights = np.ones((n_classes))
        self.sample_gauss_weights = np.ones((n_classes))

        self.sample_score = np.zeros(len(self.dataset))
        self.score_sampling_limit = 0

        data_by_class = defaultdict(list)
        for idx, label in zip(self.indices, labels):
            data_by_class[label].append(idx)

        self.data_by_class = dict(data_by_class)
        self.data_to_class = {idx: label for idx, label in zip(self.indices, labels)}

        # 클래스 라벨이 배열의 인덱스로 사용 가능한지 확인
        if sorted(self.unique_labels) != list(range(n_classes)):
            raise ValueError(
                "현재 가중치 배열은 클래스 라벨이 "
                "0부터 n_classes-1까지 연속된 정수라고 가정합니다. "
                "비연속 라벨은 클래스 인덱스로 매핑해야 합니다."
            )

        if self.local_batch_size <= 0 or self.num_replicas <= 0:
            raise ValueError("batch_size와 num_replicas는 양수여야 합니다.")

        if self.accum <= 0:
            raise ValueError("accum은 양수여야 합니다.")

        if self.batch_size < max(self.same_num) + max(self.hard_num):
            raise ValueError(
                "Global Batch 크기가 same_num과 hard_num의 "
                "최대 합보다 작습니다."
            )

    def __iter__(self):
        distributed = dist.is_available() and dist.is_initialized() and self.num_replicas > 1

        if self.rank == 0:
            self.data_by_class_copy = deepcopy(self.data_by_class)
            self.available_mask = np.ones(len(self.unique_labels), dtype=bool)
            self.sample_available_mask = np.ones(len(self.indices), dtype=bool)

            if self.sample_score.sum() > 0:
                self.score_sampling_limit = (len(self) * 0.1 // self.accum)
            else:
                self.score_sampling_limit = 0

        for step in range(len(self)):
            if self.rank == 0:
                # Global Batch 생성
                global_batch = self.make_one_batch(step)
            else:
                global_batch = None

            if distributed:
                # rank 0의 Global Batch를 모든 rank에 전달
                payload = [global_batch]
                dist.broadcast_object_list(payload, src=0)
                global_batch = payload[0]

            # 각 rank는 자신의 구간만 사용
            start = self.rank * self.local_batch_size
            end = start + self.local_batch_size

            local_batch = global_batch[start:end]

            if len(local_batch) != self.local_batch_size:
                raise RuntimeError(
                    f"Rank {self.rank}: local batch 크기가 "
                    f"{len(local_batch)}입니다."
                )
            
            yield local_batch

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size
        
    def cls_diff_weight(self, rel_real, rel_pseudo, eps, temp):
        # axis=1 기준 평균 절대 오차
        diff_weight = np.abs(rel_real - rel_pseudo).sum(axis=1, where=self.gt_mask) / self.gt_mask.sum(axis=1)

        weights = np.power(diff_weight + eps, temp)
        weights /= weights.sum()

        return weights

    def sample_diff_weight(self, rel_real, rel_pseudo, mask, eps, temp):
        diff_weight = np.abs(rel_real - rel_pseudo)

        weights = np.where(
            mask,
            np.power(diff_weight + eps, temp),
            1e-6
        )

        weights /= weights.sum(axis=1, keepdims=True)
        return weights
        
    def asymmetric_gaussian_weight(self, similarity, center=0.55, left_sigma=0.15, right_sigma=0.1):
        """
        center를 중심으로 한 비대칭 정규분포
        args:
            similarity: 분포
            center: 중심, 최빈값
            left_sigma: 왼쪽 분포의 sigma
            right_sigma: 오른쪽 분포의 sigma
        """
        weights = np.zeros_like(similarity)

        left_mask = similarity < center
        right_mask = ~left_mask

        weights[left_mask] = np.exp(-((similarity[left_mask] - center) ** 2) / (2 * left_sigma ** 2))
        weights[right_mask] = np.exp(-((similarity[right_mask] - center) ** 2) / (2 * right_sigma ** 2))

        return weights

    def update_weights(self):
        self.cls_diff_weights = self.cls_diff_weight(self.rel_real, self.rel_pseudo, 0.1, 1.0)
        self.sample_diff_weights = self.sample_diff_weight(self.rel_real, self.rel_pseudo, self.gt_mask, 0.1, 1.2)
        self.sample_gauss_weights = self.asymmetric_gaussian_weight(self.rel_pseudo, center=0.75, left_sigma=0.20, right_sigma=0.15) + 0.1

    def update_score(self, scores, top_p=10):
        scores = torch.as_tensor(scores, dtype=torch.float32).detach().reshape(-1)
        if scores.numel() != len(self.dataset):
            raise ValueError(
                f"Expected one hard score per sample ({len(self.dataset)}), "
                f"got {scores.numel()}"
            )

        # Every rank contributes its sample statistics. SampleManager normally
        # synchronizes these first, but reducing here keeps sampler state global
        # even when another caller supplies rank-local score vectors.
        if dist.is_available() and dist.is_initialized() and self.num_replicas > 1:
            device = torch.device("cpu")
            if dist.get_backend() == "nccl":
                device = torch.device("cuda", torch.cuda.current_device())
            scores = scores.to(device=device)
            dist.all_reduce(scores, op=dist.ReduceOp.SUM)
            scores /= self.num_replicas
            scores = scores.cpu()

        scores = scores.numpy()
        k = max(1, int(len(scores) * top_p / 100))
        indices = np.argpartition(scores, -k)[-k:]

        result = np.zeros_like(scores)
        result[indices] = scores[indices]
        self.sample_score = result

    def make_one_batch(self, step):
        selected_samples = []
        classes = []

        if step % self.accum == 0:
            same_k = random.randint(*self.same_num)

            if self.score_sampling_limit > 0:
                weights = self.sample_score * self.sample_available_mask

                if not np.any(weights):
                    self.score_sampling_limit = 0
                    weights = self.sample_available_mask

                score_sample = random.choices(self.indices, weights=weights, k=1)[0]
                cls = self.data_to_class[score_sample]
                self.score_sampling_limit -= 1

                self.cls = cls

                samples = self.data_by_class_copy[cls]
                samples.remove(score_sample)
                selected_samples.append(score_sample)
                self.sample_available_mask[score_sample] = False
                classes.append(cls)

                if not samples:
                    self.available_mask[cls] = False
                    same_k = len(selected_samples)

                for _ in range(same_k-1):
                    samples = self.data_by_class_copy[cls]

                    idx = random.randrange(len(samples))
                    selected_sample = samples.pop(idx)
                    selected_samples.append(selected_sample)
                    self.sample_available_mask[selected_sample] = False
                    classes.append(cls)

                    if not samples:
                        self.available_mask[cls] = False
                        same_k = len(selected_samples)
                        break

            else:
                cls = random.choices(self.unique_labels, weights=self.cls_diff_weights * self.available_mask, k=1)[0]
                self.cls = cls  

                for _ in range(same_k):
                    samples = self.data_by_class_copy[cls]

                    idx = random.randrange(len(samples))
                    # selected_samples.append(samples.pop(idx))
                    selected_sample = samples.pop(idx)
                    selected_samples.append(selected_sample)
                    self.sample_available_mask[selected_sample] = False
                    classes.append(cls)

                    if not samples:
                        self.available_mask[cls] = False
                        same_k = len(selected_samples)
                        break
        else:
            same_k = 0
            cls = self.cls


        while len(selected_samples) < self.batch_size:
            hard_k = random.randint(*self.hard_num)
            gaus_k = self.batch_size - same_k - hard_k

            selected_cls = (
                # random.choices(self.unique_labels, weights=self.sample_gauss_weights[cls] * self.available_mask, k=gaus_k)
                random.choices(self.unique_labels, k=gaus_k)
                + random.choices(self.unique_labels, weights=self.sample_diff_weights[cls] * self.available_mask, k=hard_k)
            )

            for sel_cls in selected_cls:
                samples = self.data_by_class_copy[sel_cls]

                if samples:
                    idx = random.randrange(len(samples))
                    # selected_samples.append(samples.pop(idx))
                    selected_sample = samples.pop(idx)
                    selected_samples.append(selected_sample)
                    self.sample_available_mask[selected_sample] = False
                    classes.append(sel_cls)

                    if not samples:
                        self.available_mask[sel_cls] = False

                    if len(selected_samples) >= self.batch_size:
                        break

        random.shuffle(selected_samples)
        # self.all_batches.append(selected_samples)
        # self.all_classes.append(classes)
        self.classes = classes
        # self.remain_class.append(np.array(self.unique_labels)[self.available_mask])

        return selected_samples

    
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
