import pandas as pd
import cv2
import math
import json
import torch
import numpy as np

import albumentations as A

from PIL import Image
from torch.utils.data import Dataset, DataLoader
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

def make_pretrain_dataset(args):
    train_df = pd.read_parquet(args.train)
    valid_df = pd.read_parquet(args.valid)
    print(f"Total Data: {len(train_df) + len(valid_df)}, Train: {len(train_df)}, Valid: {len(valid_df)}")

    transforms = get_preset(args.train_aug, image_size=args.image_size)
    valid_trans = get_preset(args.valid_aug, image_size=args.image_size)
    
    train_dataset = PretrainDataset(train_df.image.tolist(), train_df.target.tolist(), transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers, drop_last=True)
    train_loader.num_samples = len(train_dataset)
    train_loader.num_batches = len(train_loader)

    valid_dataset = PretrainValidDataset(valid_df.image.tolist(), valid_df.target.tolist(), transforms, valid_trans)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
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

        image1 = self.transforms(image=image)['image']
        image2 = self.transforms(image=image)['image']

        label = self.labels[idx]

        return image1, image2, label
    
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
    
class PretrainValidDataset(PretrainDataset):
    def __init__(self, images, labels, transforms, valid_transforms, df=None):
        super().__init__(images, labels, transforms, df)
        self.valid_transforms = valid_transforms

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image1 = self.transforms(image=image)['image']
        image2 = self.transforms(image=image)['image']
        image3 = self.valid_transforms(image=image)['image']

        label = self.labels[idx]

        return image1, image2, image3, label
    
    
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
    
