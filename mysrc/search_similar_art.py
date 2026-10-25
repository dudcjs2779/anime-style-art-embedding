

import shutil
import os
import argparse
import yaml
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import pandas as pd

from mysrc.utils.data import get_files
from mysrc.model import StyleFeatVisual
from mysrc.utils.aug_preset import get_preset
from mysrc.utils.precision import get_autocast, get_input_dtype
from mysrc.utils.config import read_config

class InferenceDataset(Dataset):
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        return image


def parse_args():
    parser = argparse.ArgumentParser()
    
    ## Data
    parser.add_argument(
        "--ref_path", 
        default=None,
        help=(
            "The path of reference arts for classifiy similar art."
            "All reference arts must be in the same style."
        )
    )
    parser.add_argument(
        "--data_path", 
        default=None,
        help="The path of datas for finding similar arts."
    )
    parser.add_argument(
        "--output_path",
        default="simliar_arts",
        help="Similar arts will be moved or copied to this path.",
    )
    parser.add_argument(
        "--transform", 
        choices=["simple_center_crop", "long_max_pad_valid"], 
        default="simple_center_crop"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=0
    )
    parser.add_argument(
        "--result_path", 
        default=None,
        help="if you set result path, you can skip the data inference."
    )

    ## feat
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold."
    )
    parser.add_argument(
        "--center", 
        choices=["mean", "median"], 
        default='median', 
        help="How calculate center of reference images."
    )
    parser.add_argument(
        "--how",
        choices=["move", "copy", "None"], 
        default="copy", 
        help="How to process the similar arts."
    )
    parser.add_argument(
        "--save_result", 
        type=bool,
        default=False, 
        help="The results are saved as a parquet file."
    )
    parser.add_argument(
        "--sfg", 
        type=bool, 
        default=False, 
        help="It inference style features. style features are saved to a csv file."
    )
    parser.add_argument(
        "--sfg_threshold",
        type=float,
        default=0.4,
        help="Style feature threshold."
    )

    ## Model
    parser.add_argument(
        "--precision", 
        choices=["amp", None], 
        default="amp",
        help="Model inference precision.",
    )
    parser.add_argument(
        "--input_dtype", 
        choices=["fp16", None], 
        default=None,
        help="Input image precision.",
    )

    ## config
    parser.add_argument(
        "--config",
        default=None,
    )

    return parser

def load_model(device):
    print("## Load Model")
    repo_id = "dudcjs2779/anime-style-art-embedding"

    # Download model files
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensor",
    )
    config_path = hf_hub_download(
        repo_id=repo_id,
        filename="config.json",
    )
    sfg_path = hf_hub_download(
        repo_id=repo_id,
        filename="style_feat_group.yaml",
    )

    # Load configuration and style feature groups
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    with open(sfg_path, encoding="utf-8") as f:
        sfg_encoder = yaml.safe_load(f)

    sfg_decoder = {value: key for key, value in sfg_encoder.items()}

    # Initialize model and load weights
    model = StyleFeatVisual(**config).to(device)
    state_dict = load_file(model_path)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del state_dict

    model.eval()

    return model, sfg_decoder

def main(args):
    print("## Prepare Datas")
    ref_images = get_files(args.ref_path)
    data_images = get_files(args.data_path)
    all_images = ref_images + data_images 
    ref_end = len(ref_images)

    if args.result_path:
        df = pd.read_parquet(args.result_path)
        df = df[df['is_ref'] == False]

        assert len(df) == len(data_images), "The length of data images does not match the length of the data images in the result file."
        embeddings = np.stack(df['embedding'].values)
        style_features = df['style_features'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).tolist()
        probs = df['style_features_probs'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).tolist()

    transforms = get_preset(args.transform)


    if args.result_path: 
        dataset = InferenceDataset(ref_images, transforms)
    else:
        dataset = InferenceDataset(all_images, transforms)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, sfg_decoder = load_model(device)

    input_dtype = get_input_dtype(args.input_dtype)
    autocast = get_autocast(args.precision) 

    all_style_feats, all_sfg_feat = [], []

    with torch.inference_mode():
        for images in tqdm(dataloader, desc="Inference images"):

            images = images.to(device=device, dtype=input_dtype, non_blocking=True)

            with autocast():
                output = model.inference(images)

                style_feat = output['style_cls']
                sfg_feat = output['sfg_cls']

                all_style_feats.extend(style_feat.cpu())
                all_sfg_feat.extend(sfg_feat.cpu())

        all_style_feats = torch.stack(all_style_feats)
        all_sfg_feat = torch.stack(all_sfg_feat)

    if args.result_path:
        embeddings = torch.tensor(embeddings, dtype=all_style_feats.dtype, device=all_style_feats.device)
        all_style_feats = torch.cat((all_style_feats, embeddings), dim=0)

    if args.center == "median":
        ref_center = all_style_feats[:ref_end].median(dim=0).values
    else:
        ref_center = all_style_feats[:ref_end].mean(dim=0)

    sims = (ref_center @ all_style_feats.T)
    data_sims = sims[ref_end:]
    indeices = np.where(data_sims > args.threshold)[0]
    similar_images = [data_images[idx] for idx in indeices]

    if args.how != "None":
        os.makedirs(args.output_path, exist_ok=True)
        for image in tqdm(similar_images, desc="Prcoessing similares"):
            filename = os.path.basename(image)
            dest = os.path.join(args.output_path, filename)

            if args.how =="move":
                shutil.move(image, dest)
            elif args.how =="copy":
                shutil.copy(image, dest)

    datas = {"image": all_images, "similarity": sims, "embedding": all_style_feats.numpy().tolist()}

    if args.sfg:
        all_sfg_probs = F.sigmoid(all_sfg_feat)

        decoded_sfgs = []
        sfgs_probs = []

        for sfg_probs in all_sfg_probs:
            indices = torch.where(sfg_probs > args.sfg_threshold)[0]
            decoded_sfgs.append([sfg_decoder[i.item()] for i in indices])
            sfgs_probs.append([sfg_probs[i].item() for i in indices])

        if args.result_path:
            decoded_sfgs = decoded_sfgs + style_features
            sfgs_probs = sfgs_probs+ probs

        datas['style_features'] = decoded_sfgs
        datas['style_features_probs'] = sfgs_probs

    if args.save_result:
        df = pd.DataFrame(datas)
        df['is_ref'] = False
        df.loc[:ref_end-1, 'is_ref'] = True

        os.makedirs(args.output_path, exist_ok=True)
        result_path = os.path.join(args.output_path, "similar.parquet")
        df.to_parquet(result_path, index=False)

    
if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    args = read_config(args, parser)
    
    main(args)
