import os
import torch
import numpy as np
from PIL import Image

ext_type = (".jpg", ".png", ".jpeg")

def get_files(directory, pair=True):
    images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(ext_type):
                image_path = os.path.abspath(os.path.join(root, file))
                images.append(image_path)

    print(f"{len(images)} images")
    return images

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    tensor = tensor * std + mean
    return tensor

def tensor_to_pil_image(tensor, mean=[0.6938988, 0.6323308, 0.6300056], std=[0.3098821,  0.31194064, 0.30595565]):
    if isinstance(tensor, np.ndarray):
        tensor = torch.tensor(tensor)

    # 텐서를 numpy 배열로 변환 (배치와 채널 확인)
    if len(tensor.shape) == 4:  # 배치가 포함된 텐서라면 배치를 제거
        tensor = tensor.squeeze(0)

    tensor = denormalize(tensor, mean=mean, std=std,)
    
    # 텐서가 [C, H, W] 형태일 때 [H, W, C]로 변환
    tensor = tensor.permute(1, 2, 0)

    # 텐서 값을 0-1 범위에서 0-255 범위로 스케일링
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    
    # numpy 배열을 PIL 이미지로 변환
    pil_image = Image.fromarray(tensor)

    return pil_image

def remove_zones(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(":zone.identifier"):
                filepath = os.path.abspath(os.path.join(root, file))
                os.remove(filepath)
                count += 1

    print(f"{count} file is removed")
