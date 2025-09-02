import random
import cv2
import albumentations as A
import numpy as np
import math
from typing import Any, Annotated
from pydantic import AfterValidator
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.crops import functional as fcrops
from albumentations.core.pydantic import BorderModeType
from albumentations.core.type_definitions import ColorType
from albumentations.core.transforms_interface import BaseTransformInitSchema
from albumentations.core.pydantic import (
    BorderModeType,
    InterpolationType,
    check_range_bounds,
    nondecreasing,
)

preset_list =["simple_center_crop", "art_style_aug1"]

class RandomSmallestMaxSize(A.SmallestMaxSize):
    def __init__(self, min_size = 512, max_size = 1024, interpolation = cv2.INTER_LINEAR, mask_interpolation = cv2.INTER_NEAREST, always_apply = None, p = 1):
        super().__init__(max_size, interpolation, mask_interpolation, always_apply, p)
        self.min_size = min_size
    
    def apply(self, img, max_size, **params):
        max_size = random.randint(self.min_size, max_size)
        return super().apply(img, max_size, **params)
    
class MyXYMasking(A.XYMasking):
    def apply(self, img, **params):
        val = np.random.choice([0, 255])
        self.fill = (val, val, val)
        return super().apply(img, fill_value=self.fill, **params)
    

class AutoPad(A.PadIfNeeded):
    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:
        long_side = max(params["shape"][:2])
        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = fgeometric.get_padding_params(
            image_shape=params["shape"][:2],
            min_height=long_side,
            min_width=long_side,
            pad_height_divisor=self.pad_height_divisor,
            pad_width_divisor=self.pad_width_divisor,
        )

        h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = fgeometric.adjust_padding_by_position(
            h_top=h_pad_top,
            h_bottom=h_pad_bottom,
            w_left=w_pad_left,
            w_right=w_pad_right,
            position=self.position,
            py_random=self.py_random,
        )

        return {
            "pad_top": h_pad_top,
            "pad_bottom": h_pad_bottom,
            "pad_left": w_pad_left,
            "pad_right": w_pad_right,
        }
    
class RandomResizedCropWithPad(A.RandomResizedCrop):
    class InitSchema(BaseTransformInitSchema):  # 스키마가 없으니까 padding이 빨간색으로 표시되는 버그가 발생함.
        scale: Annotated[tuple[float, float], AfterValidator(check_range_bounds(0, 1)), AfterValidator(nondecreasing)]
        ratio: Annotated[
            tuple[float, float],
            AfterValidator(check_range_bounds(0, None)),
            AfterValidator(nondecreasing),
        ]
        size: Annotated[tuple[int, int], AfterValidator(check_range_bounds(1, None))]
        interpolation: InterpolationType
        mask_interpolation: InterpolationType
        fill: ColorType
        fill_mask: ColorType
        border_mode: BorderModeType

    def __init__(
        self,
        size: tuple[int, int],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        fill: ColorType = 0,
        fill_mask: ColorType = 0,
        border_mode: BorderModeType = cv2.BORDER_CONSTANT,
        p: float = 1.0,
    ):
        super().__init__(
            size=size,
            interpolation=interpolation,
            mask_interpolation=mask_interpolation,
            p=p,
        )
        self.scale = scale
        self.ratio = ratio
        self.fill = fill
        self.fill_mask = fill_mask
        self.border_mode = border_mode

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        image_shape = params["shape"][:2]
        image_height, image_width = image_shape

        area = image_height * image_width

        for _ in range(5):
            scale = self.py_random.uniform(*self.scale)
            target_area = scale * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(self.py_random.uniform(*log_ratio))

            width = int(round(math.sqrt(target_area * aspect_ratio)))
            height = int(round(math.sqrt(target_area / aspect_ratio)))

            left_right_pad = 0
            top_bottom_pad = 0
            pad_image_width = image_width
            pad_image_height = image_height

            if image_width < width:
                width_diff = width - image_width
                left_right_pad = (width_diff + 1) // 2
                pad_image_width = image_width + left_right_pad * 2

            if image_height < height:
                height_diff = height - image_height
                top_bottom_pad = (height_diff + 1) // 2
                pad_image_height = image_height + top_bottom_pad * 2 

            pad_image_shape = (pad_image_height, pad_image_width)

            if 0 < width <= pad_image_width and 0 < height <= pad_image_height:
                i = self.py_random.randint(0, pad_image_height - height)
                j = self.py_random.randint(0, pad_image_width - width)

                h_start = i * 1.0 / (pad_image_height - height + 1e-10)
                w_start = j * 1.0 / (pad_image_width - width + 1e-10)

                crop_shape = (height, width)

                crop_coords = fcrops.get_crop_coords(pad_image_shape, crop_shape, h_start, w_start)

                return {"crop_coords": crop_coords, 'pad_top': top_bottom_pad, 'pad_bottom': top_bottom_pad, 'pad_left': left_right_pad, 'pad_right': left_right_pad}

        # Fallback to central crop
        in_ratio = image_width / image_height
        if in_ratio < min(self.ratio):
            width = image_width
            height = int(round(image_width / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            height = image_height
            width = int(round(height * max(self.ratio)))
        else:  # whole image
            width = image_width
            height = image_height

        i = (image_height - height) // 2
        j = (image_width - width) // 2

        h_start = i * 1.0 / (image_height - height + 1e-10)
        w_start = j * 1.0 / (image_width - width + 1e-10)

        crop_shape = (height, width)

        crop_coords = fcrops.get_crop_coords(image_shape, crop_shape, h_start, w_start)

        print("skiped")
        return {"crop_coords": crop_coords, 'pad_top': top_bottom_pad, 'pad_bottom': top_bottom_pad, 'pad_left': left_right_pad, 'pad_right': left_right_pad}
    
    def apply(
        self,
        img: np.ndarray,
        crop_coords: tuple[int, int, int, int],
        pad_top: int,
        pad_bottom: int,
        pad_left: int,
        pad_right: int,
        **params: Any,
    ) -> np.ndarray:
        img = fgeometric.pad_with_params(img, pad_top, pad_bottom, pad_left, pad_right, border_mode=self.border_mode, value=self.fill)
        crop = fcrops.crop(img, *crop_coords)
        resize = fgeometric.resize(crop, self.size, self.interpolation)
        return resize


def list_preset():
    return preset_list

def get_preset(preset_name, image_size=[448, 448]):
    mean = [0.6938988, 0.6323308, 0.6300056] 
    std = [0.3098821,  0.31194064, 0.30595565]
    w, h = image_size

    if preset_name == "simple_center_crop":
        transforms = A.Compose([
            A.SmallestMaxSize(max_size=max(w, h), interpolation=cv2.INTER_AREA),
            A.CenterCrop(width=w, height=h),
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ])

    elif preset_name == "art_style_aug1":
        transforms1 = A.Compose([
            A.SmallestMaxSize(max_size=max(w, h), interpolation=cv2.INTER_AREA),
            A.OneOf([
                A.RandomRotate90(p=0.2),
                A.Rotate(limit=(-180, 180), p=0.8, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]),
            ], p=0.8),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),

            A.RandomScale(scale_limit=(0, 0.5), p=0.7),
            A.RandomCrop(width=w, height=h),

            A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.3), saturation=(0.0, 1.7), hue=(-0.1, 0.1), p=0.8),
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(114, 224),
                hole_width_range=(114, 224),
                p=0.5)
        ], p=0.5)

        transforms2 = A.Compose([
            RandomSmallestMaxSize(min_size = 1024, max_size=3072, interpolation=cv2.INTER_AREA),
            A.OneOf([
                A.RandomRotate90(p=0.2),
                A.Rotate(limit=(-180, 180), p=0.8, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]),
            ], p=0.8),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),

            A.RandomCrop(width=w, height=h),

            A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.3), saturation=(0.0, 1.7), hue=(-0.1, 0.1), p=0.8),
            # A.CoarseDropout(
            #     num_holes_range=(1, 3),
            #     hole_height_range=(114, 224),
            #     hole_width_range=(114, 224),
            #     p=0.5)
        ], p=0.5)

        transforms = A.Compose([
            A.OneOf([
                transforms1,
                transforms2,
            ], p=1.0),
            
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ])
    elif preset_name == "bad_aug1":
        transforms = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]),
            ], p=1.0),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),

            A.RandomScale(scale_limit=(0, 0.2), p=0.5),
            A.RandomCrop(width=w, height=h),

            A.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.3), saturation=(0.8, 1.7), hue=(-0.1, 0.1), p=0.7),
            A.CoarseDropout(
                num_holes_range=(1, 5),
                hole_height_range=(64, 224),
                hole_width_range=(64, 224),
                p=0.5),

            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

    elif preset_name == "long_max_pad_train":
        transforms = A.Compose([
        A.LongestMaxSize(max_size=max(w, h), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
        A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
        A.OneOf([
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]),
        ], p=0.9),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),

        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=False, p=0.8),

        A.Normalize(mean=mean, std=std, max_pixel_value=255),
        ToTensorV2(),
    ], p=1.0)

    elif preset_name == "long_max_pad_rot":
        transforms = A.Compose([
        A.LongestMaxSize(max_size=max(w, h), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
        A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
        A.OneOf([
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, value=[255, 255, 255]),
        ], p=0.9),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),

        A.Normalize(mean=mean, std=std, max_pixel_value=255),
        ToTensorV2(),
    ], p=1.0)
        
    elif preset_name == "long_max_pad_valid":
        transforms = A.Compose([
        A.LongestMaxSize(max_size=max(w, h), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
        A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
        A.Normalize(mean=mean, std=std, max_pixel_value=255),
        ToTensorV2(),
    ], p=1.0)
        
    elif preset_name == "rancrop_rot":
        transforms = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
            ], p=0.7),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),
            A.RandomCrop(h, w, border_mode=cv2.BORDER_CONSTANT, p=1.0),

            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

    elif preset_name == "train_preset01":
        transforms = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
            ], p=0.7),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),

            A.OneOf([
                A.RandomCrop(448, 448, border_mode=cv2.BORDER_CONSTANT, p=0.8),
                A.Compose([
                    A.LongestMaxSize(max_size=max(448, 448), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=448, min_width=448, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                ], p=0.2),
            ], p=1.0),
            A.ToGray(p=0.1),

            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)
        
    # 640 기반
    elif preset_name == "train_preset02":
        transforms = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
            ], p=0.7),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),

            A.OneOf([
                A.Compose([
                    A.LongestMaxSize(max_size=max(448, 448), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=448, min_width=448, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                ], p=0.2),

                A.Compose([
                    A.LongestMaxSize(max_size=max(640, 640), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                    A.RandomScale(scale_limit=[-0.3, 0.0], interpolation=cv2.INTER_CUBIC, p=1.0),
                    A.RandomCrop(448, 448, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                ], p=0.8),

            ], p=1.0),
                                                            
            A.ToGray(p=0.1),

            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

    # 768 기반
    elif preset_name == "train_preset03":
        transforms = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
            ], p=0.7),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),

            A.OneOf([
                A.Compose([
                    A.LongestMaxSize(max_size=max(448, 448), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=448, min_width=448, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                ], p=0.2),

                A.Compose([
                    A.LongestMaxSize(max_size=max(640, 640), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                    A.RandomScale(scale_limit=[-0.3, 0.0], interpolation=cv2.INTER_CUBIC, p=1.0),
                    A.RandomCrop(448, 448, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                ], p=0.8),

            ], p=1.0),
                                                            
            A.ToGray(p=0.1),

            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

    # ColorJitter 도입
    elif preset_name == "train_preset04":
        transforms = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
            ], p=0.7),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),

            A.OneOf([
                A.Compose([
                    A.LongestMaxSize(max_size=max(448, 448), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=448, min_width=448, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                ], p=0.2),

                A.Compose([
                    A.LongestMaxSize(max_size=max(640, 640), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                    A.RandomScale(scale_limit=[-0.3, 0.0], interpolation=cv2.INTER_CUBIC, p=1.0),
                    A.RandomCrop(448, 448, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                ], p=0.8),
            ], p=1.0),

            A.OneOf([
                # brightness 기반 증강
                A.OneOf([
                    A.ColorJitter(
                        brightness=[1.0, 1.05], 
                        contrast=[0.95, 1.05],    
                        saturation=[0.8, 1.2],  
                        hue=[0.0, 0.0],     
                        p=0.4
                    ),

                    A.ColorJitter(
                        brightness=[0.7, 1.0], 
                        contrast=[0.8, 1.2],   
                        saturation=[0.8, 1.2],  
                        hue=[0.0, 0.0],       
                        p=0.4
                    ),

                    A.ColorJitter(
                        brightness=[1.0, 1.05],
                        contrast=[0.95, 1.05],    
                        saturation=[0, 0.8],  
                        hue=[0.0, 0.0],       
                        p=0.1
                    ),

                    A.ColorJitter(
                        brightness=[0.7, 1.0], 
                        contrast=[0.8, 1.2],    
                        saturation=[0, 0.8],  
                        hue=[0.0, 0.0],        
                        p=0.1
                    ),
                ], p=0.5),

                # contrast 기반 증강
                A.OneOf([
                    A.ColorJitter(
                        brightness=[0.95, 1.05], 
                        contrast=[1.0, 1.1],   
                        saturation=[0.8, 1.2],  
                        hue=[0.0, 0.0],         
                        p=0.4
                    ),

                    A.ColorJitter(
                        brightness=[0.9, 1.1],  
                        contrast=[0.7, 1.0],    
                        saturation=[0.8, 1.2], 
                        hue=[0.0, 0.0],         
                        p=0.4
                    ),

                    A.ColorJitter(
                        brightness=[0.95, 1.05],  
                        contrast=[1.0, 1.1],    
                        saturation=[0.0, 0.8],  
                        hue=[0.0, 0.0],         
                        p=0.1
                    ),

                    A.ColorJitter(
                        brightness=[0.9, 1.1],  
                        contrast=[0.7, 1.0],   
                        saturation=[0.0, 0.8],  
                        hue=[0.0, 0.0],         
                        p=0.1
                    ),
                ], p=0.5),
            ], p=0.5),
                                                            
            A.ToGray(p=0.1),

            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

    # ColorJitter 도입 Hue 도입
    elif preset_name == "train_preset05":
        transforms = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
            ], p=0.7),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),

            A.OneOf([
                A.Compose([
                    A.LongestMaxSize(max_size=max(448, 448), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=448, min_width=448, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                ], p=0.2),

                A.Compose([
                    A.LongestMaxSize(max_size=max(640, 640), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                    A.RandomScale(scale_limit=[-0.3, 0.0], interpolation=cv2.INTER_CUBIC, p=1.0),
                    A.RandomCrop(448, 448, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                ], p=0.8),
            ], p=1.0),

            A.OneOf([
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
            ], p=0.7),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),

            A.OneOf([
                A.Compose([
                    A.LongestMaxSize(max_size=max(448, 448), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=448, min_width=448, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                ], p=0.2),

                A.Compose([
                    A.LongestMaxSize(max_size=max(640, 640), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                    A.RandomScale(scale_limit=[-0.3, 0.0], interpolation=cv2.INTER_CUBIC, p=1.0),
                    A.RandomCrop(448, 448, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                ], p=0.8),
            ], p=1.0),

            A.Compose([
                A.ColorJitter(
                    brightness=[1.0, 1.0],  # [0.7, 1.0]
                    contrast=[1.0, 1.0],    # [0.7, 1.0]
                    saturation=[1.0, 1.0],  # [0.0, 1.2]
                    hue=[-0.5, 0.5],         # 0.3, 0.4
                    p=0.1
                ),

                A.OneOf([
                    # brightness 기반 증강
                    A.OneOf([
                        A.ColorJitter(
                            brightness=[1.0, 1.05], 
                            contrast=[0.95, 1.05],    
                            saturation=[0.8, 1.2],  
                            hue=[0.0, 0.0],     
                            p=0.4
                        ),

                        A.ColorJitter(
                            brightness=[0.7, 1.0], 
                            contrast=[0.8, 1.2],   
                            saturation=[0.8, 1.2],  
                            hue=[0.0, 0.0],       
                            p=0.4
                        ),

                        A.ColorJitter(
                            brightness=[1.0, 1.05],
                            contrast=[0.95, 1.05],    
                            saturation=[0, 0.8],  
                            hue=[0.0, 0.0],       
                            p=0.1
                        ),

                        A.ColorJitter(
                            brightness=[0.7, 1.0], 
                            contrast=[0.8, 1.2],    
                            saturation=[0, 0.8],  
                            hue=[0.0, 0.0],        
                            p=0.1
                        ),
                    ], p=0.5),

                    # contrast 기반 증강
                    A.OneOf([
                        A.ColorJitter(
                            brightness=[0.95, 1.05], 
                            contrast=[1.0, 1.1],   
                            saturation=[0.8, 1.2],  
                            hue=[0.0, 0.0],         
                            p=0.4
                        ),

                        A.ColorJitter(
                            brightness=[0.9, 1.1],  
                            contrast=[0.7, 1.0],    
                            saturation=[0.8, 1.2], 
                            hue=[0.0, 0.0],         
                            p=0.4
                        ),

                        A.ColorJitter(
                            brightness=[0.95, 1.05],  
                            contrast=[1.0, 1.1],    
                            saturation=[0.0, 0.8],  
                            hue=[0.0, 0.0],         
                            p=0.1
                        ),

                        A.ColorJitter(
                            brightness=[0.9, 1.1],  
                            contrast=[0.7, 1.0],   
                            saturation=[0.0, 0.8],  
                            hue=[0.0, 0.0],         
                            p=0.1
                        ),
                    ], p=0.5),
                ], p=0.8),

                A.ToGray(p=0.1),
            ], p=0.7),

            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

    # ColorJitter, Hue, PlasmaShadow, Posterize, ToGray
    elif preset_name == "train_preset06":
        transforms = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
            ], p=0.7),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),

            A.OneOf([
                A.Compose([
                    A.LongestMaxSize(max_size=max(448, 448), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=448, min_width=448, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                ], p=0.2),

                A.Compose([
                    A.LongestMaxSize(max_size=max(640, 640), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                    A.RandomScale(scale_limit=[-0.3, 0.0], interpolation=cv2.INTER_CUBIC, p=1.0),
                    A.RandomCrop(448, 448, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                ], p=0.8),
            ], p=1.0),

            A.Compose([
                A.OneOf([
                    # brightness 기반 증강
                    A.OneOf([
                        A.ColorJitter(
                            brightness=[1.0, 1.05], 
                            contrast=[0.95, 1.05],    
                            saturation=[0.8, 1.2],  
                            hue=[0.0, 0.0],     
                            p=0.4
                        ),

                        A.ColorJitter(
                            brightness=[0.7, 1.0], 
                            contrast=[0.8, 1.2],   
                            saturation=[0.8, 1.2],  
                            hue=[0.0, 0.0],       
                            p=0.4
                        ),

                        A.ColorJitter(
                            brightness=[1.0, 1.05],
                            contrast=[0.95, 1.05],    
                            saturation=[0, 0.8],  
                            hue=[0.0, 0.0],       
                            p=0.1
                        ),

                        A.ColorJitter(
                            brightness=[0.7, 1.0], 
                            contrast=[0.8, 1.2],    
                            saturation=[0, 0.8],  
                            hue=[0.0, 0.0],        
                            p=0.1
                        ),
                    ], p=0.5),

                    # contrast 기반 증강
                    A.OneOf([
                        A.ColorJitter(
                            brightness=[0.95, 1.05], 
                            contrast=[1.0, 1.1],   
                            saturation=[0.8, 1.2],  
                            hue=[0.0, 0.0],         
                            p=0.4
                        ),

                        A.ColorJitter(
                            brightness=[0.9, 1.1],  
                            contrast=[0.7, 1.0],    
                            saturation=[0.8, 1.2], 
                            hue=[0.0, 0.0],         
                            p=0.4
                        ),

                        A.ColorJitter(
                            brightness=[0.95, 1.05],  
                            contrast=[1.0, 1.1],    
                            saturation=[0.0, 0.8],  
                            hue=[0.0, 0.0],         
                            p=0.1
                        ),

                        A.ColorJitter(
                            brightness=[0.9, 1.1],  
                            contrast=[0.7, 1.0],   
                            saturation=[0.0, 0.8],  
                            hue=[0.0, 0.0],         
                            p=0.1
                        ),
                    ], p=0.5),
                ], p=0.8),

                A.Compose([
                    A.ColorJitter(  # 오직 Hue
                        brightness=[1.0, 1.0],
                        contrast=[1.0, 1.0],
                        saturation=[1.0, 1.0],
                        hue=[-0.5, 0.5],
                        p=0.4
                    ),

                    A.Posterize(    # 옛날 포스터 스캔 노이즈
                        num_bits=[4, 7],
                        p=0.4),

                    A.PlasmaShadow( # 점박이
                        shadow_intensity_range=[0.0, 0.1],
                        plasma_size=256,
                        roughness=1,
                        p=0.4),
                        
                    A.ToGray(p=0.25),

                ], p=0.2),

            ], p=0.75),

            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

    # ColorJitter, Hue, RGBShift, PlasmaShadow, AdditiveNoise, Posterize, ZoomBlur, AdvancedBlur, MyXYMasking, ToGray
    elif preset_name == "train_preset07":
        transforms = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
            ], p=0.7),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),

            A.OneOf([
                A.Compose([
                    A.LongestMaxSize(max_size=max(448, 448), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=448, min_width=448, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                ], p=0.2),

                A.Compose([
                    A.LongestMaxSize(max_size=max(640, 640), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                    A.RandomScale(scale_limit=[-0.3, 0.0], interpolation=cv2.INTER_CUBIC, p=1.0),
                    A.RandomCrop(448, 448, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                ], p=0.8),
            ], p=1.0),

            # A.LongestMaxSize(max_size=max(1024, 1024), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤

            # 80% 이내에서 증강
            A.Compose([
                # 유닛 1
                A.OneOf([
                    # brightness 기반 증강
                    A.OneOf([
                        A.ColorJitter(
                            brightness=[1.0, 1.05], 
                            contrast=[0.95, 1.05],    
                            saturation=[0.8, 1.2],  
                            hue=[0.0, 0.0],     
                            p=0.4
                        ),

                        A.ColorJitter(
                            brightness=[0.7, 1.0], 
                            contrast=[0.8, 1.2],   
                            saturation=[0.8, 1.2],  
                            hue=[0.0, 0.0],       
                            p=0.4
                        ),

                        A.ColorJitter(
                            brightness=[1.0, 1.05],
                            contrast=[0.95, 1.05],    
                            saturation=[0, 0.8],  
                            hue=[0.0, 0.0],       
                            p=0.1
                        ),

                        A.ColorJitter(
                            brightness=[0.7, 1.0], 
                            contrast=[0.8, 1.2],    
                            saturation=[0, 0.8],  
                            hue=[0.0, 0.0],        
                            p=0.1
                        ),
                    ], p=0.5),

                    # contrast 기반 증강
                    A.OneOf([
                        A.ColorJitter(
                            brightness=[0.95, 1.05], 
                            contrast=[1.0, 1.1],   
                            saturation=[0.8, 1.2],  
                            hue=[0.0, 0.0],         
                            p=0.4
                        ),

                        A.ColorJitter(
                            brightness=[0.9, 1.1],  
                            contrast=[0.7, 1.0],    
                            saturation=[0.8, 1.2], 
                            hue=[0.0, 0.0],         
                            p=0.4
                        ),

                        A.ColorJitter(
                            brightness=[0.95, 1.05],  
                            contrast=[1.0, 1.1],    
                            saturation=[0.0, 0.8],  
                            hue=[0.0, 0.0],         
                            p=0.1
                        ),

                        A.ColorJitter(
                            brightness=[0.9, 1.1],  
                            contrast=[0.7, 1.0],   
                            saturation=[0.0, 0.8],  
                            hue=[0.0, 0.0],         
                            p=0.1
                        ),
                    ], p=0.5),
                ], p=0.5),

                # 유닛 2
                A.Compose([
                    A.OneOf([
                        A.ColorJitter(
                            brightness=[1.0, 1.0],
                            contrast=[1.0, 1.0],
                            saturation=[1.0, 1.0],
                            hue=[-0.5, 0.5],
                            p=0.5),
                        A.RGBShift(
                            r_shift_limit=[-20, 20],
                            g_shift_limit=[-20, 20],
                            b_shift_limit=[-20, 20],
                            p=0.5),
                    ], p=0.3),

                    A.Posterize(
                        num_bits=[4, 7],
                        p=0.3),

                    # 블러 후 노이즈 또는 노이즈 후 블러
                    A.OneOf([
                        A.Compose([
                            A.OneOf([
                                A.ZoomBlur(
                                    max_factor=[1.0, 1.03],
                                    step_factor=[0.01, 0.03],
                                    p=0.3),

                                A.AdvancedBlur(
                                    blur_limit=[5, 13],
                                    sigma_x_limit=[0.2, 1.0],
                                    sigma_y_limit=[0.2, 1.0],
                                    rotate_limit=[-90, 90],
                                    beta_limit=[0.5, 8],
                                    noise_limit=[0.9, 1.1],
                                    p=0.7),
                            ], p=0.4),
                            
                            A.OneOf([
                                A.PlasmaShadow(
                                    shadow_intensity_range=[0.0, 0.1],
                                    plasma_size=256,
                                    roughness=1,
                                    p=0.5),

                                A.AdditiveNoise(
                                    noise_type="gaussian",
                                    spatial_mode="shared",
                                    noise_params={"mean_range":[0,0],"std_range":[0.0, 0.03]},
                                    approximation=1,
                                    p=0.5)
                            ],p=0.6),
                        ], p=0.5),

                        A.Compose([
                            A.OneOf([
                                A.PlasmaShadow(
                                    shadow_intensity_range=[0.0, 0.1],
                                    plasma_size=256,
                                    roughness=1,
                                    p=0.5),

                                A.AdditiveNoise(
                                    noise_type="gaussian",
                                    spatial_mode="shared",
                                    noise_params={"mean_range":[0,0],"std_range":[0.0, 0.03]},
                                    approximation=1,
                                    p=0.5)
                            ],p=0.6),

                            A.OneOf([
                                A.ZoomBlur(
                                    max_factor=[1.0, 1.03],
                                    step_factor=[0.01, 0.03],
                                    p=0.3),

                                A.AdvancedBlur(
                                    blur_limit=[5, 13],
                                    sigma_x_limit=[0.2, 1.0],
                                    sigma_y_limit=[0.2, 1.0],
                                    rotate_limit=[-90, 90],
                                    beta_limit=[0.5, 8],
                                    noise_limit=[0.9, 1.1],
                                    p=0.7),
                            ], p=0.4),
                        ], p=0.5),
                    ], 0.3),

                    MyXYMasking(
                        num_masks_x=[0, 3],
                        num_masks_y=[0, 3],
                        mask_x_length=[1, 6],
                        mask_y_length=[1, 6],
                        p=0.4,
                    ),

                    A.ToGray(p=0.2),
                ], p=0.5),
            ], p=0.8),

            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

    # train_preset07의 간소화 버전
    elif preset_name == "train_preset08":
        transforms = A.Compose([
            A.OneOf([
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
            ], p=0.7),
            A.HorizontalFlip(p=0.2),
            A.VerticalFlip(p=0.2),

            A.OneOf([
                A.Compose([
                    A.LongestMaxSize(max_size=max(448, 448), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=448, min_width=448, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                ], p=0.2),

                A.Compose([
                    A.LongestMaxSize(max_size=max(640, 640), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                    A.RandomScale(scale_limit=[-0.3, 0.0], interpolation=cv2.INTER_CUBIC, p=1.0),
                    A.RandomCrop(448, 448, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                ], p=0.8),
            ], p=1.0),

            # A.LongestMaxSize(max_size=max(1024, 1024), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤

            A.Compose([
                A.ColorJitter(
                    brightness=[0.9, 1.1], 
                    contrast=[0.8, 1.2],    
                    saturation=[0.8, 1.2],  
                    hue=[0.0, 0.0],     
                    p=0.4),

                A.RGBShift(
                    r_shift_limit=[-20, 20],
                    g_shift_limit=[-20, 20],
                    b_shift_limit=[-20, 20],
                    p=0.3),

                A.Posterize(
                    num_bits=[4, 7],
                    p=0.2),

                A.OneOf([
                    # 선 블러 후 노이즈
                    A.Compose([
                        A.OneOf([
                            A.ZoomBlur(
                                max_factor=[1.0, 1.03],
                                step_factor=[0.01, 0.03],
                                p=0.3),

                            A.AdvancedBlur(
                                blur_limit=[5, 13],
                                sigma_x_limit=[0.2, 1.0],
                                sigma_y_limit=[0.2, 1.0],
                                rotate_limit=[-90, 90],
                                beta_limit=[0.5, 8],
                                noise_limit=[0.9, 1.1],
                                p=0.7),
                        ], p=0.4),
                        
                        A.OneOf([
                            A.PlasmaShadow(
                                shadow_intensity_range=[0.0, 0.1],
                                roughness=1,
                                p=0.5),

                            A.AdditiveNoise(
                                noise_type="gaussian",
                                spatial_mode="shared",
                                noise_params={"mean_range":[0,0],"std_range":[0.0, 0.03]},
                                approximation=1,
                                p=0.5)
                        ],p=0.6),
                    ], p=0.5),

                    # 선 노이즈 후 블러
                    A.Compose([
                        A.OneOf([
                            A.PlasmaShadow(
                                shadow_intensity_range=[0.0, 0.1],
                                roughness=1,
                                p=0.5),

                            A.AdditiveNoise(
                                noise_type="gaussian",
                                spatial_mode="shared",
                                noise_params={"mean_range":[0,0],"std_range":[0.0, 0.03]},
                                approximation=1,
                                p=0.5)
                        ],p=0.6),

                        A.OneOf([
                            A.ZoomBlur(
                                max_factor=[1.0, 1.03],
                                step_factor=[0.01, 0.03],
                                p=0.3),

                            A.AdvancedBlur(
                                blur_limit=[5, 13],
                                sigma_x_limit=[0.2, 1.0],
                                sigma_y_limit=[0.2, 1.0],
                                rotate_limit=[-90, 90],
                                beta_limit=[0.5, 8],
                                noise_limit=[0.9, 1.1],
                                p=0.7),
                        ], p=0.4),
                    ], p=0.5),
                ], 0.4),

                MyXYMasking(
                    num_masks_x=[0, 3],
                    num_masks_y=[0, 3],
                    mask_x_length=[1, 6],
                    mask_y_length=[1, 6],
                    p=0.2,
                ),

                A.ToGray(p=0.15),
            ], p=0.8),

            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)



    elif preset_name == "rancrop":
        transforms = A.Compose([
            A.RandomCrop(h, w, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

    elif preset_name == "pretrain_base":
        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),

            A.OneOf([
                RandomResizedCropWithPad(
                    size=[h, w],
                    scale=[0.05, 1.0],
                    ratio=[0.75, 1.3333333333333333],
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    p=0.6),

                A.Compose([
                    AutoPad(border_mode=cv2.BORDER_CONSTANT, fill=255),
                    A.RandomResizedCrop(
                        size=[h, w],
                        scale=[0.33, 1.0],
                        ratio=[0.75, 1.3333333333333333],
                        interpolation=cv2.INTER_CUBIC,
                        p=1.0),
                ], p=0.4),
            ], p=1.0),

             A.OneOf([
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
            ], p=0.7),
           
            A.ToGray(p=0.1),

            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

    elif preset_name == "pretrain02":
        defualt_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),

            A.OneOf([
                RandomResizedCropWithPad(
                    size=[h, w],
                    scale=[0.05, 1.0],
                    ratio=[0.75, 1.3333333333333333],
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    p=0.6),

                A.Compose([
                    AutoPad(border_mode=cv2.BORDER_CONSTANT, fill=255),
                    A.RandomResizedCrop(
                        size=[h, w],
                        scale=[0.33, 1.0],
                        ratio=[0.75, 1.3333333333333333],
                        interpolation=cv2.INTER_CUBIC,
                        p=1.0),
                ], p=0.4),
            ], p=1.0),

             A.OneOf([
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=(-180, 180), p=0.7, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
            ], p=0.7),

            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
                p=0.8
            ),
            A.ToGray(p=0.2),
        ])

        aug_trans1 = A.Compose([
            A.AdditiveNoise(
                noise_type="gaussian",
                spatial_mode="shared",
                noise_params={"mean_range":[0,0],"std_range":[0.01,0.08]},
                approximation=0.8,
                p=0.1
            ),

            A.OneOf([
                A.AdvancedBlur(
                    blur_limit=[5, 15],
                    sigma_x_limit=[0.5, 2.2],
                    sigma_y_limit=[0.5, 2.2],
                    rotate_limit=[-90, 90],
                    beta_limit=[0.5, 8.0],
                    noise_limit=[0.75, 1.25],
                    p=0.5
                ),
                A.GaussianBlur(
                    blur_limit=0,
                    sigma_limit=[0.5, 2.0],
                    p=0.5
                ),
            ], p=0.9),
        ], p=1.0)



        aug_trans2 = A.Compose([
            A.OneOf([
                A.AdvancedBlur(
                    blur_limit=[5, 15],
                    sigma_x_limit=[0.5, 2.2],
                    sigma_y_limit=[0.5, 2.2],
                    rotate_limit=[-90, 90],
                    beta_limit=[0.5, 8.0],
                    noise_limit=[0.75, 1.25],
                    p=0.5
                ),
                A.GaussianBlur(
                    blur_limit=0,
                    sigma_limit=[0.5, 2.0],
                    p=0.5
                ),
            ], p=0.1),

            A.AdditiveNoise(
                noise_type="gaussian",
                spatial_mode="shared",
                noise_params={"mean_range":[0,0],"std_range":[0.01,0.08]},
                approximation=0.8,
                p=0.5
            ),

            A.Solarize([0.0, 0.7], p=0.2),
        ], p=1.0)

        end_trans = A.Compose([
            A.CoarseDropout(
                num_holes_range=[2, 5],
                hole_height_range=[0.1, 0.5],
                hole_width_range=[0.1, 0.5],
                fill=0,
                p=0.5
            ),

            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

        transforms1 = A.Compose([defualt_trans, aug_trans1, end_trans])
        transforms2 = A.Compose([defualt_trans, aug_trans2, end_trans])
        transforms = {"trans1": transforms1, "trans2": transforms2}

    return transforms



