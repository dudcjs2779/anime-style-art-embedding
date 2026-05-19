import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from mysrc.utils.augmentation import (
    ShiftScaleCenterCrop,
    RandomResizedCropAutoScale, 
    RandomResizedCropWithPad, 
    RandomSmallestMaxSize, 
    AutoPad, 
    MyXYMasking, GridNCrops, 
    BatchNormalize,
    BatchToTensor,
    )

preset_list =["simple_center_crop", "art_style_aug1"]

def list_preset():
    return preset_list

def get_preset(preset_name, image_size=[448, 448], mean=None, std=None):
    if mean is None:
        mean = [0.5, 0.5, 0.5] 
        # mean = [0.6938988, 0.6323308, 0.6300056] 
    if std is None:
        std = [0.5, 0.5, 0.5]
        # std = [0.3098821, 0.31194064, 0.30595565]

    w, h = image_size

    if preset_name == "simple_center_crop":
        transforms = A.Compose([
            A.SmallestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_AREA),
            A.CenterCrop(h, w),
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ])

    elif preset_name == "RandomResizedCropWithPad":
        transforms = A.Compose([
            RandomResizedCropWithPad(
                size=[h, w],
                scale=[0.5, 1.0],
                ratio=[1.0, 1.0],
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                fill=255,
                p=1.0),
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ])

    elif preset_name == "grid_crop_valid":
        transforms = A.Compose([
            GridNCrops(
                (h/2, w/2),
                crop_num=6,
                grid_scale=9,
                interpolation=cv2.INTER_CUBIC,
                fill=255,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0,
            ),
            BatchNormalize(mean=mean, std=std, max_pixel_value=255),
            BatchToTensor(),
        ])

    elif preset_name == "style_train01":
        shared_trans = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=[-0.0625, 0.0625],
                scale_limit=[0.0, 0.15],
                rotate_limit=[-5, 5],
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                rotate_method="ellipse",
                fill=255,
                p=0.9,
            ),
            A.LongestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_CUBIC, p=1.0),
            A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, fill=255, p=1.0),
            A.HorizontalFlip(p=0.5),
        ], additional_targets={'image2': 'image'}, p=1.0)

        style_trans = A.Compose([
            A.ColorJitter(
                brightness=0.05,
                contrast=0.05,
                saturation=0.05,
                hue=0.0,
                p=1.0
            ),

            # A.RGBShift(
            #     r_shift_limit=[-10, 10],
            #     g_shift_limit=[-10, 10],
            #     b_shift_limit=[-10, 10],
            #     p=0.9
            # ),
        ], p=1.0)

        end_trans = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ])
        transforms = {"shared_trans": shared_trans, "style_trans": style_trans, 'end_trans': end_trans}

    elif preset_name == "style_train02":
        shared_trans = A.Compose([
            A.HorizontalFlip(p=0.5),
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
            A.RandomRotate90(p=0.4),
            A.Rotate(limit=(-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
        ], additional_targets={'image2': 'image'}, p=1.0)

        style_trans = A.Compose([
            A.ColorJitter(
                brightness=0.05,
                contrast=0.05,
                saturation=0.05,
                hue=0.0,
                p=1.0
            ),

            A.RGBShift(
                r_shift_limit=[-10, 10],
                g_shift_limit=[-10, 10],
                b_shift_limit=[-10, 10],
                p=0.9
            ),
        ], p=1.0)

        end_trans = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ])

        valid_trans = A.Compose([ # Center Crop
            A.SmallestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_CUBIC),
            A.CenterCrop(h, w),
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], additional_targets={'image2': 'image'}, p=1.0)

        transforms = {"shared_trans": shared_trans, "style_trans": style_trans, 'end_trans': end_trans, 'valid_trans': valid_trans}

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
        A.LongestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
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
        A.LongestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
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
        A.LongestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
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
                    A.LongestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                ], p=0.2),

                A.Compose([
                    A.LongestMaxSize(max_size=max(640, 640), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
                    A.PadIfNeeded(min_height=640, min_width=640, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
                    A.RandomScale(scale_limit=[-0.3, 0.0], interpolation=cv2.INTER_CUBIC, p=1.0),
                    A.RandomCrop(h, w, border_mode=cv2.BORDER_CONSTANT, p=1.0),
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

    elif preset_name == "pretrain02_plain":
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

            A.ToGray(p=0.2),
        ])

        aug_trans1 = A.Compose([
   
        ], p=1.0)



        aug_trans2 = A.Compose([
           
        ], p=1.0)

        end_trans = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

        transforms1 = A.Compose([defualt_trans, aug_trans1, end_trans])
        transforms2 = A.Compose([defualt_trans, aug_trans2, end_trans])
        transforms = {"trans1": transforms1, "trans2": transforms2}

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
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

        transforms1 = A.Compose([defualt_trans, aug_trans1, end_trans])
        transforms2 = A.Compose([defualt_trans, aug_trans2, end_trans])
        transforms = {"trans1": transforms1, "trans2": transforms2}
    

    elif preset_name == "pretrain_global_loacl":
        crop1 = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.4),
            A.Rotate(limit=(-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),

            A.OneOf([
                RandomResizedCropWithPad(
                    size=[h, w],
                    scale=[0.1, 1.0],
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

            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
                p=0.8
            ),
            A.ToGray(p=0.2),
        ])
        
        crop2 = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.4),
            A.Rotate(limit=(-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),

            A.OneOf([
                A.RandomResizedCrop(
                    size=[h, w],
                    scale=[0.03, 0.15],
                    ratio=[0.75, 1.3333333333333333],
                    interpolation=cv2.INTER_CUBIC,
                    p=0.9),

                RandomResizedCropWithPad(
                    size=[h, w],
                    scale=[0.15, 1.0],
                    ratio=[0.75, 1.3333333333333333],
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    p=0.1),
            ], p=1.0),

            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.2,
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
                p=0.2
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
            ], p=0.5),

            A.Solarize([0.5, 0.5], p=0.1),
        ], p=1.0)



        aug_trans2 = A.Compose([
            A.AdditiveNoise(
                noise_type="gaussian",
                spatial_mode="shared",
                noise_params={"mean_range":[0,0],"std_range":[0.01,0.08]},
                approximation=0.8,
                p=0.2
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
            ], p=0.5),

            A.Solarize([0.5, 0.5], p=0.1),
        ], p=1.0)

        end_trans = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

        transforms1 = A.Compose([crop1, aug_trans1, end_trans])
        transforms2 = A.Compose([crop2, aug_trans2, end_trans])
        transforms = {"trans1": transforms1, "trans2": transforms2}

    elif preset_name == "pretrain_global_loacl02":
        crop1 = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.4),
            A.Rotate(limit=(-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),

            A.OneOf([
                RandomResizedCropWithPad(
                    size=[h, w],
                    scale=[0.5, 1.0],
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

            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1,
                p=0.8
            ),
            A.ToGray(p=0.2),
        ])
        
        crop2 = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.4),
            A.Rotate(limit=(-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),

            A.OneOf([
                A.RandomResizedCrop(
                    size=[h//2, w//2],
                    scale=[0.01, 0.15],
                    ratio=[0.75, 1.3333333333333333],
                    interpolation=cv2.INTER_CUBIC,
                    p=0.9),

                RandomResizedCropWithPad(
                    size=[h//2, w//2],
                    scale=[0.15, 0.5],
                    ratio=[0.75, 1.3333333333333333],
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    p=0.1),
            ], p=1.0),

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
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

        transforms1 = A.Compose([crop1, aug_trans1, end_trans])
        transforms2 = A.Compose([crop2, aug_trans2, end_trans])
        transforms = {"trans1": transforms1, "trans2": transforms2}

    elif preset_name == "mae_pretrain":

        transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.4),
            A.Rotate(limit=(-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),

            A.OneOf([
                RandomResizedCropWithPad(
                    size=[h, w],
                    scale=[0.3, 1.0],
                    ratio=[0.75, 1.3333333333333333],
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    p=0.6),

                RandomResizedCropAutoScale(
                    size=[h, w],
                    scale=[0.03, 0.3],
                    ratio=[0.75, 1.3333333333333333],
                    interpolation=cv2.INTER_CUBIC,
                    p=0.4),
            ], p=1.0),

            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.2,
                p=0.7
            ),

            A.RGBShift(
                r_shift_limit=[-20, 20],
                g_shift_limit=[-20, 20],
                b_shift_limit=[-20, 20],
                p=0.5
            ),

            A.ToGray(p=0.2),

            A.OneOf([
                A.Compose([
                    A.AdditiveNoise(
                        noise_type="gaussian",
                        spatial_mode="shared",
                        noise_params={"mean_range":[0,0],"std_range":[0.01,0.08]},
                        approximation=0.8,
                        p=0.6
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
                    ], p=0.6),
                ], p=0.8),

                A.Compose([
                    A.AdditiveNoise(
                        noise_type="gaussian",
                        spatial_mode="shared",
                        noise_params={"mean_range":[0,0],"std_range":[0.01,0.08]},
                        approximation=0.8,
                        p=0.2
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
                    ], p=0.2),

                    A.Solarize([0.0, 0.7], p=0.9),
                ], p=0.2)
            ]),

            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ])

    return transforms



