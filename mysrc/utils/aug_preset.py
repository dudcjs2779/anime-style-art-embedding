import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from mysrc.utils.augmentation import (
    ShiftScaleCenterCrop,
    RandomResizedCropAutoScale, 
    RandomResizedCropWithPad, 
    RandomSmallestMaxSize, 
    RandomScaleCropByMaskAndBBox,
    RandomGridByBbox,
    BG_Remove,
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

    elif preset_name == "long_max_pad_valid":
        transforms = A.Compose([
        A.LongestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_CUBIC), # 가장 긴변을 img_size에 맞춤
        A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, fill=255), # 패딩
        A.Normalize(mean=mean, std=std, max_pixel_value=255),
        ToTensorV2(),
    ], p=1.0)

    elif preset_name == "pretrain_global_loacl":
        crop1 = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.4),
            A.Rotate(limit=(-15, 15), p=0.5, border_mode=cv2.BORDER_CONSTANT, fill=[255, 255, 255]),
            # A.CenterCrop(h, w),

            A.OneOf([
                ShiftScaleCenterCrop(
                    height=h,
                    width=w,
                    shift_limit=(-0.2, 0.2),
                    scale_limit=(0.33, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    p=0.3,
                ),

                RandomResizedCropWithPad(
                    size=[h, w],
                    scale=[0.3, 1.0],
                    ratio=[0.75, 1.3333333333333333],
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    p=0.4),

                A.Compose([
                    AutoPad(border_mode=cv2.BORDER_CONSTANT, fill=255),
                    A.RandomResizedCrop(
                        size=[h, w],
                        scale=[0.33, 1.0],
                        ratio=[0.75, 1.3333333333333333],
                        interpolation=cv2.INTER_CUBIC,
                        p=1.0),
                ], p=0.3),
            ], p=1.0),

            A.OneOf([
                A.ColorJitter(
                    brightness=0.2, # 0.2
                    contrast=0.2,   # 0.2
                    saturation=0.2, # 0.2
                    hue=0.0,
                    p=0.9
                ),

                A.ColorJitter(
                    brightness=0.2, # 0.2
                    contrast=0.2,   # 0.2
                    saturation=0.2, # 0.2
                    hue=0.5,
                    p=0.1
                ),
            ], p=0.5),
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
                    scale=[0.1, 0.3],
                    ratio=[0.75, 1.3333333333333333],
                    interpolation=cv2.INTER_CUBIC,
                    p=0.5),

                ShiftScaleCenterCrop(
                    height=h,
                    width=w,
                    shift_limit=(-0.2, 0.2),
                    scale_limit=(0.1, 0.33),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    p=0.5,
                ),

                # RandomResizedCropWithPad(
                #     size=[h, w],
                #     scale=[0.15, 1.0],
                #     ratio=[0.75, 1.3333333333333333],
                #     interpolation=cv2.INTER_CUBIC,
                #     border_mode=cv2.BORDER_CONSTANT,
                #     fill=255,
                #     p=0.1),
            ], p=1.0),

            A.OneOf([
                A.ColorJitter(
                    brightness=0.2, # 0.2
                    contrast=0.2,   # 0.2
                    saturation=0.2, # 0.2
                    hue=0.0,
                    p=0.9
                ),

                A.ColorJitter(
                    brightness=0.2, # 0.2
                    contrast=0.2,   # 0.2
                    saturation=0.2, # 0.2
                    hue=0.5,
                    p=0.1
                ),
            ], p=0.5),
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
                    blur_limit=[3, 15],
                    sigma_x_limit=[0.3, 1.6],
                    sigma_y_limit=[0.3, 1.6],
                    rotate_limit=[-90, 90],
                    beta_limit=[0.5, 8.0],
                    noise_limit=[0.9, 1.1],
                    p=0.7
                ),
                A.GaussianBlur(
                    blur_limit=0,
                    sigma_limit=[0.5, 2.0],
                    p=0.3
                ),
            ], p=0.4),

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
                    blur_limit=[3, 15],
                    sigma_x_limit=[0.3, 1.6],
                    sigma_y_limit=[0.3, 1.6],
                    rotate_limit=[-90, 90],
                    beta_limit=[0.5, 8.0],
                    noise_limit=[0.9, 1.1],
                    p=0.7
                ),
                A.GaussianBlur(
                    blur_limit=0,
                    sigma_limit=[0.5, 2.0],
                    p=0.3
                ),
            ], p=0.4),

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

    elif preset_name == "bbox_crop_pretrain":
        crop = A.Compose([
            A.OneOf([
                A.OneOf([
                    RandomScaleCropByMaskAndBBox(
                        size=(h, w),
                        erosion_factor=0.6,
                        small_crop_factor=0.25,
                        min_bbox_object_ratio=0.3,
                        min_crop_object_ratio=0.4,
                        min_bbox_intersection_ratio=0.4,
                        min_crop_intersection_ratio=0.5,
                        scale=(0.5, 1.0),
                        bbox_scale=False,
                        interpolation=cv2.INTER_CUBIC,
                        p=0.2,
                    ),
                    RandomScaleCropByMaskAndBBox(
                        size=(h, w),
                        erosion_factor=0.6,
                        small_crop_factor=0.25,
                        min_bbox_object_ratio=0.3,
                        min_crop_object_ratio=0.4,
                        min_bbox_intersection_ratio=0.4,
                        min_crop_intersection_ratio=0.5,
                        scale=(0.3, 1.0),
                        bbox_scale=True,
                        interpolation=cv2.INTER_CUBIC,
                        p=0.4,
                    ),
                    RandomScaleCropByMaskAndBBox(
                        size=(h, w),
                        erosion_factor=0.6,
                        small_crop_factor=0.25,
                        min_bbox_object_ratio=0.3,
                        min_crop_object_ratio=0.4,
                        min_bbox_intersection_ratio=0.4,
                        min_crop_intersection_ratio=0.5,
                        scale=(0.05, 0.3),
                        bbox_scale=True,
                        interpolation=cv2.INTER_CUBIC,
                        p=0.4,
                    ),
                ], p=0.7),

                RandomGridByBbox(
                    size=(h, w),
                    grid_num=(2, 5),
                    bbox_grid_num=(2, 4),
                    bbox_scale=(0.8, 1.2),
                    scale=(0.5, 1.0),
                    interpolation=cv2.INTER_CUBIC,
                    p=0.3,
                ),
            ], p=1.0)
        ], p=1.0)


        effect_trans = A.Compose([
            A.OneOf([
                A.ColorJitter(
                    brightness=0.2, # 0.2
                    contrast=0.2,   # 0.2
                    saturation=0.2, # 0.2
                    hue=0.0,
                    p=0.9
                ),

                A.ColorJitter(
                    brightness=0.2, # 0.2
                    contrast=0.2,   # 0.2
                    saturation=0.2, # 0.2
                    hue=0.5,
                    p=0.1
                ),
            ], p=0.5),
            
            A.ToGray(p=0.2),

            A.AdditiveNoise(
                noise_type="gaussian",
                spatial_mode="shared",
                noise_params={"mean_range":[0,0],"std_range":[0.01,0.08]},
                approximation=0.8,
                p=0.2
            ),

            A.OneOf([
                A.AdvancedBlur(
                    blur_limit=[3, 15],
                    sigma_x_limit=[0.3, 1.6],
                    sigma_y_limit=[0.3, 1.6],
                    rotate_limit=[-90, 90],
                    beta_limit=[0.5, 8.0],
                    noise_limit=[0.9, 1.1],
                    p=0.7
                ),
                A.GaussianBlur(
                    blur_limit=0,
                    sigma_limit=[0.5, 2.0],
                    p=0.3
                ),
            ], p=0.4),

            BG_Remove(
                input_size=(448, 448),
                binary_mask_p=0.5,
                white_p=0.5,
                p=0.3,
            ),

            A.Solarize([0.5, 0.5], p=0.1),
        ], p=1.0)

        end_trans = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ], p=1.0)

        transforms = A.Compose(
            [crop, effect_trans, end_trans], 
            bbox_params=A.BboxParams(format='pascal_voc'),
            additional_targets={
                "bg_mask": "mask",
            },
        )

    elif preset_name == "tag_preset":
        main_trans = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=[-0.0625, 0.0625],
                scale_limit=[0.0, 0.0],
                rotate_limit=[-0, 0],
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                rotate_method="ellipse",
                fill=255,
                p=1.0,
            ),
            A.LongestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_CUBIC, p=1.0),
            A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(h*0.05, w*0.1),
                    hole_width_range=(h*0.05, w*0.1),
                    fill="0",
                    p=0.5
            ),
            A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, fill=255, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ToGray(p=0.2),
        ], p=1.0)

        style_trans = A.Compose([
            A.AdditiveNoise(
                    noise_type="gaussian",
                    spatial_mode="shared",
                    noise_params={"mean_range":[0,0],"std_range":[0.01,0.08]},
                    approximation=0.8,
                    p=0.2
            ),
            A.ColorJitter(
                brightness=(0.95, 1.05),
                contrast=(0.95, 1.05),
                saturation=(1.0, 1.0),
                hue=0.0,
                p=1.0
            ),
        ], p=1.0)

        valid_trans = A.Compose([ # Center Crop
            A.LongestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_CUBIC, p=1.0),
            A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, fill=255, p=1.0),
        ], p=1.0)

        end_trans = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ])

        transforms = A.Compose([
            main_trans, style_trans, end_trans
        ], save_applied_params=True)

    elif preset_name == "tag_preset2":
        main_trans = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=[-0.0625, 0.0625],
                scale_limit=[0.0, 0.0],
                rotate_limit=[-0, 0],
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                rotate_method="ellipse",
                fill=255,
                p=1.0,
            ),
            A.LongestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_CUBIC, p=1.0),
            A.CoarseDropout(
                    num_holes_range=(1, 4),
                    hole_height_range=(h*0.05, w*0.1),
                    hole_width_range=(h*0.05, w*0.1),
                    fill="0",
                    p=0.5
            ),
            A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, fill=255, p=1.0),
            A.HorizontalFlip(p=0.5),
        ], p=1.0)

        style_trans = A.Compose([
            A.AdditiveNoise(
                    noise_type="gaussian",
                    spatial_mode="shared",
                    noise_params={"mean_range":[0,0],"std_range":[0.01,0.08]},
                    approximation=0.8,
                    p=0.2
            ),
            A.ColorJitter(
                brightness=(0.95, 1.05),
                contrast=(0.95, 1.05),
                saturation=(1.0, 1.0),
                hue=0.0,
                p=1.0
            ),
        ], p=1.0)

        valid_trans = A.Compose([ # Center Crop
            A.LongestMaxSize(max_size=max(h, w), interpolation=cv2.INTER_CUBIC, p=1.0),
            A.PadIfNeeded(min_height=h, min_width=w, border_mode=cv2.BORDER_CONSTANT, fill=255, p=1.0),
        ], p=1.0)

        end_trans = A.Compose([
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            ToTensorV2(),
        ])

        transforms = A.Compose([
            main_trans, style_trans, end_trans
        ], save_applied_params=True)


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



