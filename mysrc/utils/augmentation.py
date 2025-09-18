import random
import numpy as np
import math
import cv2
import torch
import itertools
from typing import Any, Annotated
from pydantic import AfterValidator, BaseModel

import albumentations as A
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
    
class RandomResizedCropAutoScale(A.RandomResizedCrop):
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

    def __init__(
        self,
        size: tuple[int, int],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
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

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, tuple[int, int, int, int]]:
        image_shape = params["shape"][:2]
        image_height, image_width = image_shape

        max_scale = min(image_height, image_width) / max(image_height, image_width)
        self.scale = (self.scale[0], math.ceil(max_scale*100) / 100)
        
        area = image_height * image_width

        for _ in range(10):
            target_area = self.py_random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(self.py_random.uniform(*log_ratio))

            width = int(round(math.sqrt(target_area * aspect_ratio)))
            height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < width <= image_width and 0 < height <= image_height:
                i = self.py_random.randint(0, image_height - height)
                j = self.py_random.randint(0, image_width - width)

                h_start = i * 1.0 / (image_height - height + 1e-10)
                w_start = j * 1.0 / (image_width - width + 1e-10)

                crop_shape = (height, width)

                crop_coords = fcrops.get_crop_coords(image_shape, crop_shape, h_start, w_start)

                return {"crop_coords": crop_coords}

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

        return {"crop_coords": crop_coords}

class GridNCrops(A.ImageOnlyTransform):
    '''
    grid_scale에 따라서 이미지를 다양한 view로 쪼갠뒤 crop_num 만큼 view를 선택해 crop 합니다.
    crop한 이미지는 output_size로 resize한뒤 출력합니다.

    Args:
        output_size: 출력 이미지 크기
        crop_num: 크롭할 횟수
        grid_scale: 그리드의 크기 
            ex: grid_scale=9인 경우 1개의 그리드 뷰의 크기는 원본 이미지의 약 1/9에 해당합니다.

    '''
    class InitSchema(BaseModel):  # 스키마가 없으니까 padding이 빨간색으로 표시되는 버그가 발생함.
        output_size: tuple[int, int]
        crop_num: int
        grid_scale: int
        interpolation: InterpolationType
        fill: ColorType
        border_mode: BorderModeType

    def __init__(
            self, 
            output_size, 
            crop_num, 
            grid_scale, 
            interpolation: int = cv2.INTER_LINEAR, 
            fill: ColorType = 0,
            border_mode: BorderModeType = cv2.BORDER_CONSTANT,
            p=1.0,
        ):
        super().__init__(p=p)
        self.output_size = output_size
        self.crop_num = crop_num
        self.grid_scale = grid_scale
        self.interpolation = interpolation
        self.fill = fill
        self.border_mode = border_mode

    def get_params_dependent_on_data(self, params, data):
        params = {}
        img = data['image']

        im_h, im_w, _ = img.shape
        crop_size = math.floor(math.sqrt((im_h * im_w) / self.grid_scale))  # 크롭 크기

        ## 크롭 크기가 이미지의 w나 h보다 큰 경우 패딩을 채움
        if im_w < crop_size:
            padding = math.ceil((crop_size - im_w) / 2)
            params.update({'pad_top': 0, 'pad_bottom': 0, 'pad_left': padding, 'pad_right': padding})
            im_w += padding * 2
        elif im_h < crop_size:
            # print("im_h < crop_size")
            padding = math.ceil((crop_size - im_h) / 2)
            params.update({'pad_top': padding, 'pad_bottom': padding, 'pad_left': 0, 'pad_right': 0})
            im_h += padding * 2

        # 그리드의 행열 크기 계산
        row_num = min(round(im_h  / crop_size), self.grid_scale)
        col_num = max(round(im_w  / crop_size), 1)
        # print(row_num, col_num)

        # print("im_h, im_w:", im_h, im_w)
        # print("crop_size:", crop_size)

        possible_coords = list(itertools.product(range(row_num), range(col_num))) # [[i, j], [i, j+1], ... ]
        # print(possible_coords)

        # crop_num의 갯수를 일정하게 보장해야함
        crop_num = self.crop_num
        k = min(self.crop_num, len(possible_coords))
        sel_coords = []
        while crop_num > 0:
            sel_coords.extend(random.sample(possible_coords, k=k))
            crop_num -= k

        # sel_coords = random.sample(possible_coords, k=len(possible_coords))
        # sel_coords = sorted(sel_coords)

        # diff 음수: crop을 이어붙인 크기가 이미지 크기를 조과함 => 이미지를 조금씩 겹침.
        # diff 양수: crop을 이어붙이고도 이미지의 픽셀수가 남음 => 이미지를 조금씩 띄움.
        diff_h = im_h - (crop_size * row_num)
        diff_w = im_w - (crop_size * col_num)
        # print("diff_h, diff_w: ", diff_h, diff_w)

        h_step = (diff_h / max(row_num -1, 1))
        w_step = (diff_w / max(col_num -1, 1))
        # print("h_step, w_step: ", h_step, w_step)

        crop_coords = []
        for row, col in sel_coords:
            margin_col = round(col * w_step)
            left = col * crop_size + margin_col
            right = left + crop_size

            margin_row = round(row * h_step)
            top = row * crop_size + margin_row
            bottom = top + crop_size

            # x_min, y_min, x_max, y_max
            crop_coords.append([int(left), int(top), int(right), int(bottom)])

        params['crop_coords'] = crop_coords
        return params
    
    def apply(
        self,
        img: np.ndarray,
        crop_coords: list,
        pad_top: int = None,
        pad_bottom: int = None,
        pad_left: int= None,
        pad_right: int= None,
        **params: Any,
    ) -> np.ndarray:
        if pad_top is not None:
            print(img.shape)
            img = fgeometric.pad_with_params(img, pad_top, pad_bottom, pad_left, pad_right, border_mode=self.border_mode, value=self.fill)
            print(img.shape)

        crops = [
            fgeometric.resize(
                fcrops.crop(img, *coord),
                self.output_size,
                self.interpolation
            )
            for coord in crop_coords
        ]
        return np.stack(crops)  # (crop_num, h, w, c)
    

class BatchToTensor(A.ImageOnlyTransform):
    def __init__(self):
        super().__init__(p=1.0)

    def apply(self, img, **params):
        return torch.from_numpy(img.transpose(0, 3, 1, 2))
    

class BatchNormalize(A.ImageOnlyTransform):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255):
        super().__init__(p=1.0)
        self.mean = np.array(mean).reshape(1, 1, 1, -1).astype(np.float32)
        self.std = np.array(std).reshape(1, 1, 1, -1).astype(np.float32)
    
        self.max_pixel_value = max_pixel_value

    def apply(self, img, **params):
        img = img.astype(np.float32, copy=False)
        img /= self.max_pixel_value
        img -= self.mean
        img /= self.std
        return img
    

