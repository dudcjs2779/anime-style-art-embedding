import random
import numpy as np
import math
import cv2
import torch
import itertools
from typing import Any, Annotated, Literal, Callable
from pydantic import AfterValidator, BaseModel

import albumentations as A
from albumentations.augmentations.crops.transforms import BaseCrop
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.augmentations.crops import functional as fcrops
from albumentations.core.transforms_interface import BaseTransformInitSchema, BasicTransform
from albumentations.core.pydantic import (
    check_range_bounds,
    nondecreasing,
)
from albumentations.core.bbox_utils import normalize_bboxes, denormalize_bboxes

class RandomGridByBbox(BasicTransform):
    """
    각 bbox를 grid로 나누고 그 중에서 랜덤하게 몇개를 선택하여 새로운 이미지로 구성한다.

    Args:
        grid_num: 최종적으로 선택할 grid의 갯수의 범위
            ex: (2, 4) => 최소4개~최대16개의 grid
        bbox_grid_num: bbox를 나눌 grid의 갯수의 범위
        bbox_scale: bbox를 scale할 범위
        scale: grid 내부를 crop할때 crop의 scale 범위
    """

    class InitSchema(BaseTransformInitSchema):
        size: tuple[int, int]
        grid_num: tuple[int, int]
        bbox_grid_num: tuple[int, int]
        bbox_scale: tuple[float, float]
        scale: tuple[float, float]
        ratio: tuple[float, float]
        interpolation: int

    def __init__(
        self,
        size: tuple[int, int],
        grid_num: tuple[int, int] = (2, 4),
        bbox_grid_num: tuple[int, int] = (2, 4),
        bbox_scale: tuple[float, float] = (0.8, 1.2),
        scale: tuple[float, float] = (0.5, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation: int = cv2.INTER_LINEAR,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        self.size = size
        self.grid_num = grid_num
        self.bbox_grid_num = bbox_grid_num
        self.bbox_scale = bbox_scale
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

        if grid_num[0] > bbox_grid_num[0]:
            raise ValueError(
                "`bbox_grid_num[0]` must be >= `grid_num[0]`."
            )

        self._log_ratio = tuple(map(math.log, ratio))

    @property
    def targets(self):
        return {
            "image": self.apply,
            "images": self.apply_to_images,
            "mask": self.apply,
            "masks": self.apply_to_images,
        }

    def _make_grids(self, x1, y1, x2, y2, grid_num):
        width = x2 - x1
        height = y2 - y1
        grid_width = width // grid_num
        grid_height = height // grid_num

        grids = []
        for row in range(grid_num):
            grid_y1 = y1 + row * grid_height
            grid_y2 = y2 if row == grid_num - 1 else grid_y1 + grid_height

            for col in range(grid_num):
                grid_x1 = x1 + col * grid_width
                grid_x2 = x2 if col == grid_num - 1 else grid_x1 + grid_width
                grids.append((grid_x1, grid_y1, grid_x2, grid_y2))

        return grids

    def _random_crop(self, grid):
        x1, y1, x2, y2 = grid
        width, height = x2 - x1, y2 - y1

        area = width * height * self.py_random.uniform(*self.scale)
        ratio = math.exp(self.py_random.uniform(*self._log_ratio))

        crop_width = int(round(math.sqrt(area * ratio)))
        crop_height = int(round(math.sqrt(area / ratio)))

        if crop_width > width or crop_height > height:
            factor = min(width / crop_width, height / crop_height)
            crop_width = max(1, int(crop_width * factor))
            crop_height = max(1, int(crop_height * factor))

        crop_x1 = x1 + self.py_random.randint(0, width - crop_width)
        crop_y1 = y1 + self.py_random.randint(0, height - crop_height)
        crop_x2 = crop_x1 + crop_width
        crop_y2 = crop_y1 + crop_height

        return crop_x1, crop_y1, crop_x2, crop_y2

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ):
        image_height, image_width = params["shape"][:2]
        bboxes = data.get("bboxes", [])

        if len(bboxes) > 0:
            bboxes = denormalize_bboxes(bboxes, shape=(image_height, image_width))

            bbox_grids = []

            for bbox in bboxes:
                grid_num = self.py_random.randint(*self.bbox_grid_num)
                bbox_scale = self.py_random.uniform(*self.bbox_scale)

                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[:4]
                bbox_width, bbox_height = bbox_x2 - bbox_x1, bbox_y2 - bbox_y1

                dx = bbox_width * (bbox_scale - 1) * 0.5
                dy = bbox_height * (bbox_scale - 1) * 0.5

                bbox_x1 = round(max(0, bbox_x1 - dx))
                bbox_y1 = round(max(0, bbox_y1 - dy))
                bbox_x2 = round(min(image_width, bbox_x2 + dx))
                bbox_y2 = round(min(image_height, bbox_y2 + dy))

                bbox_grids.extend(self._make_grids(bbox_x1, bbox_y1, bbox_x2, bbox_y2, grid_num))

            max_n = min(self.grid_num[1], math.isqrt(len(bbox_grids)))
            min_n = min(self.grid_num[0], max_n)

            grid_num = self.py_random.randint(min_n, max_n)

            grids = self.py_random.sample(bbox_grids, grid_num * grid_num)

        else:
            grid_num = self.py_random.randint(*self.grid_num)
            grids = self._make_grids(0, 0, image_width, image_height, grid_num)

        crops = [self._random_crop(g) for g in grids]

        return {
            "crops": crops,
            "grid_num": grid_num,
            "pad_params": [0, 0, 0, 0],
        }

    def apply(
        self,
        img: np.ndarray,
        crops: tuple[tuple, ...],
        grid_num: int,
        **params: Any,
    ) -> np.ndarray:

        height, width = self.size
        out = np.empty(
            (height, width, *img.shape[2:]),
            dtype=img.dtype,
        )

        grid_height, gird_width = divmod(height, grid_num)[0], divmod(width, grid_num)[0]

        for i, (x1, y1, x2, y2) in enumerate(crops):
            row, col = divmod(i, grid_num)

            start_y = row * grid_height
            start_x = col * gird_width
            end_y = height if row == grid_num - 1 else start_y + grid_height
            end_x = width if col == grid_num - 1 else start_x + gird_width

            crop = img[y1:y2, x1:x2]

            out[start_y:end_y, start_x:end_x] = cv2.resize(crop, (end_x - start_x, end_y - start_y), interpolation=self.interpolation)

        return out


class BG_Remove(BasicTransform):
    """
    mask를 이용해 배경을 제거합니다.

    Args:
        input_size: 입력 이미지의 크기 ((H, W) 또는 (H, W, C)).
        binary_mask_p: mask를 binary 마스크로 전환해서 적용할 확률
        white_p: 배경석을 하얀색으로 적용할 확률, 나머지는 검은색으로 적용됩니다.
    """
    class InitSchema(BaseTransformInitSchema):
        input_size: tuple[int, int] | tuple[int, int, int]
        binary_mask_p: float
        white_p: float
        p: float

    def __init__(
        self,
        input_size: tuple[int, int] | tuple[int, int, int],
        binary_mask_p: float = 0.0,
        white_p: float = 0.5,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.input_size = (*input_size, 3) if len(input_size) == 2 else input_size
        self.binary_mask_p = binary_mask_p
        self.white_p = white_p

    @property
    def targets(self) -> dict[str, Callable[..., Any]]:
        return {
            "image": self.apply,
        }

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        mask = data.get("bg_mask", None)
        return {"mask": mask}

    def apply(self, img: np.ndarray, **params: Any) -> np.ndarray:
        mask = params.get("mask", None)

        if mask is None:
            return img

        use_binary_mask = self.py_random.random() < self.binary_mask_p
        use_white_bg = self.py_random.random() < self.white_p

        if use_binary_mask:
            foreground = mask >= 255
            result = np.empty_like(img)
            result[...] = 255 if use_white_bg else 0
            np.copyto(result, img, where=foreground[..., None]) # broadcast

            return result

        mask_u16 = mask[..., None].astype(np.uint16, copy=False)
        result = img.astype(np.uint16)
        result *= mask_u16
        result //= 255

        if use_white_bg:
            result += 255 - mask_u16

        return result.astype(np.uint8)


class RandomScaleCropByMaskAndBBox(BaseCrop):
    """
    랜덤 위치 및 스케일 Crop을 하면서 객체가 View 내에 잘 보존되도록 crop 한다.

    Args:
        erosion_factor: crop 위치를 랜덤하게 고를떄 얼마나 bbox를 얼마나 덜 보존할 것인가
            - crop이 작을 수록 더욱 bbox 안쪽으로 파고드는 경향이 있음
            - 면적 기준이 아닌 길이 기준 
        small_crop_factor: small crop 으로 간주할 기준
            - bbox_area * small_crop_factor 보다 작은 crop을 small crop 으로 간주하며 mask의 유효한 픽셀을 골라 주변을 crop 하는 방식으로 전환
        min_bbox_object_ratio: Crop 내에서 bbox 객체가 최소 몇 % 살아남을 것인가
            - (Crop 내에 객체의 픽셀 수 > 선택된 bbox 객체의 픽셀 수 * min_bbox_object_ratio) 인 경우 crop을 반환
        min_crop_object_ratio: Crop 내에서 객체가 차지할 비율이 몇 % 인가
            - (Crop 내에 객체의 픽셀 수 > crop의 면적 * min_crop_object_ratio) 인 경우 crop을 반환
        min_bbox_intersection_ratio: mask가 없을 시 Crop이 bbox와 최소 몇 % 겹쳐야 하는가
            - (Crop&bbox 겹치는 면적 >= bbox 면적 * min_bbox_intersection_ratio) 인 경우 crop 반환
        min_crop_intersection_ratio: mask가 없을 시 Crop에서 bbox가 최소 몇 % 차지할 것인가
            - (Crop&bbox 겹치는 면적 >= crop 면적 * min_crop_intersection_ratio) 인 경우 crop 반환
    bbox_scale: crop의 scaling을 선택된 bbox를 기준으로 함.
    """
    class InitSchema(BaseTransformInitSchema):
        size: tuple[int, int]  # (h, w)
        erosion_factor: float
        small_crop_factor: float
        min_bbox_object_ratio: float
        min_crop_object_ratio: float
        min_bbox_intersection_ratio: float
        min_crop_intersection_ratio: float
        scale: tuple
        ratio: tuple
        bbox_scale: bool
        interpolation: int
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

    def __init__(
        self,
        size: tuple[int, int],
        erosion_factor: float = 0.0,
        small_crop_factor: float = 0.25,
        min_bbox_object_ratio: float = 0.3,
        min_crop_object_ratio: float = 0.4,
        min_bbox_intersection_ratio: float = 0.3,
        min_crop_intersection_ratio: float = 0.4,
        scale: tuple[float, float] = (0.5, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        bbox_scale: bool = False,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0.0,
        fill_mask: tuple[float, ...] | float = 0.0,
        p: float = 1.0,
    ):
        super().__init__(p=p)

        self.size = size
        self.erosion_factor = erosion_factor
        self.small_crop_factor = small_crop_factor
        self.min_bbox_object_ratio = min_bbox_object_ratio
        self.min_crop_object_ratio = min_crop_object_ratio
        self.min_bbox_intersection_ratio = min_bbox_intersection_ratio
        self.min_crop_intersection_ratio = min_crop_intersection_ratio
        self.scale = scale
        self.ratio = ratio
        self.bbox_scale = bbox_scale
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.fill = fill
        self.fill_mask = fill_mask

    def get_params_dependent_on_data(self, params: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        image_height, image_width = params["shape"][:2]
        area = image_height * image_width

        bboxes = data.get("bboxes", [])
        mask = data.get("mask", None)

        # Crop size
        if len(bboxes) > 0:
            bboxes = denormalize_bboxes(data["bboxes"], shape=(image_height, image_width))
            reference_bbox = self.py_random.choice(bboxes)
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = reference_bbox[:4].round().astype(np.int64)

            bbox_width = bbox_x2 - bbox_x1
            bbox_height = bbox_y2 - bbox_y1
            bbox_area = bbox_width * bbox_height
        else:
            reference_bbox = None

        if self.bbox_scale and reference_bbox is not None:
            scale_area = bbox_area
        else:
            scale_area = area

        scale = self.py_random.uniform(*self.scale)
        target_area = scale * scale_area

        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        aspect_ratio = math.exp(self.py_random.uniform(*log_ratio))

        width = int(round(math.sqrt(target_area * aspect_ratio)))
        height = int(round(math.sqrt(target_area / aspect_ratio)))

        scale_factor = min(image_width / width, image_height / height)

        if scale_factor < 1:
            width = int(width * scale_factor)
            height = int(height * scale_factor)

        crop_area = width * height

        # No bbox
        if reference_bbox is None:
            crop_x1 = self.py_random.randint(0, image_width - width)
            crop_y1 = self.py_random.randint(0, image_height - height)
            crop_x2 = crop_x1 + width
            crop_y2 = crop_y1 + height

            return {"crop_coords": (crop_x1, crop_y1, crop_x2, crop_y2), "pad_params": [0, 0, 0, 0]}

        # Mask information used below
        if mask is not None:
            bbox_mask = mask[bbox_y1:bbox_y2, bbox_x1:bbox_x2]
            bbox_mask_count = np.count_nonzero(bbox_mask)

        # Small crop by mask
        if mask is not None and crop_area < bbox_area * self.small_crop_factor:
            for i in range(100):
                y = np.random.randint(bbox_height)
                x = np.random.randint(bbox_width)

                if bbox_mask[y, x]:
                    y += bbox_y1
                    x += bbox_x1

                    crop_x1 = np.clip(x - width // 2, 0, image_width - width)
                    crop_y1 = np.clip(y - height // 2, 0, image_height - height)
                    crop_x2 = crop_x1 + width
                    crop_y2 = crop_y1 + height

                    # print(f"crop_area: {crop_area}")
                    # print(f"bbox_area * small_crop_factor: {bbox_area * self.small_crop_factor}")
                    # print(f"yx: ({i}){(y, x)}")
                    # print(f"crop_coords: {(crop_x1, crop_y1, crop_x2, crop_y2)}")

                    return {"crop_coords": (crop_x1, crop_y1, crop_x2, crop_y2), "pad_params": [0, 0, 0, 0]}

        # Crop range around bbox
        if self.erosion_factor < 1.0:
            eroded_width = bbox_width * (1.0 - self.erosion_factor)
            eroded_height = bbox_height * (1.0 - self.erosion_factor)

            min_crop_x = np.clip(bbox_x1 + eroded_width - width, 0, image_width - width)
            max_crop_x = np.clip(bbox_x2 - eroded_width, 0, image_width - width)
            min_crop_y = np.clip(bbox_y1 + eroded_height - height, 0, image_height - height)
            max_crop_y = np.clip(bbox_y2 - eroded_height, 0, image_height - height)
        else:
            min_crop_x = np.clip(bbox_x1 - width, 0, image_width - width)
            max_crop_x = np.clip(bbox_x2, 0, image_width - width)
            min_crop_y = np.clip(bbox_y1 - height, 0, image_height - height)
            max_crop_y = np.clip(bbox_y2, 0, image_height - height)

        # Pre-compute thresholds.
        if mask is not None:
            min_bbox_mask_count = bbox_mask_count * self.min_bbox_object_ratio
            min_crop_mask_count = crop_area * self.min_crop_object_ratio

        for i in range(10):
            crop_x1 = int(self.py_random.uniform(min_crop_x, max_crop_x))
            crop_y1 = int(self.py_random.uniform(min_crop_y, max_crop_y))
            crop_x2 = crop_x1 + width
            crop_y2 = crop_y1 + height

            if mask is not None:
                # Calculate crop mask area only once.
                crop_mask_count = np.count_nonzero(mask[crop_y1:crop_y2, crop_x1:crop_x2])

                # bbox_object_ratio >= min_bbox_object_ratio
                # OR
                # crop_object_ratio >= min_crop_object_ratio
                if crop_mask_count >= min_bbox_mask_count or crop_mask_count >= min_crop_mask_count:
                    # print(f"bbox_object_ratio: {crop_mask_count / bbox_mask_count}")
                    # print(f"crop_object_ratio: {crop_mask_count / crop_area}")
                    return {"crop_coords": (crop_x1, crop_y1, crop_x2, crop_y2), "pad_params": [0, 0, 0, 0]}

            else:
                intersection_width = max(0, min(bbox_x2, crop_x2) - max(bbox_x1, crop_x1))
                intersection_height = max(0, min(bbox_y2, crop_y2) - max(bbox_y1, crop_y1))
                intersection_area = intersection_width * intersection_height

                if intersection_area >= bbox_area * self.min_bbox_intersection_ratio or intersection_area >= crop_area * self.min_crop_intersection_ratio:
                    # print(f"bbox_area * {self.min_bbox_intersection_ratio}: {bbox_area * self.min_bbox_intersection_ratio}")
                    # print(f"crop_area * {self.min_crop_intersection_ratio}: {crop_area * self.min_crop_intersection_ratio}")
                    # print(f"intersection_area: {intersection_area}")
                    return {"crop_coords": (crop_x1, crop_y1, crop_x2, crop_y2), "pad_params": [0, 0, 0, 0]}

        # Fallback
        crop_x1 = self.py_random.randint(0, image_width - width)
        crop_y1 = self.py_random.randint(0, image_height - height)
        crop_x2 = crop_x1 + width
        crop_y2 = crop_y1 + height

        # print(f"fallback: {(crop_x1, crop_y1, crop_x2, crop_y2)}")
        return {"crop_coords": (crop_x1, crop_y1, crop_x2, crop_y2), "pad_params": [0, 0, 0, 0]}

    def apply(self, img: np.ndarray, crop_coords: tuple[int, int, int, int], **params: Any) -> np.ndarray:
        pad_params = params.get("pad_params")
        padded = fgeometric.pad_with_params(img, *pad_params, border_mode=self.border_mode, value=self.fill)
        crop = fcrops.crop(padded, *crop_coords)
        return fgeometric.resize(crop, self.size, self.interpolation)

    def apply_to_bboxes(self, bboxes: np.ndarray, crop_coords: tuple[int, int, int, int], **params: Any) -> np.ndarray:
        pad_params = params.get("pad_params")
        image_height, image_width = params["shape"][:2]
        padded_height = image_height + pad_params[0] + pad_params[1]
        padded_width = image_width + pad_params[2] + pad_params[3]
        return fcrops.crop_bboxes_by_coords(bboxes, crop_coords, (padded_height, padded_width))


class ShiftScaleCenterCrop(BaseCrop):
    class InitSchema(BaseTransformInitSchema):
        height: int
        width: int
        shift_limit: tuple
        scale_limit: tuple
        ratio: tuple
        interpolation: int
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ]
        fill: tuple[float, ...] | float
        fill_mask: tuple[float, ...] | float

    def __init__(
        self,
        height: int,
        width: int,
        shift_limit: tuple[float, float] = (-0.2, 0.2),
        scale_limit: tuple[float, float] = (0.5, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: Literal[
            cv2.BORDER_CONSTANT,
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_WRAP,
            cv2.BORDER_REFLECT_101,
        ] = cv2.BORDER_CONSTANT,
        fill: tuple[float, ...] | float = 0.0,
        fill_mask: tuple[float, ...] | float = 0.0,
        p: float = 1.0,
    ):
        super().__init__(p=p)
        
        self.height = height
        self.width = width
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.ratio = ratio
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.fill = fill
        self.fill_mask = fill_mask

    def get_params_dependent_on_data(
        self,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> dict[str, Any]:  # Changed return type to be more flexible
        image_shape = params["shape"][:2]
        image_height, image_width = image_shape

        anchor_length = min(image_height, image_width)  # 가장큰 center crop size
        
        image_aspect = max(image_width / image_height, image_height / image_width)

        # 종횡비 차이가 클수록 padding 증가
        pad_factor = max(0.0, image_aspect - 1.2)

        # 너무 과도해지지 않도록 제한
        pad_factor = min(pad_factor * 0.25, 1.0)

        base_anchor = min(image_height, image_width)

        adaptive_pad = round(base_anchor * pad_factor)

        virtual_height = image_height + adaptive_pad * 2
        virtual_width = image_width + adaptive_pad * 2

        anchor_length = min(virtual_height, virtual_width)

        # shift 어그멘테이션
        shift_x, shift_y = self.py_random.uniform(*self.shift_limit), self.py_random.uniform(*self.shift_limit)
        crop_height, crop_width = self.height, self.width
        center_x =  image_width // 2
        center_y =  image_height // 2

        center_x = center_x + round(anchor_length * shift_x)
        center_y = center_y + round(anchor_length * shift_y)

        scale = math.sqrt(self.py_random.uniform(*self.scale_limit))
        # scale = 1.0
        crop_size = round(anchor_length * scale)

        # 종횡비 어그멘테이션
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        aspect_ratio = math.exp(self.py_random.uniform(*log_ratio))
        # aspect_ratio = 0.8

        crop_width = round(crop_size * aspect_ratio)
        crop_height = round(crop_size / aspect_ratio)

        y_min = center_y - crop_height // 2
        y_max = y_min + crop_height
        x_min = center_x - crop_width // 2
        x_max = x_min + crop_width

        # crop 범위가 이미지를 넘어가는 경우 패딩, y좌표는 클수록 하단에 이미지의 하단에 가깝다.
        pad_top = max(0, -y_min)
        pad_left = max(0, -x_min)
        
        pad_bottom = max(0, y_max - image_height)
        pad_right = max(0, x_max - image_width)

        y_min = max(0, y_min)
        x_min = max(0, x_min)

        y_max = min(image_height, y_max)
        x_max = min(image_width, x_max)

        crop_coords = {"y_min": y_min, "y_max": y_max, "x_min": x_min, "x_max": x_max}
        pad_params = {"h_pad_top": pad_top, "h_pad_bottom": pad_bottom, "w_pad_left": pad_left, "w_pad_right": pad_right}

        return {"crop_coords": crop_coords, "pad_params": pad_params}

    def apply(
        self,
        img: np.ndarray,
        crop_coords: dict[str, int],
        **params: Any,
    ) -> np.ndarray:
        pad_params = params.get("pad_params")

        cropped = fcrops.crop(img, **crop_coords)
        if pad_params is not None:
            padded = fgeometric.pad_with_params(cropped, **pad_params, border_mode=self.border_mode, value=self.fill)
        resized = fgeometric.resize(padded, [self.height, self.width], self.interpolation)

        return resized
    

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
        interpolation: int
        mask_interpolation: int
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        border_mode: int

    def __init__(
        self,
        size: tuple[int, int],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        fill: tuple[float, ...] | float = 0,
        fill_mask: tuple[float, ...] | float = 0,
        border_mode = cv2.BORDER_CONSTANT,
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

        # print("skiped")
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
        interpolation: int
        mask_interpolation: int

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
        interpolation: int
        fill: tuple[float, ...] | float
        border_mode: int

    def __init__(
            self, 
            output_size, 
            crop_num, 
            grid_scale, 
            interpolation: int = cv2.INTER_LINEAR, 
            fill: tuple[float, ...] | float = 0,
            border_mode = cv2.BORDER_CONSTANT,
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
    

