from os import PathLike
import colorsys
from typing import BinaryIO, Optional, Sequence, Union, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def generate_distinct_colors(n: int) -> list[tuple[int, int, int]]:
    """n개의 서로 구분되는 RGB 색상 리스트를 생성합니다."""
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


def draw_bboxes(
    image: Image.Image,
    bboxes: Union[Sequence[float], Sequence[Sequence[float]]],
    labels: Optional[Union[str, Sequence[str]]] = None,
    confidences: Optional[Union[float, Sequence[float]]] = None,
    show_label: bool = True,
    show_confidence: bool = True,
    box_color: Union[str, tuple] = "red",
    box_colors: Optional[Sequence[Union[str, tuple]]] = None,
    random_colors: bool = False,
    text_color: Union[str, tuple] = "white",
    line_width: int = 3,
    font_size: int = 20,
) -> Image.Image:
    """PIL 이미지 위에 바운딩 박스(Bounding Box)와 라벨/신뢰도를 그립니다."""
    output_image = image.copy()
    draw = ImageDraw.Draw(output_image)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # 입력 데이터 정규화 (단일 입력 -> 리스트 형태)
    if len(bboxes) == 4 and isinstance(bboxes[0], (int, float)):
        bboxes = [bboxes]
    bboxes = list(bboxes)
    num_boxes = len(bboxes)

    if labels is None:
        labels = [None] * num_boxes
    elif isinstance(labels, str):
        labels = [labels]
    else:
        labels = list(labels)

    if confidences is None:
        confidences = [None] * num_boxes
    elif isinstance(confidences, (int, float)):
        confidences = [confidences]
    else:
        confidences = list(confidences)

    # 입력 데이터 개수 검증
    if len(labels) != num_boxes:
        raise ValueError(f"bboxes 개수({num_boxes})와 labels 개수({len(labels)})가 일치하지 않습니다.")
    if len(confidences) != num_boxes:
        raise ValueError(f"bboxes 개수({num_boxes})와 confidences 개수({len(confidences)})가 일치하지 않습니다.")

    # 박스 색상 결정 (우선순위: box_colors > random_colors > box_color)
    if box_colors is not None:
        colors = list(box_colors)
        if len(colors) != num_boxes:
            raise ValueError(f"bboxes 개수({num_boxes})와 box_colors 개수({len(colors)})가 일치하지 않습니다.")
    elif random_colors:
        colors = generate_distinct_colors(num_boxes)
    else:
        colors = [box_color] * num_boxes

    # 바운딩 박스 및 텍스트 시각화
    for bbox, label, confidence, color in zip(bboxes, labels, confidences, colors):
        x1, y1, x2, y2 = bbox

        # 박스 영역 그리기
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        # 표시할 텍스트 구성
        text_parts = []
        if show_label and label is not None:
            text_parts.append(str(label))
        if show_confidence and confidence is not None:
            text_parts.append(f"{confidence:.2f}")

        if not text_parts:
            continue

        text = " ".join(text_parts)

        # 텍스트 배경 및 위치 계산
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = x1
        text_y = max(0, y1 - text_height - 4)

        # 텍스트 배경 박스 및 텍스트 랜더링
        draw.rectangle(
            [text_x, text_y, text_x + text_width + 6, text_y + text_height + 4],
            fill=color,
        )
        draw.text((text_x + 3, text_y + 2), text, fill=text_color, font=font)

    return output_image

def alpha_mix(image, mask, white_bg=False):
    mask = mask.convert("L")
    if white_bg:
        image = image.convert('RGB')
        background = Image.new("RGB", image.size, (255, 255, 255))

        image = Image.composite(image, background, mask)

        return image
    else:
        image = image.convert('RGBA')
        image.putalpha(mask)

        return image
