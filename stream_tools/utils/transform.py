from typing import Tuple, List

import torch
from torch import Tensor
import torch.nn.functional as F
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode


@torch.compile(dynamic=True, backend='eager')
def letterbox(
    img: Tensor,
    new_shape: Tuple[int, int] = (640, 640),
    auto: bool = False,
    scale_fill: bool = False,
    scaleup: bool = False,
    stride: int = 32,
) -> Tuple[Tensor, Tuple[float, float], Tuple[float,float], List[int]]:
    '''Letterbox from https://github.com/ultralytics/ultralytics/blob/537c50e45f94b214338c6e53bc9822b15fe3a595/ultralytics/data/augment.py#L740
    Inputs:
    img: float BxCxHxW
    '''

    shape = img.shape[-2:]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r)),
    dw, dh = float(new_shape[1] - new_unpad[0]), float(new_shape[0] - new_unpad[1])  # wh padding
    if auto:  # minimum rectangle
        # dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        dw, dh = torch.remainder(dw, stride), torch.remainder(dh, stride)
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        # img = F.interpolate(img, size=new_unpad[::-1])
        img = resize(img, new_unpad[::-1], interpolation=InterpolationMode.BILINEAR, antialias=True)
        # img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    pad = (left,right,top,bottom)
    img = F.pad(img, pad, value=114.0)
    return img, ratio, (dw, dh), shape
