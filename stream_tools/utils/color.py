import torch
from torch import Tensor


@torch.compile(dynamic=True, backend="eager")
def yuv_to_rgb(frames: Tensor) -> Tensor:
    """Converts YUV BCHW dims torch tensor to RGB BCHW dims torch tensor

    :param frames: YUV BCHW dims torch tensor
    :return: RGB BCHW dims torch tensor
    """
    frames = frames.to(torch.float32).div_(255)
    y = frames[..., 0, :, :]
    u = frames[..., 1, :, :] - 0.5
    v = frames[..., 2, :, :] - 0.5

    r = y + 1.14 * v
    g = y + -0.396 * u - 0.581 * v
    b = y + 2.029 * u

    rgb = torch.stack([r, g, b], 1).clamp_(0, 1)
    return rgb
