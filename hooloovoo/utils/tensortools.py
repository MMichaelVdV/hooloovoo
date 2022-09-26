from typing import Iterable, Union, Optional

import torch.nn.functional as f
import torch

from hooloovoo.utils.arbitrary import LRTB


def interpolate_tensor(t: torch.tensor, size):
    """
    Re-sizes a tensor 't' image to have the same width and height of 'size'.
    Does not waste cpu cycles if width and height already match that of 'size'.
    Note that pytorch tensor dimension convention is 'BCHW'.
    """
    out_size = size[-2:]
    t_size = t.shape[-2:]
    t_ndim = len(t.shape)
    if out_size != t_size:
        if t_ndim == 3:
            t = t.unsqueeze(1)

        if t.dtype == torch.int64:
            t_as_float = t.float()
            t_as_float_scaled = f.interpolate(t_as_float, size=out_size, mode="nearest")
            t_scaled = t_as_float_scaled.long()
        elif t.dtype == torch.float32:
            t_as_float = t
            t_as_float_scaled = f.interpolate(t_as_float, size=out_size, mode="bilinear", align_corners=False)
            t_scaled = t_as_float_scaled
        else:
            raise ValueError("Cannot resize dtype: %s" % str(t.dtype))

        if t_ndim == 3:
            t_scaled = t_scaled.squeeze(1)
    else:
        t_scaled = t
    return t_scaled


def crop_tensor(x: torch.Tensor, crop_box: LRTB):
    """Crops a tensor in the last two channels, given a crop box of form ``(left, right, top, bottom)``"""
    left, right, top, bottom = crop_box
    return x[..., top:bottom, left:right]


def unpad_tensor(x: torch.Tensor, padding: LRTB):
    """
    Removes padding from a tensor in the last two channels, padding is given as ``(left, right, top, bottom)``
    """
    left, right, top, bottom = padding
    h, w = x.shape[-2:]
    return x[..., top:(h-bottom), left:(w-right)]


def prepare_class_weights(class_weights: Optional[Iterable[Union[int, float]]], device: torch.device):
    """
    Converts any iterable with numeric values into a torch float tensor on the desired device.
    If the input is `None`, the output is also `None`.
    This boilerplate is often needed to convert class weights to exact type needed by pytorch.
    """
    if class_weights is not None:
        return torch.tensor(class_weights, device=device, dtype=torch.float)