from typing import Optional, Union, Tuple, Any, List, Callable

import numpy as np
import torch
from PIL import ImageOps, Image
from PIL.Image import Image as PilImage
from matplotlib.cm import get_cmap
from matplotlib.colors import to_rgba
from skimage import segmentation, filters
from skimage.measure import label
from skimage.morphology import dilation, disk, skeletonize
from skimage.segmentation import mark_boundaries

from hooloovoo.monkeypatch import convex_hull_image
from hooloovoo.utils.arbitrary import LRTB, BBox
from hooloovoo.utils.functional import default, mapl

SomeImage = Union[PilImage, np.ndarray, torch.Tensor]


def project_segmentation_contours(image: np.ndarray, binary_mask: np.ndarray, color=(1.0, 0.0, 0.5)):
    """
    :returns: the input image with the mask superimposed as an outline.
    """
    return segmentation.mark_boundaries(image_as_numpy(image), image_as_numpy(binary_mask),
                                        color=color, outline_color=color)


def project_chull(in_image: np.ndarray, binary_mask: np.ndarray, color=(0, 0.5, 1)):
    """
    :returns: the input image with the convex hull superimposed plus the vertex coordinates as a list of(row, column),
    in counter clockwise order.
    """
    hull_mask, hull = convex_hull_image(binary_mask)
    hull_img = mark_boundaries(in_image, hull_mask, color=color, outline_color=color)
    vertices = []
    if hull is not None:
        for vertex in hull.vertices:
            vertices.append(tuple(hull.points[vertex, :]))
    return hull_img, vertices


def project_bbox(in_image: np.ndarray, binary_mask: np.ndarray, color=(1, 0, 0)) -> Tuple[Any, BBox]:
    """
    :return: An image with the bounding box superimposed as a contour plus the bounding box itself.
    """
    bb = bbox(binary_mask)
    if bb is not None:
        box_mask = np.zeros_like(binary_mask)
        box_mask[bb.rmin:bb.rmax, bb.cmin:bb.cmax] = 1
        box_img = mark_boundaries(in_image, box_mask, color=color, outline_color=color)
    else:
        box_img = in_image
    return box_img, bb


def project_skeleton(in_image: np.ndarray, binary_mask: np.ndarray, color=(0, 0, 0)):
    """
    :returns: the image with the skeleton superimposed on top, as well as a binary mask of the skeleton pixels.
    """
    skeleton_mask = skeletonize(binary_mask)
    skeleton_img = mark_boundaries(in_image, skeleton_mask, color=color, outline_color=None, mode='inner')
    return skeleton_img, skeleton_mask


def bbox(binary_mask: np.ndarray) -> Optional[BBox]:
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if np.sum(rows) == 0 or np.sum(cols) == 0:
        return None
    else:
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return BBox(rmin=rmin, rmax=rmax, cmin=cmin, cmax=cmax)


def binarize_mask(image, threshold=None):
    """
    :param image: A greyscale image, bright pixels are interpreted as foreground, dark pixels as background.
    :param threshold: A brightness threshold to binarize the image.
        If left to `None` then the 'otsu' threshold of the mask will be used.
    :return: a binary (0/1) integer array, 1 is foreground, 0 is background.
    """
    if threshold is None:
        threshold = filters.threshold_otsu(image)
    return (image > threshold).astype(np.int)


def largest_cc_and_neighbours(binary_mask: np.ndarray, include_distance: Union[int, float]) -> np.ndarray:
    """
    Get the largest connected component of a mask.
    Returns a mask comprising this largest cc and all other components within `include_distance` to this largest cc.
    """
    dilation_radius = include_distance//2
    target_dilated = dilation(binary_mask.astype(np.int), disk(dilation_radius))
    biggest_object = get_largest_cc(target_dilated)
    biggest_object_sharp = np.logical_and(binary_mask, biggest_object)
    return biggest_object_sharp.astype(np.int)


def remove_small_cc(binary_mask: np.ndarray, min_size: int) -> np.ndarray:
    """
    Remove all cc's smaller than the gives amount of pixels.
    """
    labels = label(binary_mask)
    sizes = np.bincount(labels.flat)
    big_enough = np.flatnonzero(sizes[1:] >= min_size) + 1
    return np.isin(labels, big_enough)


def has_foreground(img: PilImage, frac: float) -> bool:
    """
    :returns: `True` if the fraction of foreground pixels is higher than the given `frac`.
    """
    fraction_foreground = (image_as_numpy(img) > (255 / 2)).mean()
    return fraction_foreground > frac


def background_difficulty(x: PilImage, y: PilImage):
    """ Gives higher weights to images where the background colors resemble the foreground colors."""
    x_np = image_as_numpy(x)
    y_np = image_as_numpy(y) > (255 / 2)
    mean_foreground_rgb = np.array([[np.mean(x_np[y_np], axis=0)]])
    local_foreground_color_diff = np.sqrt(np.sum(np.square(x_np - mean_foreground_rgb), axis=2))
    background_difficulty_score = 1 - np.mean(local_foreground_color_diff[~y_np]) / np.sqrt(255 ** 2 * 3)

    # print(background_difficulty_score)
    # img_show(x, mean_foreground_rgb/255, local_foreground_color_diff/255,
    #          ncol=3, wait=True)

    return background_difficulty_score


def torch_image_to_numpy_image(image: torch.Tensor):
    n_dim = len(image.shape)
    if image.requires_grad:
        image = image.detach()
    image = image.cpu()
    wrong_dim = ValueError("Cannot convert {s:}-shaped tensor to numpy image".format(s=image.shape))

    if n_dim == 4:
        if image.shape[0] == 1:
            return torch_image_to_numpy_image(image.squeeze(0))
        else:
            raise wrong_dim
    elif n_dim == 3:
        if image.shape[0] == 3:
            return image.permute(1, 2, 0).numpy()
        elif image.shape[0] == 1:
            return torch_image_to_numpy_image(image.squeeze(0))
        else:
            raise wrong_dim
    elif n_dim == 2:
        return image.numpy()
    else:
        raise wrong_dim


def image_as_numpy(image: SomeImage) -> np.ndarray:
    if torch.is_tensor(image):
        return torch_image_to_numpy_image(image)
    elif isinstance(image, PilImage):
        # noinspection PyTypeChecker
        return np.array(image)
    elif isinstance(image, np.ndarray):
        return image
    else:
        raise TypeError("Cannot convert object of type {} to numpy array".format(type(image)))


def image_as_pil(image: SomeImage, mode=None) -> PilImage:
    image = image_as_numpy(image)
    if mode is None:
        if len(image.shape) == 2:
            mode = "L"
        elif len(image.shape) == 3:
            mode = "RGB"
        else:
            raise ValueError("Cannot guess mode for image of shape: {}".format(image.shape))
    if np.max(image) <= 1:
        image = image * 255
    image = np.uint8(image)
    return Image.fromarray(image, mode=mode)


def pad_pil_image(image: PilImage, padding: Optional[LRTB]) -> PilImage:
    # note different conventions for image dimensions between torch and PIL.
    if padding is not None and sum(padding) > 0:
        left, right, top, bottom = padding
        return ImageOps.expand(image, (left, top, right, bottom))
    else:
        return image


def crop_pil_image(image: PilImage, box: LRTB, strict=True):
    l, r, t, b = box
    crop = image.crop([l, t, r, b])
    if strict:
        crop.load()
    return crop


def get_largest_cc(mask: np.ndarray):
    labels = label(mask)
    if labels.max() == 0:  # assume at least 1 CC
        raise ValueError("mask does not contain any connected components")
    largets_cc = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largets_cc


def is_image_filename(f: str) -> bool:
    return f.lower().endswith((".jpg", ".png", ".tif", ".tiff"))


def _get_class_colors(n):
    if n < 2:
        raise ValueError("Need at least two classes")
    elif n == 2:
        return [(0, 0, 0, 1), (1, 1, 1, 1)]
    else:
        colors = get_cmap("Set1", n - 1).colors
        n_unique = len(set(mapl(tuple, colors)))
        if n_unique < n - 1:
            raise ValueError(
                f"The default color range (modified 'Set1', matplotlib) has only {n_unique + 1} distinct colors, "
                f"but the output has {n} classes (including background). "
                f"Please use a custom color list instead."
            )
        return [(0, 0, 0, 1)] + list(colors)


def color_classes(y: torch.Tensor, class_colors: List = None,
                  call_class: Callable[[torch.Tensor], torch.Tensor] = None) -> PilImage:
    """
    :param y: A tensor with segmentation scores (shape CxHxW) Each class is a separate channel.
    :param class_colors: Which colors to use for each class.
        By default a modified 'Set1' from 'matplotlib' is used for more than two colors
        ('Set1' is modified such that class 0 is black).
        Any list of matlab colors in string/hex/(r,g,b)/(r,g,b,a) is accepted.
    :param call_class: A function to assign classes to each pixel.
        Must take a floating point tensor and return a long tensor of shape (HxW).
        By default the 'argmax' in the channel direction is used.
    :return: A PIL image where each pixel is colored according to its class.
    """
    class_colors = default(class_colors, _get_class_colors(y.shape[0]))
    call_class = default(call_class, lambda y_: torch.argmax(y_, dim=0))
    class_assignment = call_class(y)
    image_data = np.empty((y.shape[-2], y.shape[-1], 4))
    for class_index, color in enumerate(class_colors):
        color_rgba = to_rgba(color)
        mask = class_assignment == class_index
        image_data[mask, :] = color_rgba
    return image_as_pil(image_data, "RGBA")
