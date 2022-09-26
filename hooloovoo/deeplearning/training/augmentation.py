import random

from torchvision.transforms import functional as f, ColorJitter

from hooloovoo.utils.arbitrary import LRTB, HW
from hooloovoo.utils.imagetools import crop_pil_image


def random_mirror_and_quarter_rot_fn():
    """
    Randomly rotates an image by a some straight angle and/or mirrors the image.
    Non-deterministic.

    By randomly and independently choosing whether or not to do
    - a horizontal flip,
    - a vertical flip and,
    - a 90 rotation.
    All possible combinations of mirroring and 90/180/270 degree rotations are achieved. Because, for example, a
    horizontal flip and vertical flip is equivalent to a 180 degree rotation. Or a vertical flip and 90 degree
    rotation is equivalent to a horizontal flip and -90 degree rotation.

    :return: a function that applies the random transform.
    """

    seed1 = random.random()
    seed2 = random.random()
    seed3 = random.random()

    def apply_transform(image):
        if seed1 > 0.5:
            image = f.vflip(image)

        if seed2 > 0.5:
            image = f.hflip(image)

        if seed3 > 0.5:
            image = f.rotate(image, angle=90)

        return image

    return apply_transform


def random_hflip_and_small_rot_fn(max_rot: float, expand: bool):
    """
    Adds a random horizontal flip and rotation to the images.
    Non-deterministic.

    :param max_rot: the maximum number of degrees an image can be rotated either clockwise or counterclockwise.
    :param expand: increase the output image size to fit the rotated image.
    :return: a function that applies the random transform.
    """
    seed1 = random.random()
    seed2 = random.random()
    
    def apply_transform(image):
        rot = (seed1 * 2 - 1) * max_rot

        if seed2 > 0.5:
            image = f.hflip(image)

        image = f.rotate(image, angle=rot, expand=expand)

        return image
    
    return apply_transform


def random_color_jitter_fn(brightness, contrast, saturation, hue):
    return ColorJitter(brightness, contrast, saturation, hue)


def random_crop_box(in_image_size: HW, out_image_size: HW) -> LRTB:
    """
    creates a function that takes a random crop from images.
    The crop box is the same each time the created function is called.
    """
    in_h, in_w = in_image_size
    out_h, out_w = out_image_size
    left_crop = random.randint(0, max(in_w - out_w, 0))
    right_crop = min(left_crop + out_w, in_w)
    upper_crop = random.randint(0, max(in_h - out_h, 0))
    lower_crop = min(upper_crop + out_h, in_h)

    return LRTB(left_crop, right_crop, upper_crop, lower_crop)


def random_crop_fn(in_image_size: HW, out_image_size: HW):
    """
    creates a function that takes a random crop from images.
    The crop box is the same each time the created function is called.
    """
    lrtb = random_crop_box(in_image_size, out_image_size)

    def apply_transform(image):
        return crop_pil_image(image, lrtb)

    return apply_transform
