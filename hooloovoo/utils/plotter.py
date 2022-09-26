import math
from typing import Union

import matplotlib
import matplotlib.backends.backend_agg as plt_backend_agg
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from cytoolz import concat
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hooloovoo.utils.functional import default
from hooloovoo.utils.imagetools import image_as_numpy


def auto_greyscale(ax: Axes, image: Union[Image.Image, np.ndarray, torch.Tensor], param_dict: dict = None):
    param_dict = default(param_dict, {})
    image = image_as_numpy(image)

    if len(image.shape) == 2 and np.max(image) <= 1 and np.min(image) >= 0:
        # greyscale image
        param_dict.setdefault("vmin", 0)
        param_dict.setdefault("vmax", 1)
        ax.imshow(image, **param_dict)
    elif len(image.shape) == 2:
        # heat map image
        axes_img = ax.imshow(image, **param_dict)
        ax.figure.colorbar(mappable=axes_img, ax=ax)
    else:
        ax.imshow(image, **param_dict)


def img_show(*images, nrow: int = None, ncol: int = None, wait=False, fig: Figure = None):
    matplotlib.use('Qt5Agg')
    fig: Figure = default(fig, plt.gcf())
    fig.clf()

    n_images = len(images)
    if ncol is None and nrow is None:
        nrow = 1
    if nrow is None:
        nrow = int(math.ceil(n_images/ncol))
    if ncol is None:
        ncol = int(math.ceil(n_images/nrow))
    if n_images > nrow * ncol:
        raise ValueError("Too many images ({}) for a {} x {} grid".format(n_images, nrow, ncol))

    axes = fig.subplots(nrows=nrow, ncols=ncol, squeeze=False)
    fig.tight_layout()
    for i, ax in enumerate(concat(axes)):
        auto_greyscale(ax, images[i])
    plt.pause(0.01)
    if wait:
        plt.waitforbuttonpress()


def figure_to_image(figure: Figure, close=True):
    """Render matplotlib figure to numpy format.

    Note that this requires the ``matplotlib`` package.

    Args:
        figure (matplotlib.pyplot.figure): matplotlib figure
        close (bool): Flag to automatically close the figure

    Returns:
        numpy.array: image in [CHW] order
    """
    canvas = plt_backend_agg.FigureCanvasAgg(figure)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
    image_chw = np.moveaxis(image_hwc, source=2, destination=0)
    if close:
        plt.close(figure)
    return image_chw
