from typing import Iterator
from typing import Union, Iterable, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from hooloovoo.deeplearning.inference.infer_piecewise import infer_piecewise
from hooloovoo.deeplearning.networks.controls import Controls
from hooloovoo.utils.functional import default
from hooloovoo.utils.imagetools import image_as_numpy


def plot_train_example(example, yhat, loss, fg_class=1, bg_class=0, **kwargs):
    """
    :param example: x and y in NCHW format
    :param yhat: output in NCHW format BEFORE softmax
    :param loss: in HW format
    :param fg_class: which channel contains the the class you are interested in?
    :param bg_class: which channel is the 'nothing' class?
    :param kwargs:
    :return: A plot showing, x, y,
    the channels of the foreground and background class,
    predicted y for the foreground and the pixel-wise loss
    """
    x, y = [image_as_numpy(img[0]) for img in example]
    yhat_bg, yhat_fg = map(image_as_numpy, yhat[0, [bg_class, fg_class]])
    _egm_bg, segm_fg = map(image_as_numpy, yhat.softmax(1)[0, [bg_class, fg_class]])
    loss = image_as_numpy(loss[0])

    fig_kw = {"figsize": (14, 8), "constrained_layout": True}
    fig_kw.update(kwargs)

    fig = plt.figure(**fig_kw)
    fig.clf()
    axes = fig.subplots(nrows=2, ncols=3)

    axes[0, 0].imshow(x)
    axes[0, 0].set_title(r'$x$')
    axes[1, 0].imshow(y)
    axes[1, 0].set_title(r'$y$')

    img01 = axes[0, 1].imshow(yhat_fg)
    axes[0, 1].set_title(r'$\hat{y}$ fg')
    fig.colorbar(mappable=img01, ax=axes[0, 1])
    img11 = axes[1, 1].imshow(yhat_bg)
    axes[1, 1].set_title(r'$\hat{y}$ bg')
    fig.colorbar(mappable=img11, ax=axes[1, 1])

    axes[0, 2].imshow(segm_fg, vmin=0, vmax=1)
    axes[0, 2].set_title(r'segmentation fg')
    imgl = axes[1, 2].imshow(loss)
    axes[1, 2].set_title(r'per pixel loss')
    fig.colorbar(mappable=imgl, ax=axes[1, 2])

    return fig


def plot_eval_example(x, y, yhat, fg_channel=1, **kwargs):
    """
    :param x: HW, CHW or NCHW where N=1
    :param y: HW or NHW where N=1
    :param yhat: NCHW AFTER softmax
    :param fg_channel: which channel in yhat to show
    """
    fig_kw = {"figsize": (14, 4), "constrained_layout": True}
    fig_kw.update(kwargs)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, **fig_kw)
    ax1.imshow(image_as_numpy(x))
    ax2.imshow(image_as_numpy(y))
    ax3.imshow(image_as_numpy(yhat[0, fg_channel]), vmin=0, vmax=1)
    return fig


XYList = Union[Dataset, Iterable[Tuple[torch.Tensor, torch.Tensor]]]
EvalResults = Iterator[Tuple[float, float, Figure]]


class EvalLoop:
    def __init__(self, model: Controls, device: torch.device, infer_kwargs=None, fig_kwargs=None):
        self.model = model
        self.device = device
        self.infer_kwargs = default(infer_kwargs, {})
        self.fig_kwargs = default(fig_kwargs, {})

    def evaluate(self, ds: XYList, n_eval: int, n_plot: int, fg_class: int = 1) -> EvalResults:
        """
        :param ds: pairs of x and y images, must both be in CHW and HW format, not BCHW and BHW
        :param n_eval: how many images of ds to evaluate
        :param n_plot: how many images of ds to plot
        :param fg_class: which class to plot
        :return:
        """
        assert len(ds) >= n_eval >= n_plot
        loss_fn = torch.nn.CrossEntropyLoss()

        for i, (x, y) in enumerate(ds):
            if i < n_eval:
                yhat = infer_piecewise(model=self.model, x=x.unsqueeze(0), **self.infer_kwargs)

                # loss
                loss = loss_fn(yhat, y.unsqueeze(0))

                # iou scores
                probs = yhat.softmax(1)
                y_pred = image_as_numpy(probs[0, fg_class] > 0.5)
                y_true = image_as_numpy(y)
                intersection = np.sum(np.logical_and(y_pred, y_true))
                union = np.sum(np.logical_or(y_pred, y_true))
                iou_score = intersection / union if union > 0 else np.nan

                # plot
                figure = plot_eval_example(x, y, probs, **self.fig_kwargs) if i < n_plot else None

                yield loss, iou_score, figure
            else:
                break


def tb_log_eval(writer: SummaryWriter, results: Dict[str, EvalResults], verbose=False, *args, **kwargs):
    iou_medians = {}
    iou_means = {}
    iou_scores = []

    loss_medians = {}
    loss_means = {}
    losses = []

    for prefix, result in results.items():
        if verbose: print(f"evaluating {prefix} example: ", end='')
        for i, (loss, iou_score, fig) in enumerate(result):
            if verbose: print(str(i + 1), end=', ')
            losses.append(loss)
            if not np.isnan(iou_score):
                iou_scores.append(iou_score)
            if fig is not None:
                writer.add_figure(tag="eval/{}/example_{}".format(prefix, i), figure=fig, *args, **kwargs)
        if verbose: print()
        writer.add_histogram(tag="eval/iou/{}".format(prefix), values=np.array(iou_scores), *args, **kwargs)
        writer.add_histogram(tag="eval/loss/{}".format(prefix), values=np.array(losses), *args, **kwargs)
        # print(prefix + ": " + str(result.iou_scores))
        loss_medians[prefix] = np.median(losses)
        loss_means[prefix] = np.mean(losses)
        iou_medians[prefix] = np.median(iou_scores)
        iou_means[prefix] = np.mean(iou_scores)
    writer.add_scalars("eval/loss/median", loss_medians, *args, **kwargs)
    writer.add_scalars("eval/loss/mean", loss_means, *args, **kwargs)
    writer.add_scalars("eval/iou/median", iou_medians, *args, **kwargs)
    writer.add_scalars("eval/iou/mean", iou_means, *args, **kwargs)
    writer.close()
    return loss_medians, loss_means, iou_medians, iou_means
