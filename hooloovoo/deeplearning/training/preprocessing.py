import math
from typing import Union, List, Generator, Generic, NamedTuple

import numpy as np
from PIL.Image import Image
from torch.utils.data import Dataset, Subset

from hooloovoo.deeplearning.training.augmentation import random_crop_box
from hooloovoo.utils.arbitrary import HW, LRTB, A
from hooloovoo.utils.functional import identity, raise_, both


# ----------------------------------------------------------------------------------------------------------------------
# Helper methods for preprocessing images
from hooloovoo.utils.imagetools import crop_pil_image


def _equal_spread_crop_boxes(in_image_size: "HW[int]",
                             cropped_image_size: "HW[int]",
                             min_overlap: "HW[int]") -> Generator[LRTB, None, None]:
    in_h, in_w = in_image_size
    out_h, out_w = cropped_image_size
    ovr_h, ovr_w = min_overlap

    def compute_anchors(l, p, o):
        """
        The aim is to lay out 'n' pieces of size 'p' next to each other,
        with an overlap 'o_ >= o' between two subsequent pieces,
        where the total lenght after the pieces are layed out is 'l'.
        Additionally, 'n' should be as small as possible and all pieces must be spaced out evenly across 'l'.

        Notice that zero overlap would give us a lenght of 'p * n',
        therefore 'p * n - l' is the size that needs to be taken up by overlapping.
        There are 'n - 1' overlapping regions in 'n' pieces,
        as such, for a given 'n', the size of each overlap is 'o_ = (pn - l)/(n - 1)'.

        We want 'o_ >= o', therefore: '(pn - l)/(n - 1) >= o', or after rearrangments: 'n >= (l - o)/(p - o)'.
        The smallest natural number 'n' to satisfy this condition is ceil((l - o)/(p - o)).
        """
        if p >= l:
            return [0]
        elif o >= p:
            raise Exception("Minimal overlap between pieces must be smaller than piece size")
        else:
            n = math.ceil((l - o)/(p - o))
            o_ = (p * n - l)/(n - 1)
            return [i * (p - o_) for i in range(n)]

    def compute_ranges(l, p, o):
        anchors = compute_anchors(l, p, o)
        return [(int(anchor), int(anchor) + int(p)) for anchor in anchors]

    y_ranges = compute_ranges(in_h, out_h, ovr_h)
    x_ranges = compute_ranges(in_w, out_w, ovr_w)

    for top, bottom in y_ranges:
        for left, right in x_ranges:
            yield LRTB(left, right, top, bottom)


def equal_spread_crop_boxes(in_image_size: Union[int, "HW[int]"],
                            cropped_image_size: Union[int, "HW[int]"],
                            bidir_overlap: Union[int, float, "HW[Union[int, float]]"]) -> List[LRTB]:
    """
    Creates nicely spread crop boxes over a whole image, having the following properties:
    * The midpoints of each crop are spaced at equal distances on each axis.
    * Also crops at edges of the image are sized as specified in `out_image_hw` (*)
    * Crops overlap by at least the amount given in `bidir_overlap`.

    (*) except when the crop size is higher than the input image size.

    :param in_image_size: the size of the image to chop in pieces.
    :param cropped_image_size: the size of each crop, the crop boxes can be smaller than this if the image is too small.
    :param bidir_overlap: the minimal amount of overlap between adjacent crops. A number between zero and one is
        is interpreted as a ratio, bigger numbers are interpreted as pixels. Two numbers can be given to specify
        the overlap in width an height. The overlap will often be much higher than this if the crop does not fit
        an integer number of times into the image.
    """
    in_image_size = HW.get_from(in_image_size)
    cropped_image_size = HW.get_from(cropped_image_size)
    bidir_overlap = HW.get_from(bidir_overlap)

    def get_size_and_padding(out_size, bidir_overlap_):
        if 0 < bidir_overlap_ < 1:
            return out_size * (1 - bidir_overlap_), out_size * bidir_overlap_
        else:
            return out_size, bidir_overlap_

    out_h, out_w = cropped_image_size
    overlap_h, overlap_w = bidir_overlap
    out_h, padding_h = get_size_and_padding(out_h, overlap_h)
    out_w, padding_w = get_size_and_padding(out_w, overlap_w)

    return list(_equal_spread_crop_boxes(in_image_size, HW(out_h, out_w), HW(padding_h, padding_w)))


def random_crop_boxes(in_image_size: HW, cropped_image_size: HW, n: int) -> List[LRTB]:
    return [random_crop_box(in_image_size, cropped_image_size) for _ in range(n)]


def split_images(image: Image, *images, split_fn, **kwargs):
    """
    Splits several images in the same way. Usually done for input-output pairs.
    """
    for img in images:
        if not image.size == img.size:
            raise ValueError("Splitting must happen on images of equal size")
    size = image.size[::-1]  # PIL size: WH, not HW.
    boxes = split_fn(in_image_size=size, **kwargs)
    images_ = (image,) + images
    for box in boxes:
        yield [crop_pil_image(img, box) for img in images_]


# ----------------------------------------------------------------------------------------------------------------------
# Train-eval-test splits

class TrainTest(NamedTuple):
    train: A
    test: A


class TrainEval(NamedTuple):
    """
    Contains a train and test split. As wel as a random subset of the train split, which can be compared to the test
    split to detect over-fitting.
    """
    train: A
    eval: TrainTest  # TrainTest[A]


def split_random_n(n: int, size_test: Union[int, float]) -> TrainTest:  # -> TrainTest[np.array]
    """
    :param n: the size of your dataset
    :param size_test: the amount of test examples, can be a fraction (float) or an absolute amount (int)
    :return: two arrays, the first one with train indices, the second one with test indices.
    """
    if isinstance(size_test, float) and 0 <= size_test <= 1:
        n_test = int(n * size_test)
    elif isinstance(size_test, int) and 0 <= size_test <= n:
        n_test = size_test
    else:
        raise Exception("Invalid specification of test split size, "
                        "must be either float in range [0,1] or int in range [0, {}]".format(n))
    n_train = n - n_test
    indices = np.arange(n)
    np.random.shuffle(indices)
    return TrainTest(indices[:n_train], indices[n_train:])


# noinspection PyUnresolvedReferences,PyShadowingNames
def split_random(o: Union[int, list, np.array, Dataset], size_test: Union[int, float], out=TrainEval) \
        -> Union[TrainTest, TrainEval]:
    """
    Randomly splits an object into a train and test set or a train, eval-train and eval-test set.
    :param o: if an int, returns numpy arrays with indices, if an array, list or DataSet, return subsets.
    :param size_test: the amount of test examples, can be a fraction (float) or an absolute amount (int)
    :param out: Determines to only return train and test or to return train, eval-train and eval-test.
    :return: Either a 'TrainTest' or a 'TrainEval'. The eval-train set is created by taking the first n examples from
    the train set, where n is the lenght of the test set.
    """
    n = o if type(o) is int else len(o)
    get_subset = identity if type(o) is int \
        else (lambda ixs: np.array(o)[ixs]) if isinstance(o, np.ndarray) \
        else (lambda ixs: [o[i] for i in ixs]) if isinstance(o, list) \
        else (lambda ixs: Subset(o, ixs)) if isinstance(o, Dataset) \
        else raise_(TypeError("Cannot split object of type: " + str(type(o))))

    tr_i, te_i = split_random_n(n, size_test)
    tr, eval_te = both(get_subset)((tr_i, te_i))
    if issubclass(out, TrainTest):
        return TrainTest(train=tr, test=eval_te)
    if issubclass(out, TrainEval):
        eval_tr = get_subset(tr_i[:len(te_i)])
        return TrainEval(train=tr, eval=TrainTest(train=eval_tr, test=eval_te))
    raise TypeError("Cannot make a split of type: " + str(out))
