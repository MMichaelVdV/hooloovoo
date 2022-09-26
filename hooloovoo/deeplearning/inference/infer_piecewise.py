import gc
from itertools import chain
from typing import Tuple, Generator, List, Iterator

import torch
from PIL.Image import Image as PilImage
from torch.nn import ZeroPad2d
from torchvision.transforms import functional as f

from hooloovoo.deeplearning.networks.controls import Controls
from hooloovoo.utils.arbitrary import LRTB, HW
from hooloovoo.utils.functional import identity
from hooloovoo.utils.tensortools import unpad_tensor, interpolate_tensor, crop_tensor

DEFAULT_MAX_SIZE = HW(float("inf"), float("inf"))
DEFAULT_CONTENT_PADDING = 72
DEFAULT_ZERO_PADDING = 24
DEFAULT_BLEEDOVER = 4


def infer_image(model: Controls, x: PilImage, out_device: torch.device = None, **kwargs) -> torch.Tensor:
    """
    Run a single image through a model, possibly in pieces, then apply softmax.

    See `deeplearning.inference.infer_piecewise.infer_piecewise` for details.

    :param model: A segmentation network
    :param x: An RGB image
    :param out_device: which device to use to stitch the image together, defaults to cpu.
    :param kwargs: passed to `infer_piecewise`.
        Note that `process_y_fn`, if given, is executed right before the softmax.
    :return: A Tensor of shape (CxHxW), corresponding to the model segmentation scores after softmax.
        Each class is a separate channel.
    """
    softmax = torch.nn.Softmax2d()
    if 'process_y_fn' in kwargs:
        process_y_fn = lambda piece: softmax(kwargs['process_y_fn'](piece))
    else:
        process_y_fn = softmax
    kwargs['process_y_fn'] = process_y_fn
    if out_device is None:
        out_device = torch.device('cpu')

    x_tensor = f.to_tensor(x)
    y_tensor = infer_piecewise(model, x_tensor.unsqueeze(0), out_device=out_device, **kwargs).squeeze(0)

    return y_tensor


def infer_piecewise(model: Controls, x: torch.Tensor,
                    max_size: "HW[int]" = DEFAULT_MAX_SIZE,
                    content_padding: int = DEFAULT_CONTENT_PADDING,
                    zero_padding: int = DEFAULT_ZERO_PADDING,
                    bleedover: int = DEFAULT_BLEEDOVER,
                    inference_device=None, out_device=None,
                    process_x_fn=identity, process_y_fn=identity, verbose=False) -> torch.Tensor:
    """
    Ensures that autograd is disabled for the whole network, then performs inference on the image x. The image
    can have any shape above the minimum size accepted by the network minus the zero padding.
    Padding, forward pass and unpadding are handled automatically.
    Images that are too big are automatically processed in pieces and then pasted back together.

    Don't use this function for training, to train on too large images, see the ``deeplearning.training.augment``
    module.

    An image can be so large that the forward pass fails due to lack of memory (either RAM or VRAM). To segment
    those images, ``max_size`` can be useful. If x is larger than ``max_size``, then the image is chopped up in
    overlapping pieces of at most ``max_size`` (+ extra padding, see note below), then inference is performed on
    each piece, then the pieces are pasted back together. If, ``max_size`` is not set, than it is assumed that
    the input will fit in the device's memory and no cutting/pasting is done.

    Beware of the interplay between the batch dimension and ``max_size``. Bigger batches require a lower
    ``max_size`` for the same device. In fact it is almost always more sensible to have a batch size of 1 if the
    image is too big.

    Note that if chopping up is required, memory needs to be allocated for the image, for the network working on an
    individual piece, and for the segmentation output. If even the sum of these parts is too big to fit on memory,
    then you'll have too cut up the image yourself and save each piece on the file system, then processes piece by
    piece.

    .. note::

        A naive solution would be to chop up the image in non-overlapping pieces. However, this means the network
        passes by the top, left, bottom and right edge of each piece. An edge of an image is always more poorly
        predicted since it is not possible to infer what is on an edge by looking at surrounding pixels. As such,
        the result, when pasted back together will have a grid-like appearance.

        So, a better solution is chop up the image in overlapping pieces. Then cut away a part of each piece that is
        close to the edge. Essentially, crop the pieces. And then put the cropped pieces back together. As such a
        border of ``content_padding`` pixels is added around each piece, then inference is done, and then a
        ``content_padding`` - ``bleedover`` pixel border is removed. Therefore each piece bleeds over by
        ``bleedover`` pixels into the next and vice-versa. This overlap region is averaged out between
        pieces. Some experimentation has shown that this gives better results than allowing no overlap.

        However, this not the end of the story. In addition to the padding above, each piece is also padded with
        an extra row of ``zero_padding`` zeros to make prediction better on edge-pixels.

        However, all this deconstruction-padding-reconstruction logic is handled in the background by the function.
        The only concern for using this function is knowing that a fairly large band is added around each piece,
        and therefore a good max size setting might be a bit lower than expected.


    :param model: A segmentation network.
    :param x: A Tensor representing an RGB image, dimensions NxCxHxW. Where N is the batch size, C = 3, and H, W are
        the width and height of the image.
    :param max_size: determines from which size images need to be chopped up.
    :param content_padding: how much padding of image content is added around each input piece before doing the forward
        pass.
    :param zero_padding: a black border added the input (piece) before doing the forward pass.
        It is added after being processed by ``process_x_fn``.
        It is removed after completing the forward pass, before being processed by ``process_y_fn``.
    :param bleedover: how many pixels of the ``content_padding`` are kept around each output piece when
        combining the output pieces into one image.
        This amount of pixels will be averaged with overlapping pixels from neighbouring pieces.
        The bleedover is bidirectional, as such the averaged region is twice as large as the bleedover parameter.
    :param inference_device: The device where inference will take place, may be different from ``x.device``.
    :param out_device: The device on which to reconstruct the output if needed. And the device on which to return
        the output.
    :param process_x_fn: A function to pre-process the input (piece) before doing inference. The function is executed on
        the inference device. If omitted, ``x`` is fed directly into the network.
    :param process_y_fn: A function to post-process the output (piece) after doing inference. The function is executed
        on the inference device. If omitted, inference returns the unscaled segmentation scores.
    :param verbose: Print whether image is processed in one go or in pieces. Also print which piece is being
        processed.
    :return: A Tensor containing unscaled segmentation scores processed by process_y_fn.
    """
    with torch.no_grad():
        return _infer_piecewise(
            model=model, x=x,
            max_size=max_size,
            content_padding=content_padding,
            zero_padding=zero_padding,
            bleedover=bleedover,
            inference_device=inference_device, out_device=out_device,
            process_x_fn=process_x_fn, process_y_fn=process_y_fn, verbose=verbose
        )


def _infer_piecewise(model: Controls, x: torch.Tensor,
                     max_size: "HW[int]",
                     content_padding: int,
                     zero_padding: int,
                     bleedover: int,
                     inference_device, out_device,
                     process_x_fn, process_y_fn, verbose=False) -> torch.Tensor:

    if zero_padding < 0:
        raise ValueError("zero_padding must be a natural number")
    if bleedover < 0:
        raise ValueError("bleedover must be a natural number")
    if content_padding < bleedover:
        raise ValueError("content_padding must be greater than bleedover")

    model.disable_training()
    if inference_device is None:
        inference_device = x.device
    if out_device is None:
        out_device = x.device

    # This takes no time when already on correct device.
    model.to(inference_device)

    pad_op = ZeroPad2d(padding=zero_padding)
    unpad_op = lambda t: unpad_tensor(t, LRTB.all(zero_padding))

    def infer_piece(x_piece: torch.Tensor):
        # What happens here:
        # (1) piece of image to inference device (e.g. a GPU) (2) pre-process on inference device
        # (3) add padding (4) feedforward piece through network (5) scale to input size (6) remove padding
        # (7) post-process on inference device (8) bring back to output device (typically CPU).
        #
        # Step (5) is needed for networks that return an image of lower/different resolution than their input.
        # This is done before un-padding because if the network output is rescaled, than the border added by
        # padding is also rescaled. So we need to scale the padding back up to original resolution before we
        # remove it. This way we can make sure the correct amount of padding is removed.

        _release_mem()

        processed_x_on_inference_device = process_x_fn(x_piece.to(inference_device))
        padded_x_on_inference_device = pad_op(processed_x_on_inference_device)
        y_piece = interpolate_tensor(model(padded_x_on_inference_device), padded_x_on_inference_device.shape)
        unpadded_y_on_inference_device = unpad_op(y_piece)
        processed_y_on_out_device = process_y_fn(unpadded_y_on_inference_device).detach().to(out_device)

        return processed_y_on_out_device

    assert model.n_autograd_enabled_parameters() == 0

    h, w = x.shape[-2:]
    max_h, max_w = max_size
    if h > max_h or w > max_w:

        large_padding = content_padding
        small_padding = content_padding - bleedover

        # To go from the large padding to the small padding
        def compute_smaller_crop_box(crop_box: LRTB) -> LRTB:
            left, right, top, bottom = crop_box
            _l = 0 if left == 0 else small_padding
            _t = 0 if top == 0 else small_padding
            r_ = 0 if right == w else small_padding
            b_ = 0 if bottom == h else small_padding
            return LRTB(left + _l, right - r_, top + _t, bottom - b_)

        crop_boxes_large, pieces = chop_big_tensor(x, max_size, HW(large_padding, large_padding))
        crop_boxes_small = [compute_smaller_crop_box(crop_box) for crop_box in crop_boxes_large]
        n_pieces = len(crop_boxes_large)

        def generate_output_chunk() -> Generator[torch.Tensor, None, None]:
            n = 1
            for crop_box_large, crop_box_small, piece in zip(crop_boxes_large, crop_boxes_small, pieces):
                large_piece = infer_piece(piece)
                padding = LRTB(*(abs(crop_box_large[edge] - crop_box_small[edge]) for edge in range(4)))
                small_piece: torch.Tensor = unpad_tensor(large_piece, padding)

                small_piece_size = HW(*tuple(small_piece.shape[-2:]))
                small_box_size = HW(crop_box_small[3] - crop_box_small[2], crop_box_small[1] - crop_box_small[0])
                assert small_piece_size == small_box_size

                if verbose:
                    print("processing piece: {} of {}".format(n, n_pieces))
                n += 1

                yield small_piece

        return combine_pieces_on_canvas(crop_boxes_small, generate_output_chunk())

    else:
        if verbose:
            print("processing image in one shot")
        return infer_piece(x)


def _release_mem():
    # Saves the day when the image barely fits in graphics card memory.
    # Pytorch seems to keeps memory around from e.g. previous infer_piecewise() runs.
    torch.cuda.empty_cache()  # cuda.empty_cache() doesn't freak out when cuda is not being used.

    # On some platforms memory keeps increasing after running each new piece on the CPU.
    gc.collect()  # TODO: I'm not sure if this helps

# ----------------------------------------------------------------------------------------------------------------------
# Stuff related to chopping up too large images


def generate_crop_boxes(image_size: HW, crop_size: HW, overlap_size: HW = HW(0, 0)) -> Generator[LRTB, None, None]:
    """
    Generate regularly spaced crop boxes for a large image, covering the entire image. Goes left then down.
    Crops coming at the right or bottom edge of image will often be smaller than ``crop_size_hw``.

    :param image_size: height and width of the image.
    :param crop_size: height and width of the crops.
    :param overlap_size: half of the bi-directional overlap in width and height between neighbouring crops.
        E.g. if the overlap width is set to 10 pixels, the two adjacent crops will overlap by 20 pixels.
    :return: a generator that iterates over crop bounding boxes, from left to right, then down and again from the left,
        like a typewriter. The bounding boxes are returned as: ``(left, right, top, bottom)``.
    """
    y = 0  # goes from top to bottom
    x = 0  # goes from left to right
    image_h, image_w = image_size
    crop_h, crop_w = crop_size
    overlap_h, overlap_w = overlap_size

    while True:
        # The bounding box not taking into account image boundaries
        _top = y - overlap_h
        _left = x - overlap_w
        bottom_ = y + crop_h + overlap_h
        right_ = x + crop_w + overlap_w

        # the bounding box taking into account image boundaries
        top = max(0, _top)
        left = max(0, _left)
        bottom = min(bottom_, image_h)
        right = min(right_, image_w)
        yield LRTB(left, right, top, bottom)

        if right < image_w:
            # the right edge of the image is not yet reached, so continue sliding right
            x += crop_w
        else:
            # the right edge of the image has been reached, check if I should go down or stop
            if bottom < image_h:
                # the bottom edge of the image has not yet been reached, but the right edge has been reached,
                # so go down and start from the left
                x = 0
                y += crop_h
            else:
                # the bottom-right corner of the image has been reached.
                return


def chop_big_tensor(x: torch.Tensor, crop_size: HW, overlap_size: HW = HW(0, 0))\
        -> Tuple[List[LRTB], Generator[torch.Tensor, None, None]]:
    """
    Makes a generator that chops up a large tensor up into smaller pieces.
    ``x`` should have shape ``([N1 , N2, ...] H, W)``. See also ``generate_crop_boxes``.

    .. note::

        The generated pieces are still backed by the big image tensor. So modifying them modifies the source image.
        If you don't want this, use ``tensor.clone()``.

    :returns: A list of crop boxes and a lazy generator of image pieces.
    """
    image_size = HW(*(x.shape[-2:]))
    crop_boxes = list(generate_crop_boxes(image_size, crop_size, overlap_size))

    def generator():
        for crop_box in crop_boxes:
            yield crop_tensor(x, crop_box)

    return crop_boxes, generator()


def _check_piece(crop_box: LRTB, piece: torch.Tensor, expected_dtype, expected_device):
    left, right, top, bottom = crop_box
    if right - left != piece.shape[-1] or bottom - top != piece.shape[-2]:
        raise ValueError("incompatible dimensions for crop box and image piece: {} and {}"
                         .format(crop_box, piece.shape))
    if piece.dtype != expected_dtype:
        raise ValueError("Got a piece with a different dtype than the supplied canvas. Either change the canvas dtype "
                         "or the dtype of the pieces")
    # does not have any runtime cost when already on correct device.
    return piece.to(expected_device)


def combine_pieces_on_canvas(crop_boxes: Iterator[LRTB], pieces: Generator[torch.Tensor, None, None],
                             canvas: torch.Tensor = None, record_dtype=torch.uint8) -> torch.Tensor:
    """
    Pastes multiple image pieces coming from different locations onto a black canvas. Overlapping pieces are averaged
    out.

    :param crop_boxes: iterator of crop boxes. Crop boxes must be specified as ``(left, right, top, bottom)``.
    :param pieces: A generator of image pieces that match the ``crop_boxes``.
        If the pieces are on a different device as ``canvas``, then they will be copied to the canvas device.
    :param canvas: The tensor on which to put the ``pieces``. Canvas values are replaced by the content of the pieces.
        The canvas must have the same dtype as the pieces, but does not need to be on the same device.
        If omitted:
            - a new black canvas is created that can hold all pieces.
            - the canvas is created on the same ``device`` as the first image piece.
    :param record_dtype: To keep track of averaging weights, this function holds a record of how many pieces have been
        pasted on each location. If there is reason to believe that more than 255 pieces will be placed on the same spot
        of the canvas, then choose something that has more instances than uint8.
    :return: A, tensor, the merge of the all image pieces.
    """

    # ugly low level details...

    crop_boxes = list(crop_boxes)
    first_crop_box = crop_boxes[0]
    first_piece = next(pieces)
    pieces = chain([first_piece], pieces)
    pieces_device = first_piece.device
    pieces_dtype = first_piece.dtype

    if canvas is None:
        h = 0
        w = 0
        for crop_box in crop_boxes:
            _, right, _, bottom = crop_box
            if bottom > h:
                h = bottom
            if right > w:
                w = right

        canvas_dtype = first_piece.dtype
        canvas_shape = list(first_piece.shape)
        canvas_shape[-1] = w
        canvas_shape[-2] = h
        canvas = torch.zeros(canvas_shape, dtype=canvas_dtype, device=pieces_device)
    else:
        # only doing this for checking the dtype.
        _check_piece(first_crop_box, first_piece, canvas.dtype, pieces_device)
        h, w = canvas.shape[-2:]

    record = torch.zeros((h, w), dtype=record_dtype, device=pieces_device)

    # and now the real work...

    for crop_box, piece in zip(crop_boxes, pieces):
        piece = _check_piece(crop_box, piece, pieces_dtype, canvas.device)
        left, right, top, bottom = crop_box
        crop = ..., slice(top, bottom), slice(left, right)

        numerator = record[crop].type(pieces_dtype)
        denominator = numerator + 1
        canvas[crop] = canvas[crop] * numerator/denominator + piece * 1/denominator

        record[crop] += 1

    return canvas
