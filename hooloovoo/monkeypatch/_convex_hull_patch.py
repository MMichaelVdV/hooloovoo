"""
Patched convex hull algorithm from:
<https://github.com/scikit-image/scikit-image/blob/master/skimage/morphology/convex_hull.py#L21>

The original implementation only returns the hull as a mask,
which is nice for visualisation but rather useless for downstream analysis.
This patch returns the hull coordinates as well.
"""

from itertools import product
from typing import Tuple, Optional

import numpy as np
from scipy.spatial import ConvexHull
# noinspection PyProtectedMember
from skimage._shared.utils import warn
from skimage.measure.pnpoly import grid_points_in_poly
# noinspection PyProtectedMember
from skimage.morphology._convex_hull import possible_hull
from skimage.util import unique_rows


def _offsets_diamond(ndim):
    offsets = np.zeros((2 * ndim, ndim))
    for vertex, (axis, offset) in enumerate(product(range(ndim), (-0.5, 0.5))):
        offsets[vertex, axis] = offset
    return offsets


def convex_hull_image(image, offset_coordinates=True, tolerance=1e-10) -> Tuple[np.ndarray, Optional[ConvexHull]]:
    """Compute the convex hull image of a binary image.
    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.
    Parameters
    ----------
    image : ndarray
        Binary input image. This array is cast to bool before processing.
    offset_coordinates : bool, optional
        If ``True``, a pixel at coordinate, e.g., (4, 7) will be represented
        by coordinates (3.5, 7), (4.5, 7), (4, 6.5), and (4, 7.5). This adds
        some "extent" to a pixel when computing the hull.
    tolerance : float, optional
        Tolerance when determining whether a point is inside the hull. Due
        to numerical floating point errors, a tolerance of 0 can result in
        some points erroneously being classified as being outside the hull.
    Returns
    -------
    hull : (M, N) array of bool
        Binary image with pixels in convex hull set to True.
    References
    ----------
    .. [1] https://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/
    """
    ndim = image.ndim
    if np.count_nonzero(image) == 0:
        warn("Input image is entirely zero, no valid convex hull. "
             "Returning empty image", UserWarning)
        return np.zeros(image.shape, dtype=np.bool_), None
    # In 2D, we do an optimisation by choosing only pixels that are
    # the starting or ending pixel of a row or column.  This vastly
    # limits the number of coordinates to examine for the virtual hull.
    if ndim == 2:
        coords = possible_hull(np.ascontiguousarray(image, dtype=np.uint8))
    else:
        coords = np.transpose(np.nonzero(image))
        if offset_coordinates:
            # when offsetting, we multiply number of vertices by 2 * ndim.
            # therefore, we reduce the number of coordinates by using a
            # convex hull on the original set, before offsetting.
            hull0 = ConvexHull(coords)
            coords = hull0.points[hull0.vertices]

    # Add a vertex for the middle of each pixel edge
    if offset_coordinates:
        offsets = _offsets_diamond(image.ndim)
        coords = (coords[:, np.newaxis, :] + offsets).reshape(-1, ndim)

    # repeated coordinates can *sometimes* cause problems in
    # scipy.spatial.ConvexHull, so we remove them.
    coords = unique_rows(coords)

    # Find the convex hull
    hull = ConvexHull(coords)
    vertices = hull.points[hull.vertices]

    # If 2D, use fast Cython function to locate convex hull pixels
    if ndim == 2:
        mask = grid_points_in_poly(image.shape, vertices)
    else:
        gridcoords = np.reshape(np.mgrid[tuple(map(slice, image.shape))],
                                (ndim, -1))
        # A point is in the hull if it satisfies all of the hull's inequalities
        coords_in_hull = np.all(hull.equations[:, :ndim].dot(gridcoords) +
                                hull.equations[:, ndim:] < tolerance, axis=0)
        mask = np.reshape(coords_in_hull, image.shape)

    return mask, hull
    #            ^ This is the only functional difference from the original implementation
