import hashlib
import os
import re
import unicodedata
from abc import ABC
from contextlib import contextmanager
from importlib import util as import_utils
from typing import NamedTuple, Generic, TypeVar, Sized, Iterable, Callable, Tuple

import numpy as np

A = TypeVar('A')
B = TypeVar('B')

F = TypeVar('F')
G = TypeVar('G')

K = TypeVar('K')
V = TypeVar('V')

X = TypeVar('X')
Y = TypeVar('Y')


Lazy = Callable[[], A]


def slugify(value, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces to hyphens.
    Remove characters that aren't alphanumerics, underscores, or hyphens.
    Convert to lowercase. Also strip leading and trailing whitespace.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    value = re.sub(r'[-\s]+', '-', value)
    return value


def hexdigest(o):
    return hashlib.sha256(repr(o).encode()).hexdigest()


def hex_of_hash(o, length: int = 6):
    """
    Creates a hex string based on the string representation of an object. Prepends zeros if the hex string is shorter
    than the requested length.
    """
    prefix = hexdigest(o)
    return ("0" * max(length - len(prefix), 0) + prefix)[:length]


def get_module_dir(module_name):
    return os.path.dirname(import_utils.find_spec(module_name).origin)


def seed_from(o):
    """
    Creates an integer seed between 0 and 2**32 - 1 from the repr of the given object
    """
    return int(hexdigest(o), 16) % (2**32 - 1)


@contextmanager
def np_seed(seed=0):
    """
    Run a block of code with a certain numpy seed,
    return to the previous RNG state after the code block ends successfully or throws an error.
    """
    seed = seed
    state = None

    try:
        state = np.random.get_state()
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


class SizedIterable(Sized, Iterable, ABC):
    pass


class LRTB(NamedTuple):
    left: int
    right: int
    top: int
    bottom: int

    @staticmethod
    def all(pad: int) -> "LRTB":
        return LRTB(pad, pad, pad, pad)


class BBox(NamedTuple):
    """Min and max row and column coordinates of a bounding box."""
    rmin: int
    rmax: int
    cmin: int
    cmax: int


class HW(Generic[A], NamedTuple):
    height: A
    width: A

    @staticmethod
    def square(length: A) -> "HW[A]":
        return HW(length, length)

    @staticmethod
    def get_from(*args) -> "HW":

        if len(args) == 1:
            args = args[0]
        elif len(args) > 2:
            raise ValueError("Expected at most 2 arguments")

        try:
            if len(args) == 1:
                h, w = args, args
            elif len(args) == 2:
                h, w = args
            else:
                raise ValueError("Cannot make HW with more than 2 values")
        except TypeError:
            h, w = args, args

        return HW(h, w)

    def wh(self) -> Tuple[A, A]:
        """Returns a tuple `(width, height)` instead of the default `(height, width)`"""
        class _WH(Generic[A], NamedTuple):
            width: A
            height: A
        return _WH(self.width, self.height)


class Var:
    def __init__(self, x):
        self.value = x
