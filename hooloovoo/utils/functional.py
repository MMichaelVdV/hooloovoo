import itertools
from collections import defaultdict
from typing import List, Any, Callable, Tuple, Optional, Iterable, Dict

from hooloovoo.utils.arbitrary import A, B


def identity(x: A) -> A:
    return x


def partition(p: Callable[[A], bool], xs: Iterable[A]):
    yeas = []
    neas = []
    for x in xs:
        (yeas if p(x) else neas).append(x)
    return yeas, neas


def default(x, default_value):
    if x is None:
        return default_value
    else:
        return x


def maybe(f: Callable, o: Optional, if_missing: Any = None):
    if o is None:
        return if_missing
    else:
        return f(o)


def both(f: Callable[[A], B]) -> Callable[[Tuple[A, A]], Tuple[B, B]]:
    return lambda tup: (f(tup[0]), f(tup[1]))


def fst(xs):
    return xs[0]


def snd(xs):
    return xs[1]


def merge_dicts(*dict_args, accumulator=None):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.

    :param dict_args: Any number of dictionaries
    :param accumulator: Some dictionary-like object that can be updated with keys from dictionaries in ``dict_args``.
        Defaults to an ordinary empty dict.

    Code copied and modified from Stack Overflow answer by Aaron Hall https://stackoverflow.com/a/26853961
    """
    if accumulator is not None:
        result = accumulator
    else:
        result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def map_dict_values(f, d, accumulator=None):
    """
    Maps a function f over the values of dict d. Does not modify d. Returns a fresh dict or the accumulator updated
    with the ``key: f(value)`` pairs.
    """
    if accumulator is not None:
        result = accumulator
    else:
        result = {}
    for k, v in d.items():
        result[k] = f(v)
    return result


def group_by(xs: List[A], f: Callable[[A], B]) -> Dict[B, A]:
    ans = defaultdict(list)
    for x in xs:
        key = f(x)
        values = ans[key]
        values.append(x)
    return ans


def flatten(nested_lists):
    """Transforms a list of lists into a single flat list"""
    return [item for nested_list in nested_lists for item in nested_list]


def product(factors):
    total = 1
    for arg in factors:
        total *= arg
    return total


def sort_unique(l: List[Any]):
    return list(sorted(set(l)))


def find(l: List[A], p: Callable[[A], bool]) -> Optional[Tuple[int, A]]:
    for i, a in enumerate(l):
        if p(a):
            return i, a


def mapl(f: Callable, *xs: Iterable):
    return list(map(f, *xs))


def raise_(e):
    """Allows raising exceptions in a lambda"""
    raise e


def chunk_by(n: int, iterable: Iterable):
    """
    Turns an iterator into an iterator of chunks (which are also iterators),
    each chunk (except for, sometimes, the last one) has size `n`.

    https://stackoverflow.com/a/8998040/2496293
    """
    it = iter(iterable)
    while True:
        chunk_it = itertools.islice(it, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield itertools.chain((first_el,), chunk_it)
