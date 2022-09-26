import os
from typing import Iterator, Tuple, List


def touchfile(*path, exists_ok=False):
    f = os.path.join(*path)
    if not exists_ok and os.path.exists(f):
        raise FileExistsError(f)
    else:
        with open(f, "a") as fh:
            fh.write("")
    return f


def makedir(*path, exists_ok=False, recursive=False, mode=0o777):
    d = os.path.join(*path)
    if recursive:
        os.makedirs(d, exist_ok=exists_ok, mode=mode)
    else:
        try:
            os.mkdir(d, mode=mode)
        except FileExistsError as e:
            if exists_ok:
                pass
            else:
                raise e
    return d


def require_directory(d: str):
    if not os.path.isdir(d):
        raise ValueError("Could not open directory: " + d)


def walk_relative(top: str, *args, **kwargs) -> Iterator[Tuple[List[str], List[str], List[str]]]:
    """
    Exactly the same as `os.walk`,
    except that the first return value of each step is the relative path from the top directory,
    cut into pieces using `split_path`.
    """
    for parent, dirs, files in os.walk(top, *args, **kwargs):
        from_top = os.path.relpath(parent, top)
        pieces = split_path(from_top)
        yield pieces, dirs, files


def split_path(path: str) -> List[str]:
    p = os.path.normpath(path)
    return p.split(os.sep)


def symlink(src, dst, force=False, **kwargs):
    make_link = lambda: os.symlink(src, dst, **kwargs)
    if force:
        try:
            make_link()
        except FileExistsError:
            os.remove(dst)
            make_link()
    else:
        make_link()
