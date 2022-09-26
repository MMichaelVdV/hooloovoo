from typing import List


def _as_lines(txt) -> List[str]:
    if isinstance(txt, list):
        return [str(x) for x in txt]
    else:
        return str(txt).splitlines()


def _v_paste(left, right, sep="") -> List[str]:
    left = _as_lines(left)
    right = _as_lines(right)

    left_len = max(map(len, left))
    height = max(len(left), len(right))
    lines = []
    for i in range(height):
        l = left[i] if i < len(left) else ""
        r = right[i] if i < len(right) else ""
        lines.append(l.ljust(left_len) + sep + r)
    return lines


def v_paste(*args, **kwargs) -> List[str]:
    if len(args) > 2:
        return _v_paste(args[0], v_paste(*args[1:], **kwargs), **kwargs)
    else:
        return _v_paste(args[0], args[1], **kwargs)


def as_block(lines: List[str]):
    return "\n".join(lines)


def as_tree(items: list, split0="┬", split1="├", split2="└", vert="│", horiz="─", fill=" "):
    if not (len(split1) == len(split2) == len(vert)):
        raise ValueError("split1, split2 and vert must have same lenght")
    if len(fill) != 1:
        raise ValueError("fill must be a single character")

    split_indent = fill*len(vert)
    horiz_indent = fill*len(horiz)

    out_lines = []
    for i, item in enumerate(items):
        item = _as_lines(item)
        head = item[0]
        body = item[1:]
        if i == 0:
            if i < len(items) - 1:
                split, down = split0, vert
            else:
                split, down = horiz, split_indent
        else:
            if i < len(items) - 1:
                split, down = split1, vert
            else:
                split, down = split2, split_indent
        out = [split + horiz + head] + ["{}{}".format(down, horiz_indent) + line for line in body]
        out_lines.extend(out)
    return out_lines


def format_list(l: list):
    n_digits = len(l) // 10 + 1
    format_digit_str = "{{:{}d}}.)".format(n_digits)
    format_digit = lambda d: format_digit_str.format(d)

    entries = []
    for i, li in enumerate(l):
        k = format_digit(i)
        if isinstance(li, dict):
            li2 = format_dict(li)
        elif isinstance(li, list):
            li2 = format_list(li)
        else:
            li2 = str(li)
        entry = v_paste(k, li2, sep=" ")
        entries.append(entry)
    return as_tree(entries)


def format_dict(d: dict):
    entries = []
    for k, v in d.items():
        if isinstance(v, dict):
            v2 = format_dict(v)
        elif isinstance(v, list):
            v2 = format_list(v)
        else:
            v2 = str(v)
        entry = v_paste(k, ":", v2, sep=" ")
        entries.append(entry)
    return as_tree(entries)


if __name__ == "__main__":
    # a = ["a", "a"]
    # b = ["b", "b", "b"]
    # print(as_block(v_paste(a, b, b, b, sep="-")))
    # print(as_block(as_tree([a, b])))
    # print(as_block(as_tree([a, b], horiz="──o ", fill=" ")))
    # print(as_block(as_tree([a, b], split1="#├", split2="#└", vert="#│", horiz="──o#", fill="#")))

    print(as_block(format_dict({
        "foo": 1,
        "some bar": 2,
        "buz": {
            "x": True,
            "y": False
        },
        "bux": "multi\nline",
        "long\nkey": 1,
        "key": {
            "other\nlong\nkey": True,
            "y": False
        },
        "foo\nbar": 42
    })))
