from collections import OrderedDict
from typing import Dict, List

from hooloovoo.utils.format_tools import format_dict, as_block, v_paste
from hooloovoo.utils.functional import default


class Tree:
    """
    Can be used to drop arbitrary values into.
    """
    def __init__(self, d: Dict = None):
        self.__kv = OrderedDict()
        for k, v in default(d, OrderedDict()).items():
            if isinstance(v, dict):
                v2 = self.__class__(v)
            elif isinstance(v, list):
                v2 = self._from_list(v)
            else:
                v2 = v

            self._add_kv(k, v2)

    @classmethod
    def _from_list(cls, l: List) -> List:
        l2 = []
        for li in l:
            if isinstance(li, dict):
                li2 = cls(li)
            elif isinstance(li, list):
                li2 = cls._from_list(li)
            else:
                li2 = li
            l2.append(li2)
        return l2

    def _add_kv(self, k, v):
        if not isinstance(k, str):
            raise ValueError(self.__class__.__name__ + " keys must be strings")
        self.__kv.__setitem__(k, v)

    def __repr__(self):
        return "Tree" + repr(self.__kv)[11:]

    def __str__(self):
        return as_block(v_paste([self.__class__.__name__], format_dict(self.to_dict())))

    def __eq__(self, other):
        if not isinstance(other, Tree):
            return False
        else:
            return self.__kv == other.__kv

    # For child classes
    # -----------------

    def _get_kv(self):
        return self.__kv

    def _set_kv(self, kv):
        self.__kv = kv

    # Direct property-like access
    # ---------------------------

    def __getattr__(self, item):
        return self.__kv.__getitem__(item)

    def __setattr__(self, key, value):
        if key is "_Tree__kv":
            super(Tree, self).__setattr__(key, value)
        else:
            return self._add_kv(key, value)

    def __delattr__(self, item):
        return self.__kv.__delitem__(item)

    # Quoted dict-like access
    # -----------------------

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        return self.__setattr__(key, value)

    def __delitem__(self, key):
        return self.__delattr__(key)

    # Extra sugar
    # -----------

    def with_default(self, default_value):
        items = self.__kv
        _add_kv = self._add_kv

        class Default:
            def __getattr__(self, item):
                if item in items:
                    return items.__getitem__(item)
                else:
                    # items.__setitem__(item, default_value)
                    _add_kv(item, default_value)
                    return default_value

            def __setattr__(self, key, value):
                # return items.__setitem__(key, value)
                return _add_kv(key, value)

            def __getitem__(self, item):
                return self.__getattr__(item)

            def __delitem__(self, key):
                return self.__delattr__(key)

        return Default()

    def __contains__(self, item):
        return item in self.__kv

    def keys(self):
        return self.__kv.keys()

    def __iter__(self):
        return iter(self.__kv.values())

    # serialization

    def to_dict(self) -> Dict:
        result = OrderedDict()
        for k, v in self.__kv.items():
            if isinstance(v, Tree):
                v2 = v.to_dict()
            elif isinstance(v, List):
                v2 = self._serialize_list(v)
            else:
                v2 = v

            result[k] = v2
        return result

    def _serialize_list(self, l: List):
        assert isinstance(l, list)

        l2 = []
        for li in l:
            if isinstance(li, Tree):
                li2 = li.to_dict()
            elif isinstance(li, list):
                li2 = self._serialize_list(li)
            else:
                li2 = li
            l2.append(li2)
        return l2

    @staticmethod
    def from_dict(d: Dict) -> "Tree":
        return Tree(d)

    def state_dict(self):
        return self.to_dict()

    def deepcopy(self):
        return Tree(self.to_dict())

    def load_state_dict(self, d):
        self.__setstate__(d)

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        self._set_kv(self.__class__(state)._get_kv())


def t(**kwargs):
    return Tree(kwargs)
