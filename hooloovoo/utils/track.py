from enum import Enum
from typing import Dict, Any, Callable, NoReturn, Tuple

from hooloovoo.utils.arbitrary import V, Lazy
from hooloovoo.utils.functional import default, raise_

Condition = Callable[[V, V], bool]


class Tracker:

    def __init__(self):
        self._tracked = {}

    def track(self, **default_value: V):
        """
        Adds a concise mechanism to build a function that only returns true after a certain value has changed according
        to a certain condition.

        EXAMPLE::

            TODO

        :param default_value: The initial value of the parameter to track.
        :return: callable with arguments: 1) the condition on which to return True and 2) auto_update, a bool.
        """
        if len(default_value) != 1:
            raise Exception(f"Can only track exactly 1 value, {len(default_value)} given")

        key, default_value_ = default_value.popitem()
        self._tracked.setdefault(key, default_value_)

        def get_condition(condition: Condition[V], new: Lazy[V] = None, auto_update: bool = True) -> "TrackedVariable":
            """
            :param condition: condition to trigger, (old, new) -> bool
            :param new: how to get the new value of the variable
            :param auto_update: should the old value automatically be replaced by the new value if the condition is met?
                Default is True.
            :return: A `TrackedVariable`.
            """
            return TrackedVariable(self._tracked, key, condition=condition, new=new, auto_update=auto_update)

        return get_condition


class _Op(Enum):
    PURE = 0
    AND = 1
    OR = 2
    NOT = -1


# !noinspection PyUnboundLocalVariable,PyUnresolvedReferences
class _Exp:
    def __init__(self, op: _Op, *args: "_Exp"):
        self._op = op
        self._args = args
        self._check()

    def _raise_bad_op(self) -> NoReturn:
        raise AssertionError("Unknown operator: " + str(self._op))

    def _check(self):
        for arg in self._args:
            assert isinstance(arg, _Exp)
        if self._op == _Op.PURE:
            assert isinstance(self, TrackedVariable)
        elif self._op == _Op.AND:
            assert len(self._args) == 2
        elif self._op == _Op.OR:
            assert len(self._args) == 2
        elif self._op == _Op.NOT:
            assert len(self._args) == 1
        else:
            self._raise_bad_op()

    def __and__(self, other) -> "_Exp":
        return _Exp(_Op.AND, self, other)

    def __or__(self, other) -> "_Exp":
        return _Exp(_Op.OR, self, other)

    def __neg__(self) -> "_Exp":
        return _Exp(_Op.NOT, self)

    def _eval(self, new: Dict[str, Any]) -> bool:
        """
        Evaluates the expression and modifies the `new` input parameter in-place such that it contains all the new
        values of each tracked variable.
        """
        if self._op == _Op.PURE:
            assert isinstance(self, TrackedVariable)
            triggered, v = self._call(new.get(self.key), should_update=False)
            new[self.key] = v
        elif self._op == _Op.AND:
            triggered = self._args[0].eval(False)(**new) and self._args[1].eval(False)(**new)
        elif self._op == _Op.OR:
            triggered = self._args[0].eval(False)(**new) or self._args[1].eval(False)(**new)
        elif self._op == _Op.NOT:
            triggered = not self._args[0].eval(False)(**new)
        else:
            self._raise_bad_op()
        return triggered

    def eval(self, should_update: bool = None):
        def get_kwargs(**new: Any):
            triggered = self._eval(new)

            # only update values when the whole expression is triggered
            def update(tv: TrackedVariable):
                if should_update is None:
                    if tv.auto_update:
                        tv.update(new.get(tv.key))
                if should_update is True:
                    tv.update(new.get(tv.key))
            self.apply_to_leaves(update)

            return triggered

        return get_kwargs

    def __call__(self, **new) -> bool:
        return self.eval()(**new)

    def __bool__(self) -> bool:
        return self()

    def apply_to_leaves(self, f: Callable[["TrackedVariable"], None]):
        if self._op == _Op.PURE:
            assert isinstance(self, TrackedVariable)
            f(self)
        else:
            for arg in self._args:
                arg.apply_to_leaves(f)

    def update_all(self, **new):
        def update_leaf(tv: TrackedVariable):
            tv.update(new.get(tv.key))
        self.apply_to_leaves(update_leaf)

    def values(self) -> Dict[str, Any]:
        vals = {}

        def get_val(tv: TrackedVariable):
            vals[tv.key] = tv.value
        self.apply_to_leaves(get_val)
        return vals

    def _repr(self, root: bool) -> str:
        if self._op == _Op.PURE:
            assert isinstance(self, TrackedVariable)
            txt = "{}={}".format(self.key, self.value)
        elif self._op == _Op.AND:
            txt = "({} & {})".format(self._args[0]._repr(False), self._args[1]._repr(False))
        elif self._op == _Op.OR:
            txt = "({} | {})".format(self._args[0]._repr(False), self._args[1]._repr(False))
        elif self._op == _Op.NOT:
            txt = "-{}".format(self._args[0]._repr(False))
        else:
            self._raise_bad_op()

        if root:
            sur = "TrackedVariable({})" if self._op == _Op.PURE else "Exp`{}`"
            return sur.format(txt)
        else:
            return txt

    def __repr__(self) -> str:
        return self._repr(True)


class TrackedVariable(_Exp):
    def __init__(self, store: Dict[str, Any], key: str, condition: Condition[V], new: Lazy[V], auto_update: bool):
        super(TrackedVariable, self).__init__(_Op.PURE, self)
        self._store = store
        self.key = key
        self._condition = condition
        self._new = default(new, lambda: raise_(RuntimeError("No `new` function set for variable " + str(key))))
        self.auto_update = auto_update

    def _call(self, new: V = None, should_update: bool = None) -> Tuple[bool, V]:
        old: V = self.value
        new: V = self._new() if new is None else new  # cannot use default() here because it will always trigger _new()
        should_update = default(should_update, self.auto_update)
        triggered: bool = self._condition(old, new)

        # print(f"triggered {key}: {triggered}, old: {old}, new: {new}")
        if should_update:
            self.update(new)
        return triggered, new

    def __call__(self, new: V = None, should_update: bool = None) -> bool:
        return self._call(new, should_update)[0]

    def update(self, new: V = None) -> None:
        new: V = self._new() if new is None else new  # cannot use default() here because it will always trigger _new()
        self._store[self.key] = new

    @property
    def value(self) -> V:
        return self._store[self.key]
