import copy
import re
import time
from datetime import timedelta, datetime
from enum import Enum
from functools import wraps
from typing import List, Dict, Optional


class Unit(Enum):
    nanoseconds = 1
    microseconds = 1e3
    milliseconds = 1e6
    seconds = 1e9

    def convert_ns(self, ns: int):
        return ns / self.value


class CheckPoint:
    def __init__(self, duration_ns: int, realtime_ns: int, unit: Unit = Unit.seconds):
        self.unit: Unit = unit
        self.duration_ns: int = duration_ns
        self.realtime_ns: int = realtime_ns

    def __repr__(self):
        return "CheckPoint<{:.3g}>({}ns, {}ns, unit={})".format(
            self.duration, self.duration_ns, self.realtime_ns, self.unit.name)

    def __eq__(self, other):
        return isinstance(other, CheckPoint) and self.__dict__ == other.__dict__

    @property
    def duration(self):
        return self.unit.convert_ns(self.duration_ns)

    @property
    def realtime(self):
        return self.unit.convert_ns(self.realtime_ns)

    def copy(self):
        return CheckPoint(self.duration_ns, self.realtime_ns, self.unit)

    def __getitem__(self, item):
        if item == 0:
            return self.duration
        elif item == 1:
            return self.realtime
        else:
            raise IndexError(item)

    def to_dict(self) -> Dict:
        return {"unit": self.unit.value,
                "duration_ns": self.duration_ns,
                "realtime_ns": self.realtime_ns}

    @staticmethod
    def from_dict(d: Dict) -> "CheckPoint":
        return CheckPoint(d["duration_ns"], d["realtime_ns"], Unit(d["unit"]))


class TimerState(Enum):
    INACTIVE = 0
    ACTIVE = 1
    PAUSED = 2


class TimerException(Exception):
    pass


class Timer:
    def __init__(self, unit: Unit = Unit.seconds):
        self.unit = unit
        self._laps: List[CheckPoint] = []
        self._state: TimerState = TimerState.INACTIVE
        # in ns, the current time is compared to this reference to determine the duration the timer is running
        self._duration_reference: Optional[int] = None
        # in ns, how long the timer was running when it got paused
        self._duration_when_paused: Optional[int] = None

    @property
    def state(self):
        return self._state

    def __repr__(self):
        return "Timer<{}, {:.3g}>(unit={})".format(self.state.name, self.peek.duration, self.unit.name)

    @property
    def _current_state_msg(self):
        return ", current state: " + self.state.name

    def start(self) -> CheckPoint:
        if self.state is not TimerState.INACTIVE:
            raise TimerException("Timer already started" + self._current_state_msg)
        else:
            assert len(self._laps) == 0
            cp = CheckPoint(0, time.time_ns())
            self._laps.append(cp)
            self._duration_reference = cp.realtime_ns
            self._state = TimerState.ACTIVE
            return cp

    @property
    def start_time(self):
        if self.state is TimerState.INACTIVE:
            raise TimerException("Cannot get start time of inactive timer" + self._current_state_msg)
        else:
            return self._laps[0].realtime

    @property
    def duration(self):
        return self.peek.duration

    @property
    def peek(self) -> CheckPoint:
        current_time = time.time_ns()
        if self.state is TimerState.INACTIVE:
            duration = 0
        elif self.state is TimerState.ACTIVE:
            duration = current_time - self._duration_reference
        elif self.state is TimerState.PAUSED:
            duration = self._duration_when_paused
        else:
            raise AssertionError
        return CheckPoint(duration, current_time, self.unit)

    def lap(self) -> CheckPoint:
        cp = self.peek
        self._laps.append(cp.copy())
        return cp

    def pause(self) -> CheckPoint:
        if self.state is not TimerState.ACTIVE:
            raise TimerException("Can only pause a running timer" + self._current_state_msg)
        cp = self.peek
        self._duration_reference = None
        self._duration_when_paused = cp.duration_ns
        self._state = TimerState.PAUSED
        return cp

    def resume(self) -> CheckPoint:
        if self.state is not TimerState.PAUSED:
            raise TimerException("Can only resume a paused timer" + self._current_state_msg)
        cp = self.peek
        self._duration_reference = cp.realtime_ns - self._duration_when_paused
        self._duration_when_paused = None
        self._state = TimerState.ACTIVE
        return cp

    def reset(self) -> CheckPoint:
        if self.state is TimerState.INACTIVE:
            raise TimerException("Cannot reset inactive timer" + self._current_state_msg)
        cp = self.peek
        self.__init__(self.unit)
        return cp

    @property
    def checkpoints(self) -> List[CheckPoint]:
        return copy.copy(self._laps)

    def to_dict(self):
        return {"state": self.state.value,
                "unit": self.unit.value,
                "laps": [cp.to_dict() for cp in self._laps],
                "duration": self.peek.duration_ns}

    @staticmethod
    def from_dict(d: Dict):
        state = TimerState(d["state"])
        unit = Unit(d["unit"])
        timer = Timer(unit)
        if state is not TimerState.INACTIVE:
            timer._duration_when_paused = d["duration"]
            timer._state = TimerState.PAUSED
            timer._laps = [CheckPoint.from_dict(cp) for cp in d["laps"]]
            if state is TimerState.ACTIVE:
                timer.resume()
        return timer

    def __getstate__(self):
        return self.to_dict()

    def __setstate__(self, state):
        self.__dict__.update(self.__class__.from_dict(state).__dict__)


def current_time_seconds():
    return time.time()


def time_hms(t):
    seconds_ = t
    seconds = int(seconds_ % 60)
    minutes_ = seconds_ // 60
    minutes = int(minutes_ % 60)
    hours_ = minutes_ // 60
    hours = int(hours_)
    return hours, minutes, seconds


def parse_hms(s: str):
    hms_ = re.fullmatch(r"(\d+h)?(\d+m)?(\d+s)?", s)
    if not hms_:
        raise ValueError("could not parse hms string: " + s)
    h, m, s = (0 if x is None else int(x[:-1]) for x in hms_.groups())
    return h, m, s


def hms(s: str):
    h, m, s = parse_hms(s)
    return s + m * 60 + h * 60 * 60


# noinspection PyPep8Naming
class throttle:
    """
    Decorator that prevents a function from being called more than once every
    time period.
    To create a function that cannot be called more than once a minute:
        @throttle("1m")
        def my_fun():
            pass

    To make sure at least one minute is between the time the previous call ENDED and the next call:
        @throttle("1m", blocking=True)
        def my_fun():
            pass

    adapted from: https://gist.github.com/ChrisTM/5834503
    """
    def __init__(self, every="0s", blocking=False):
        h, m, s = parse_hms(every)
        self.throttle_period = timedelta(
            seconds=s, minutes=m, hours=h
        )
        self.blocking = blocking

        self.time_of_last_call = datetime.min

        self.is_busy = False
        self.time_of_last_call_ended = datetime.min

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if self.blocking:
                if not self.is_busy:
                    now = datetime.now()
                    time_since_last_call_ended = now - self.time_of_last_call_ended
                    if time_since_last_call_ended > self.throttle_period:
                        self.is_busy = True
                        result = fn(*args, **kwargs)
                        self.is_busy = False
                        self.time_of_last_call_ended = datetime.now()
                        return result
            else:
                now = datetime.now()
                time_since_last_call = now - self.time_of_last_call

                if time_since_last_call > self.throttle_period:
                    self.time_of_last_call = now
                    return fn(*args, **kwargs)

        return wrapper
