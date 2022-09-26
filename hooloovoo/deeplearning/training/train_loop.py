from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum, auto
from typing import Callable, Any, Generator, List, Dict, Optional

import numpy as np
import torch
# noinspection PyUnresolvedReferences
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from hooloovoo.deeplearning.networks.controls import Controls
from hooloovoo.utils.arbitrary import Var
from hooloovoo.utils.timer import Timer, TimerException, hms, time_hms
from hooloovoo.utils.track import Tracker
from hooloovoo.utils.tree import Tree


class Event(Enum):
    TRAINING_STARTED = auto()
    TRAINING_RESUMED = auto()
    TRAINING_ENDED = auto()
    TRAINING_ABORTED = auto()
    EPOCH_END = auto()
    TRAIN_STEP_START = auto()
    TRAIN_STEP_END = auto()


class StopTraining(Exception):
    def __init__(self, msg=None, **kwargs):
        super(StopTraining, self).__init__(msg)
        self.kwargs = kwargs


class TrainLoop(ABC, Tracker):

    def __init__(self, model, dataloader, optimizer, scheduler=None):
        super(TrainLoop, self).__init__()
        self._state: Tree = Tree()
        # noinspection PyTypeChecker
        self.event_handlers: Dict[Event, List[Callable]] = OrderedDict((x, []) for x in iter(Event))

        self.model: Controls = model
        self.dataloader: DataLoader = dataloader
        self.optimizer: Optimizer = optimizer
        self.scheduler: Any = scheduler

    @property
    def state(self):
        return self._state

    # ------------------------------------------------------------------------------------------------------------------
    # To be overridden

    @abstractmethod
    def forward_pass(self, example) -> Any:
        pass

    @abstractmethod
    def compute_loss(self, example, yhat) -> torch.Tensor:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Training

    def train(self, resume_from=None):
        try:
            if resume_from is None:
                self._event(Event.TRAINING_STARTED)
            else:
                self._event(Event.TRAINING_RESUMED, resume_from=resume_from)
            for example in self._generate_batches():  # infinite generator
                self._train_step(example)
        except StopTraining as e:
            self._event(Event.TRAINING_ENDED, error=e)
        except BaseException as e:
            self._event(Event.TRAINING_ABORTED, error=e)
            raise

    def _generate_batches(self, generator: Generator = None, restart: Callable[[], Generator] = None):
        if generator is None:
            generator = iter(self.dataloader)
        if restart is None:
            restart = lambda: iter(self.dataloader)

        batch_generator = Var(generator)
        while True:
            try:
                yield next(batch_generator.value)
            except StopIteration:
                self._event(Event.EPOCH_END)
                batch_generator.value = restart()

    def _train_step(self, example):
        self._event(Event.TRAIN_STEP_START, example=example)
        yhat = self.forward_pass(example)
        loss = self.compute_loss(example, yhat)
        loss.backward()
        self._event(Event.TRAIN_STEP_END, example=example, yhat=yhat, loss=loss)

    # ------------------------------------------------------------------------------------------------------------------
    # Event handling

    def _event(self, event: Event, **kwargs):
        for handler in self.event_handlers[event]:
            handler(**kwargs, state=self._state, event=event)

    def add_handler(self, event: Event, handler: Callable, pos: int = None):
        if self.find_handler(event, handler) is not None:
            raise ValueError("handler already added")

        handlers = self.event_handlers[event]
        if pos is None:
            pos = len(handlers)
        handlers.insert(pos, handler)

    def del_handler(self, event: Event, handler: Callable):
        self.event_handlers[event].remove(handler)

    def shift_handler(self, event: Event, handler: Callable, to: int = float("-inf")):
        handlers = self.event_handlers[event]
        i = handlers.index(handler)
        handlers.pop(i)
        new_i = i + to
        if new_i > len(handlers):
            new_i = len(handlers)
        if new_i < 0:
            new_i = 0
        handlers.insert(new_i, handler)

    def find_handler(self, event: Event, handler: Callable) -> Optional[int]:
        try:
            return self.event_handlers[event].index(handler)
        except ValueError:
            return None

    # ----------------
    # Default handlers

    def add_default_handler_zero_grad(self):
        """
        Adds a handler that zeros the gradient in the optimizer at the beginning of each training step.
        """
        self.add_handler(Event.TRAIN_STEP_START, lambda **kwargs: self.optimizer.zero_grad())

    def add_default_handler_optim_step(self, closure: Callable = None):
        """
        Adds a handler that zeros steps the optimizer at the end of each training step.
        :param closure: an optional closure that will be passed to the optimizer.step()
        """
        if closure is None:
            # noinspection PyArgumentList
            f = lambda **kwargs: self.optimizer.step()
        else:
            f = lambda **kwargs: self.optimizer.step(closure)
        self.add_handler(Event.TRAIN_STEP_END, f)

    def add_default_handler_train_timer(self):
        def do(f):
            def _do(**_kwargs):
                timer = self.state.with_default(Timer()).timer
                try:
                    f(timer)
                except TimerException:
                    pass
            return _do
        self.add_handler(Event.TRAINING_STARTED, do(Timer.start))
        self.add_handler(Event.TRAINING_RESUMED, do(Timer.resume))
        self.add_handler(Event.TRAINING_ENDED, do(Timer.pause))
        self.add_handler(Event.TRAINING_ABORTED, do(Timer.pause))

    def add_default_handler_example_epoch_count(self):
        def bump_example(yhat, **_kwargs):
            batch_size = yhat.shape[0]
            self.state.with_default(0).n_batches += 1
            self.state.with_default(0).n_examples += batch_size
            self.state.with_default(0).n_examples_current_epoch += batch_size

        def bump_epoch(**_kwargs):
            self.state.n_epochs += 1
            self.state.n_examples_current_epoch = 0
        self.state.n_epochs = 1

        def resume_epoch(**_kwargs):
            # When resuming, we restart loading all examples, so discard examples from the current epoch
            self.state.n_examples_current_epoch = 0
            # It is hard to say what epoch we are in, since the epoch before the resume was only half finished.
            # So we keep the epoch counter at the current value.

        self.add_handler(Event.TRAIN_STEP_END, bump_example)
        self.add_handler(Event.EPOCH_END, bump_epoch)
        self.add_handler(Event.TRAINING_RESUMED, resume_epoch)

    def add_default_handler_track_loss(self):
        def log_loss(loss: torch.Tensor, **_kwargs):
            self.state.loss = loss.item()

            loss_history: List = self.state.with_default([]).loss_history
            loss_history.append(self.state.loss)
            if len(loss_history) > len(self.dataloader):
                loss_history.pop(0)

            self.state.mean_epoch_loss = sum(loss_history)/len(loss_history)

            minimum_mean_epoch_loss = self.state.with_default(Tree(dict(loss=np.inf))).minimum_mean_epoch_loss
            if self.state.mean_epoch_loss < minimum_mean_epoch_loss.loss:
                self.state.minimum_mean_epoch_loss = Tree(dict(
                    loss=self.state.mean_epoch_loss,
                    epoch=self.state.n_epochs if "n_epochs" in self.state else -1,
                    example=self.state.n_examples if "n_examples" in self.state else -1
                ))

        self.add_handler(Event.TRAIN_STEP_END, log_loss)

    # I list the methods manually because using 'dir' returns them in alphabetical order.
    _default_handlers = [
        add_default_handler_zero_grad,
        add_default_handler_optim_step,
        add_default_handler_train_timer,
        add_default_handler_example_epoch_count,
        add_default_handler_track_loss,
    ]

    def add_default_handlers(self):
        for handler_adder in self._default_handlers:
            handler_adder(self)

    # ------------------------------------------------
    # Non-default handlers, they require configuration

    def add_handler_display_console(self, every="10s"):
        def display_console(**_kwargs):
            time_training_running = self.state.timer.duration
            time_since_display_console = \
                time_training_running - self.state.with_default(float("-inf")).time_of_display_console

            n_examples = self.state.n_examples
            n_examples_since_display_console = n_examples - self.state.with_default(0).n_examples_of_display_console
            if time_since_display_console > hms(every):
                self.state.time_of_display_console = time_training_running
                self.state.n_examples_of_display_console = n_examples

                print("({t}) example: {n_ex} of ~{ep_size:.0f} in epoch {n_ep} ({n_ex_t} total)"
                      " ; loss = {loss:.3f} mean epoch loss = {mel:.3f}"
                      " (min = {mmel:f} at epoch {mmelep:d}, example {mmelex:d})"
                      " ; {speed:.1f} examples/sec".format(
                       t="{:02d}:{:02d}:{:02d}".format(*time_hms(time_training_running)),
                       loss=self.state.loss,
                       mel=self.state.mean_epoch_loss,
                       mmel=self.state.minimum_mean_epoch_loss.loss,
                       mmelep=self.state.minimum_mean_epoch_loss.epoch,
                       mmelex=self.state.minimum_mean_epoch_loss.example,
                       speed=n_examples_since_display_console / time_since_display_console,
                       n_ex=self.state.n_examples_current_epoch,
                       n_ex_t=self.state.n_examples,
                       ep_size=len(self.dataloader) * self.state.n_examples/self.state.n_batches,
                       n_ep=self.state.n_epochs),
                      flush=True
                      )
        self.add_handler(Event.TRAIN_STEP_END, display_console)

    # ------------------------------------------------------------------------------------------------------------------
    # Saving/restoring

    def save(self, path: str):
        """
        To make custom classes compatible with loading/saving, make them implement the pickle protocol
        """
        self.save_model(path + ".model.pth")
        self.save_train(path + ".train.pth")

    def save_model(self, path):
        torch.save(self.state_dict()["model"], path)

    def save_train(self, path):
        torch.save(self.state_dict()["train"], path)

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "train": {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "state": self.state.state_dict(),
                "tracked": self._tracked
            }
        }

    def load(self, path: str, **kwargs):
        """
        To make custom classes compatible with loading/saving, make them implement the pickle protocol
        """
        d = {
            "model": torch.load(path + ".model.pth", **kwargs),
            "train": torch.load(path + ".train.pth", **kwargs),
        }
        self.load_state_dict(d)

    def load_state_dict(self, d: Dict):
        self.model.load_state_dict(d["model"])
        self.optimizer.load_state_dict(d["train"]["optimizer"])
        if d["train"]["scheduler"] is not None:
            self.scheduler.load_state_dict(d["train"]["scheduler"])
        self._state.load_state_dict(d["train"]["state"])
        self._tracked.update(d["train"]["tracked"])


if __name__ == "__main__":
    print(Event)
