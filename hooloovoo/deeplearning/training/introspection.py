from typing import NamedTuple

import torch
from torch.nn import Module
from torch.optim import Optimizer

from hooloovoo.deeplearning.networks.controls import Controls
from hooloovoo.utils.functional import flatten


class ParamInfo(NamedTuple):
    name: str
    shape: torch.Size
    has_autograd: bool
    in_optimizer: bool


def optimizing_parameters(optimizer):
    return (group["params"] for group in optimizer.param_groups)


def parameter_overview(module: Module, optimizer: Optimizer = None):
    if optimizer is None:
        in_optimizer = lambda _: None
    else:
        optimizing_params = set(flatten(optimizing_parameters(optimizer)))

        def in_optimizer(p):
            try:
                optimizing_params.remove(p)
                return True
            except KeyError:
                return False

    for name, param in module.named_parameters():
        yield ParamInfo(
            name=name,
            shape=param.shape,
            has_autograd=param.requires_grad,
            in_optimizer=in_optimizer(param)
        )


def children_overview(module: Module, optimizer: Optimizer = None):
    for name, child in module.named_children():
        yield (name, parameter_overview(child, optimizer))


def print_param_overview(module: Controls, optimizer: Optimizer = None, width: int = 40):
    line = "  {name:%ds} {shape:<16s} {has_autograd:12s} {in_optimizer:}" % width
    line_length = 50 + width
    print("_" * line_length)
    print(line.format(name="name", shape="shape", has_autograd="autograd", in_optimizer="in optimizer"))
    print("-" * line_length)
    for name, overviews in children_overview(module, optimizer):
        print(":: {} ::".format(name))
        for overview in overviews:
            if overview.in_optimizer is None:
                in_optimizer = "N.A."
            else:
                in_optimizer = str(overview.in_optimizer)
            shape_str = "x".join("{:d}".format(d) for d in overview.shape)
            print(line.format(
                name=overview.name, shape=shape_str,
                has_autograd=str(overview.has_autograd), in_optimizer=in_optimizer
            ))
    print("-" * line_length)
    print("total amount of parameters: {}".format(module.n_parameters()))
    print("total amount of trainable parameters: {}".format(module.n_autograd_enabled_parameters()))
    print("_" * line_length)


