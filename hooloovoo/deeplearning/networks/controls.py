from abc import ABC
from typing import Iterable

import torch
import torch.nn as nn

from hooloovoo.deeplearning.networks.inspection import Inspection
from hooloovoo.utils.functional import product


class Controls(Inspection, ABC):

    def disable_pretrainable_features_autograd(self):
        self.toggle_pretrainable_features_autograd(on=False)

    def enable_pretrainable_features_autograd(self):
        self.toggle_pretrainable_features_autograd(on=True)

    def toggle_pretrainable_features_autograd(self, on: bool):
        self.toggle_autograd(on=on, parameters=self.pretrainable_parameters())

    def disable_autograd(self, parameters: Iterable[torch.Tensor] = None):
        self.toggle_autograd(on=False, parameters=parameters)

    def enable_autograd(self, parameters: Iterable[torch.Tensor] = None):
        self.toggle_autograd(on=True, parameters=parameters)

    def toggle_autograd(self, on: bool, parameters: Iterable[torch.Tensor] = None):
        """
        Toggles the autograd engine on/off for the given parameters. It is advisable to turn autograd off for all layers
        when doing inference, as this saves computation time.

        :param on: Set to ``True`` to turn the autograd engine on for the give layers. Set to ``False`` to turn it off.
        :param parameters: A parameter (=Tensor) iterator. Defaults to whole network.
        """
        if parameters is None:
            parameters = self.parameters()

        for param in parameters:
            param.requires_grad = on

        # To enable/disable dropout
        self.train(self.is_training())

    def named_autograd_enabled_parameters(self, module: nn.Module = None):
        if module is None:
            module = self
        return ((name, param) for name, param in module.named_parameters() if param.requires_grad)

    def n_autograd_enabled_parameters_per_child(self, children: Iterable[nn.Module] = None):
        if children is None:
            children = self.named_children()
        return ((name, self.n_autograd_enabled_parameters(child)) for name, child in children)

    def n_autograd_enabled_parameters(self, module: nn.Module = None):
        if module is None:
            module = self
        return sum(product(p.shape) for p in module.parameters() if p.requires_grad)

    # ---

    def disable_training(self):
        self.toggle_training(False)

    def enable_training(self):
        self.toggle_training(True)

    def toggle_training(self, on):
        self.toggle_autograd(on)
        assert self.training == on

    def is_training(self) -> bool:
        if self.n_autograd_enabled_parameters() > 0:
            return True
        else:
            return False
