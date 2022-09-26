from abc import ABC

import torch.nn as nn

from hooloovoo.utils.functional import product


class Inspection(nn.Module):
    """
    The term 'pretrainable' is defined with respect to some parameter/module.
    It means that pre-trained values are available for that parameter or module.
    """

    # not abstract because you don't need this method if you want to train from scratch.
    def named_non_pretrainable_children(self):
        """
        Raises NotImplementedError  by default
        :return: all child modules of this module for which no pre-trained values can be loaded.
        """
        raise NotImplementedError

    def named_pretrainable_children(self):
        non_pretrainable_children_names = set(name for name, _ in self.named_non_pretrainable_children())
        return ((name, child) for name, child in self.named_children() if name not in non_pretrainable_children_names)

    # ---

    def non_pretrainable_children(self):
        for name, child in self.named_non_pretrainable_children():
            yield child

    def pretrainable_children(self):
        for name, child in self.named_pretrainable_children():
            yield child

    # ---

    def non_pretrainable_parameters(self):
        return (parameter for child in self.non_pretrainable_children() for parameter in child.parameters())

    def pretrainable_parameters(self):
        return (parameter for child in self.pretrainable_children() for parameter in child.parameters())

    # ---

    def named_non_pretrainable_parameters(self):
        non_pretrainable_parameters = list(self.non_pretrainable_parameters())
        for name, parameter in self.named_parameters():
            for npp in non_pretrainable_parameters:
                if parameter.shape == npp.shape:
                    if parameter is npp:
                        yield name, parameter

    def named_pretrainable_parameters(self):
        pretrainable_parameters = self.pretrainable_parameters()
        for name, parameter in self.named_parameters():
            for pp in pretrainable_parameters:
                if parameter.shape == pp.shape:
                    if parameter is pp:
                        yield name, parameter

    # ---

    def n_parameters(self):
        return sum(product(p.shape) for p in self.parameters())
