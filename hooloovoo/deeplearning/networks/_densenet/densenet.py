import re
from collections import OrderedDict
from typing import NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils import model_zoo

from hooloovoo.deeplearning.networks.controls import Controls
from hooloovoo.utils.tree import Tree


class DenseNetConfig(NamedTuple):
    r"""
    1) input_downscale (float): with which factor to downsample the input, to speed up training
    2) num_init_features (int): the number of filters to learn in the first convolution layer
    3) growth_rate (int): how many filters to add each layer (`k` in paper)
    4) block_config (list of 4 ints): how many layers in each pooling block
    5) bn_size (int): multiplicative factor for number of bottle neck layers
      (i.e. bn_size * k features in the bottleneck layer)
    6) drop_rate (float): dropout rate after each dense layer
    7) n_planes (int): how many feature channels to use during upsampling.
    8) n_classes (int): number of output classes
    """
    input_downscale: float
    num_init_features: int
    growth_rate: int
    block_config: Tuple[int, int, int, int]
    bn_size: int
    drop_rate: int
    n_planes: int
    n_classes: int

    @staticmethod
    def default():
        return PublishedModel.M121()


class PublishedModel:
    r"""Densenet model with structure that exactly matches the paper."""

    @staticmethod
    def _defaults():
        url = lambda txt: 'https://download.pytorch.org/models/densenet' + txt
        cfg = lambda *args: DenseNetConfig(*args)._asdict()
        return {
            'M121': (cfg(1, 64, 32, (6, 12, 24, 16), 4, 0, 16, 2), url("121-a639ec97.pth")),
            'M161': (cfg(1, 96, 48, (6, 12, 36, 24), 4, 0, 16, 2), url("161-8d451a50.pth")),
            'M169': (cfg(1, 64, 32, (6, 12, 32, 32), 4, 0, 16, 2), url("201-c1103571.pth")),
            'M201': (cfg(1, 64, 32, (6, 12, 48, 32), 4, 0, 16, 2), url("169-b2777c0a.pth")),
        }

    _free_params = {"input_downscale", "n_planes", "n_classes"}

    def __init__(self, name, **kwargs):
        cfg, url = PublishedModel._defaults()[name]
        for kw in kwargs:
            if kw in PublishedModel._free_params:
                cfg[kw] = kwargs[kw]
            else:
                raise ValueError(f"Cannot change param {kw} as it would not correspond to a published model")

        self.name = name
        self.url = url
        self.cfg = DenseNetConfig(**cfg)


# noinspection PyProtectedMember
for name_ in PublishedModel._defaults():
    setattr(PublishedModel, name_, lambda **kwargs: PublishedModel(name_, **kwargs))


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = f.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _UpBlock(nn.Module):
    def __init__(self, bypass_in_planes, in_planes, out_planes=None):
        super(_UpBlock, self).__init__()

        if out_planes is None:
            out_planes = in_planes

        self.process_bypass = nn.Sequential(OrderedDict([
            ("norm", nn.BatchNorm2d(bypass_in_planes)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv", nn.Conv2d(bypass_in_planes, in_planes, kernel_size=1))
        ]))

        self.process_join = nn.Sequential(OrderedDict([
            ("norm", nn.BatchNorm2d(in_planes)),
            ("relu", nn.ReLU(inplace=True)),
            ("conv", nn.Conv2d(in_planes, out_planes, kernel_size=1))
        ]))

    def forward(self, low_res_features, bypass):
        bypass_size = bypass.shape[-2:]
        # print(" bypass shape %s" % str(bypass.shape))
        # print("feature shape %s" % str(low_res_features.shape))
        upsampled_features = f.interpolate(low_res_features, size=bypass_size,
                                           mode="bilinear", align_corners=False)
        bypass_features = self.process_bypass(bypass)
        return self.process_join(upsampled_features + bypass_features)


class DenseNet(Controls):
    r"""Decapitated Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    """
    def __init__(self, config: DenseNetConfig = DenseNetConfig.default()):
        super(DenseNet, self).__init__()
        self.config = config

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, config.num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(config.num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = config.num_init_features
        for i, num_layers in enumerate(config.block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=config.bn_size, growth_rate=config.growth_rate, drop_rate=config.drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * config.growth_rate
            block.num_features = num_features
            if i != len(config.block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # head: upsampling layers
        n_channels_last_feature, _ = self.back_last_feature_layer()
        self.head = nn.ModuleDict({
            "uturn": nn.Sequential(OrderedDict([
                ("norm", nn.BatchNorm2d(n_channels_last_feature)),
                ("relu", nn.ReLU(inplace=True)),
                ("avgp", nn.AvgPool2d(kernel_size=2, stride=2)),
                ("conv", nn.Conv2d(n_channels_last_feature, self.config.n_planes, kernel_size=1, padding=0)),
            ])),
            "interpolate": nn.Sequential(*[
                _UpBlock(n_channels, self.config.n_planes) for n_channels, _ in self.back_feature_layers()
            ]),
            "segment": nn.Sequential(OrderedDict([
                ("last_interpolation", _UpBlock(3, self.config.n_planes, self.config.n_classes)),
                ("prelu", nn.PReLU()),
            ]))
        })

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.config.input_downscale != 1:
            downsampled_x = f.interpolate(x, scale_factor=self.config.input_downscale,
                                          mode="bilinear", align_corners=False)
        else:
            downsampled_x = x

        backbone_features = []
        backbone_feature = downsampled_x
        for c, layer in self.back_feature_layers():
            backbone_feature = layer(backbone_feature)
            backbone_features.append(backbone_feature)

        head_feature = self.head.uturn(backbone_feature)
        for feature_layer, interpolate in zip(reversed(backbone_features), reversed(self.head.interpolate)):
            head_feature = interpolate(head_feature, feature_layer)

        segmentation = self.head.segment.last_interpolation(head_feature, downsampled_x)
        segmentation = self.head.segment.prelu(segmentation)

        return segmentation

    @staticmethod
    def published_structure(structure: PublishedModel, pretrained: bool):
        r"""Densenet model with structure that exactly matches the paper.

        Args:
            structure (PublishedModel): Which model structure to use (121, 161, 169 or 201)
            pretrained (bool): If True, returns a model pre-trained on ImageNet
        """
        config: DenseNetConfig = structure.cfg
        dnetb = DenseNet(config)

        if pretrained:
            # '.'s are no longer allowed in module names, but pervious _DenseLayer
            # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
            # They are also in the checkpoints in model_urls. This pattern is used
            # to find such keys.
            dot_pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$'
            )
            # The network is decapitated, this pattern identifies the head layer.
            head_pattern = re.compile(r'^(classifier|features.norm5).*')

            state_dict = model_zoo.load_url(structure.url)
            for key in list(state_dict.keys()):
                # replace dots in denselayers
                res = dot_pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
                # remove head weights
                res = head_pattern.match(key)
                if res:
                    del state_dict[key]

            # add the new head (the upsampling layers) to the state dict
            untrained_part = {k: v for k, v in dnetb.state_dict().items() if k.startswith("head")}
            state_dict.update(untrained_part)
            dnetb.load_state_dict(state_dict, strict=True)
        return dnetb

    @staticmethod
    def from_settings(settings: Tree):
        if "published" in settings.backbone:
            structure = PublishedModel(settings.backbone.published, **settings.head)
            return DenseNet.published_structure(structure, pretrained=settings.backbone.pretrained)
        else:
            structure = DenseNetConfig(**settings.backbone, **settings.head)
            return DenseNet(structure)

    def back_feature_layers(self):
        features = self.features[0:5]
        out_channels = features[-1].num_features
        yield out_channels, features
        n_blocks = (len(self.features) - 4) // 2 + 1
        for i in range(1, n_blocks):
            features = self.features[4 + 2*i-1:4 + 2*i+1]
            out_channels = features[-1].num_features
            yield out_channels, features

    def back_last_feature_layer(self):
        features = self.features[-2:]
        out_channels = features[-1].num_features
        return out_channels, features

    def named_non_pretrainable_children(self):
        return self.head.children()


if __name__ == "__main__":
    model = DenseNet.published_structure(pretrained=True, structure=PublishedModel.M121)
    for flayer in model.back_feature_layers():
        print("-------------------------------")
        print(flayer)
