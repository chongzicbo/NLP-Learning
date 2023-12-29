# -*-coding:utf-8 -*-

"""
# File       : 提取任意一层的输出.py
# Time       ：2023/4/24 10:47
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""

## 1.方法一

import torch
import torch.nn as nn
import torchvision.models

model = nn.Sequential(
    nn.Conv2d(3, 9, 1, 1, 0, bias=False),
    nn.BatchNorm2d(9),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1)),
)

x = torch.rand([2, 3, 224, 224])
for i in range(len(model)):
    x = model[i](x)
    if i == 2:
        ReLu_out = x
print(ReLu_out.shape)
print(x.shape)

## 2.方法2
from collections import OrderedDict

import torch
from torch import nn


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


m = torchvision.models.resnet18(pretrained=True)
print(m)
new_m = IntermediateLayerGetter(m, {"layer1": "feat1", "layer3": "feat2"})

out = new_m(torch.rand(1, 3, 224, 224))
# print(out)
print([(k, v.shape) for k, v in out.items()])
