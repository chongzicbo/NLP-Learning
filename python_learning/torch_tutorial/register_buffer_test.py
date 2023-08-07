# -*-coding:utf-8 -*-

"""
# File       : register_buffer_test.py
# Time       ：2023/2/14 15:46
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch
import torch.nn as nn
from collections import OrderedDict


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.param_nn = nn.Sequential(
            OrderedDict(
                [
                    ("conv", nn.Conv2d(1, 1, 3, bias=False)),
                    ("fc", nn.Linear(1, 2, bias=False)),
                ]
            )
        )

        self.register_buffer("param_buf", torch.randn(1, 2))
        self.register_parameter("param_reg", nn.Parameter(torch.randn(1, 2)))

        self.param_attr = torch.randn(1, 2)

    def forward(self, x):
        return x


if __name__ == "__main__":
    net = Model()
    for p in net.named_parameters():
        print(p)
