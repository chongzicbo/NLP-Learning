# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: dropouts.py
@time: 2022/6/26 15:41
"""

import paddle
import paddle.nn as nn


class SharedDropout(nn.Layer):
    """SharedDropout"""

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def forward(self, x):
        """Forward network"""
        if self.training and self.p > 0:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= paddle.unsqueeze(mask, axis=1) if self.batch_first else mask
        return x

    @staticmethod
    def get_mask(x, p):
        """Generate the mask matrix of the dropout by the input."""
        mask = paddle.uniform(shape=x.shape, min=0, max=1) >= p
        mask = paddle.cast(mask, "float32")
        mask = mask / (1 - p)
        return mask


class IndependentDropout(nn.Layer):
    """IndependentDropout"""

    def __init__(self, p=0.5):
        super(IndependentDropout, self).__init__()
        self.p = p

    def forward(self, *items):
        """Forward network"""
        if self.training and self.p > 0:
            masks = [
                paddle.uniform(shape=x.shape[:2], min=0, max=1) >= self.p for x in items
            ]
            masks = [paddle.cast(x, "float32") for x in masks]
            total = paddle.add(*masks)
            scale = len(items) / paddle.maximum(total, paddle.ones_like(total))
            masks = [mask * scale for mask in masks]
            items = [
                item * paddle.unsqueeze(mask, axis=-1)
                for item, mask in zip(items, masks)
            ]
        return items
