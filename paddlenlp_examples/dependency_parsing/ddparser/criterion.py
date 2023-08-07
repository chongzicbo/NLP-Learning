# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: criterion.py
@time: 2022/6/26 15:39
"""

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from utils import index_sample


class ParserCriterion(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(ParserCriterion, self).__init__(*args, **kwargs)

    def __call__(self, s_arc, s_rel, arcs, rels, mask):
        arcs = paddle.masked_select(arcs, mask)
        rels = paddle.masked_select(rels, mask)

        select = paddle.nonzero(mask)
        s_arc = paddle.gather_nd(s_arc, select)
        s_rel = paddle.gather_nd(s_rel, select)

        s_rel = index_sample(s_rel, paddle.unsqueeze(arcs, axis=1))

        arc_cost = F.cross_entropy(s_arc, arcs)
        rel_cost = F.cross_entropy(s_rel, rels)

        avg_cost = paddle.mean(arc_cost + rel_cost)
        return avg_cost
