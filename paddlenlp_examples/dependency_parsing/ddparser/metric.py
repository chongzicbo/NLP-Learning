# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: metric.py
@time: 2022/6/26 15:39
"""

import numpy as np

import paddle
from paddle.metric import Metric


class ParserEvaluator(Metric):
    """
    UAS and LAS for dependency parser.
    UAS = number of words assigned correct head / total words
    LAS = number of words assigned correct head and relation / total words
    """

    def __init__(self, name='ParserEvaluator', eps=1e-8):
        super(ParserEvaluator, self).__init__()

        self.eps = eps
        self._name = name
        self.reset()

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def update(self, arc_preds, rel_preds, arcs, rels, mask):
        select = paddle.nonzero(mask)
        arc_mask = paddle.gather_nd(arc_preds == arcs, select)
        rel_mask = paddle.logical_and(
            paddle.gather_nd(rel_preds == rels, select), arc_mask)

        self.total += len(arc_mask)
        self.correct_arcs += np.sum(arc_mask.numpy()).item()
        self.correct_rels += np.sum(rel_mask.numpy()).item()

    def accumulate(self):
        uas = self.correct_arcs / (self.total + self.eps)
        las = self.correct_rels / (self.total + self.eps)
        return uas, las

    def name(self):
        """
        Returns metric name
        """
        return self._name
