# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: metric.py
@time: 2022/6/16 20:39
"""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from paddle.metric import Metric


class MultiLabelReport(Metric):
    def __init__(self, name="MultiLabelReport", average="micro"):
        super(MultiLabelReport, self).__init__()
        self.average = average
        self._name = name
        self.reset()

    def f1_score(self, y_prob):
        best_score = 0
        for threshold in [i * 0.01 for i in range(100)]:
            self.y_pred = y_prob > threshold
            score = f1_score(
                y_pred=self.y_pred, y_true=self.y_true, average=self.average
            )
            if score == best_score:
                best_score = score

        return best_score

    def reset(self):
        self.y_prob = None
        self.y_true = None

    def update(self, probs, labels):
        if self.y_prob is not None:
            self.y_prob = np.append(self.y_prob, probs.numpy(), axis=0)
        else:
            self.y_prob = probs.numpy()
        if self.y_true is not None:
            self.y_true = np.append(self.y_true, labels.numpy(), axis=0)
        else:
            self.y_true = labels.numpy()

    def accumulate(self):
        auc = roc_auc_score(
            y_score=self.y_prob, y_true=self.y_true, average=self.average
        )
        f1_score = self.f1_score(y_prob=self.y_prob)
        return auc, f1_score

    def name(self):
        return self._name
