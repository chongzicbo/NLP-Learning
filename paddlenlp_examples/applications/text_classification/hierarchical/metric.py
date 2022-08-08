# -*- encoding: utf-8 -*-
'''
@File    :   metric.py
@Time    :   2022/07/28 21:06:26
@Author  :   chengbo 
@Version :   1.0
@Desc    :   None
'''

# here put the import lib
import imp
import numpy as np
from sklearn.metrics import f1_score,classification_report
from paddle.metric import Metric
from paddlenlp.utils.log import logger

class MetricReport(Metric):
    """
    F1 score for hierarchical text classification task.
    """

    def __init__(self, name='MetricReport', average='micro'):
        super(MetricReport, self).__init__()
        self.average = average
        self._name = name
        self.reset()

    def reset(self):
        """
        Resets all of the metric state.
        """
        self.y_prob = None
        self.y_true = None

    def f1_score(self, y_prob):
        """
        Compute micro f1 score and macro f1 score
        """
        threshold = 0.5
        self.y_pred = y_prob > threshold
        micro_f1_score = f1_score(y_pred=self.y_pred,
                                  y_true=self.y_true,
                                  average='micro')
        macro_f1_score = f1_score(y_pred=self.y_pred,
                                  y_true=self.y_true,
                                  average='macro')
        return micro_f1_score, macro_f1_score

    def update(self, probs, labels):
        """
        Update the probability and label
        """
        if self.y_prob is not None:
            self.y_prob = np.append(self.y_prob, probs.numpy(), axis=0)
        else:
            self.y_prob = probs.numpy()
        if self.y_true is not None:
            self.y_true = np.append(self.y_true, labels.numpy(), axis=0)
        else:
            self.y_true = labels.numpy()

    def accumulate(self):
        """
        Returns micro f1 score and macro f1 score
        """
        micro_f1_score, macro_f1_score = self.f1_score(y_prob=self.y_prob)
        return micro_f1_score, macro_f1_score

    def report(self):
        """
        Returns classification report
        """
        self.y_pred = self.y_prob > 0.5
        logger.info("classification report:\n" +
                    classification_report(self.y_true, self.y_pred, digits=4))

    def name(self):
        """
        Returns metric name
        """
        return self.name