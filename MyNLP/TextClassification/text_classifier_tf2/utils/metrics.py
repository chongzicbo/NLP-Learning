#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: metrics.py
@time: 2022/10/31 9:57
"""
import sys

sys.path.append("../../text_classifier_tf2")
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np


def cal_metrics(y_true, y_pred):
    """
    指标计算
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    average = classifier_config['metrics_average']
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    each_classes = classification_report(y_true, y_pred, output_dict=True, labels=np.unique(y_pred),
                                         zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}, each_classes
