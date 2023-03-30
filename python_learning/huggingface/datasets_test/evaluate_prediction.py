# -*-coding:utf-8 -*-

"""
# File       : evaluate_prediction.py
# Time       ：2023/3/21 17:28
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from datasets import list_metrics

metrics_list=list_metrics()
print(metrics_list)

from datasets import load_metric
metric=load_metric('glue','mrpc')

print(metric.inputs_description)

