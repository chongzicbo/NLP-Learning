# -*-coding:utf-8 -*-

"""
# File       : summarization.py
# Time       ：2023/4/10 15:38
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from datasets import load_dataset
billsum = load_dataset('billsum',split='ca_test')
billsum=billsum.train_test_split(test_size=0.2)

print(billsum[0])