# -*-coding:utf-8 -*-

"""
# File       : example-01.py
# Time       ：2023/3/21 16:31
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import datasets

dataset=datasets.load_from_disk('/tmp/NLP-Learning/LLM/ChatGLM-Tuning/data/alpaca')
for x in dataset:
    print(x['input_ids'])
    break
print(dataset['input_ids'][0:2])