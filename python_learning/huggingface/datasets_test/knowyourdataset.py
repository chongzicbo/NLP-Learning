# -*-coding:utf-8 -*-

"""
# File       : knowyourdataset.py
# Time       ：2023/3/21 16:59
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from datasets import load_dataset

dataset = load_dataset('rotten_tomatoes', split='train')
print(dataset[0])
print(dataset['text'][0:2])
print(dataset[0]['text'])
print(dataset[0:3])

iterable_dataset = load_dataset('food101', split='train', streaming=True)
for example in iterable_dataset:
    print(example)
    break

print(next(iter(iterable_dataset)))

print(list(iterable_dataset.take(3)))
