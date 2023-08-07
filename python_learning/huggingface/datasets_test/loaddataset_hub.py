# -*-coding:utf-8 -*-

"""
# File       : loaddataset_hub.py
# Time       ：2023/3/21 16:43
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from datasets import load_dataset_builder

ds_builder = load_dataset_builder("rotten_tomatoes")
print(ds_builder.info.description)
print(ds_builder.info.features)

from datasets import load_dataset

dataset = load_dataset("rotten_tomatoes", split="train")

print(dataset)

from datasets import get_dataset_split_names

print(get_dataset_split_names("rotten_tomatoes"))
dataset = load_dataset("rotten_tomatoes")
print(dataset)

print(dataset.keys())
