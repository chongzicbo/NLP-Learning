# -*-coding:utf-8 -*-

"""
# File       : preprocess.py
# Time       ：2023/3/21 17:08
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from transformers import AutoTokenizer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("rotten_tomatoes", split="train")

print(tokenizer(dataset[0]["text"]))

def tokenization(example):
    return tokenizer(example["text"])

dataset = dataset.map(tokenization, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])
print(dataset.format['type'])