# -*-coding:utf-8 -*-

"""
# File       : unilm_test.py
# Time       ：2023/2/6 9:49
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/unilm-base-cased")
print(tokenizer.tokenize("Summarize"))
print(tokenizer.encode("Summarize:"))
