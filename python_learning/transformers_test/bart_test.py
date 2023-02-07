# -*-coding:utf-8 -*-

"""
# File       : bart_test.py
# Time       ：2023/1/31 16:40
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from transformers import BartTokenizer

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
print(tokenizer.encode("Summarize:", add_special_tokens=False))
print(tokenizer.num_special_tokens_to_add())
ids = [0, 38182, 3916, 2072, 35, 1437, 10127, 5219, 35, 38, 17241, 15269, 4, 1832, 47, 236, 103, 116, 646, 3388, 510,
       742, 6509, 35, 9136, 328, 646, 3388, 510, 742, 10641, 35, 38, 581, 836, 47, 3859, 48433, 2]
print(tokenizer.decode(ids))
