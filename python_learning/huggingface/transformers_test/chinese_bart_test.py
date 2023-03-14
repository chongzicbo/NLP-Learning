# -*-coding:utf-8 -*-

"""
# File       : chinese_bart_test.py
# Time       ：2023/2/17 10:59
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""

from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline

model_path = "/home/bocheng/data/pretrain_model/chinese/bart-base-chinese"
tokenizer = BertTokenizer.from_pretrained(
    model_path)  # Please use BertTokenizer for the model vocabulary. DO NOT use original BartTokenizer.
model = BartForConditionalGeneration.from_pretrained(model_path)
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)
print(text2text_generator("武汉是[MASK]的省会", max_length=50, do_sample=False))
