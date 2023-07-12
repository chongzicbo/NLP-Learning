# -*-coding:utf-8 -*-

"""
# File       : gpt2_test.py
# Time       ：2023/4/28 11:51
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader
model_name_or_path = "uer/gpt2-chinese-cluecorpussmall"
tokenizer=AutoTokenizer.from_pretrained(model_name_or_path)
text="你今天还好吗"
tokenizerd_text=tokenizer(text, add_special_tokens=False, truncation=True)
print(tokenizerd_text)

tokenizerd_text=tokenizer(text, truncation=True)
print(tokenizerd_text)