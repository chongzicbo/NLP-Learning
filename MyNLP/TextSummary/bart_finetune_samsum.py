# -*-coding:utf-8 -*-

"""
# File       : bart_finetune_samsum.py
# Time       ：2023/2/15 15:34
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import wandb
wandb.login()
model_checkpoint = "facebook/bart-base"
from datasets import load_dataset, load_metric

raw_datasets = load_dataset("samsum")
metric = load_metric("rouge")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 512
max_target_length = 128

def preprocess_function(examples):
    inputs = [doc for doc in examples["dialogue"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)