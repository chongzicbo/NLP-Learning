# -*-coding:utf-8 -*-

"""
# File       : trainer_test.py
# Time       ：2023/3/7 17:31
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch

print(torch.cuda.is_available())

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

from datasets import load_dataset

dataset = load_dataset("rotten_tomatoes")
# print(dataset["train"]["text"])


def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])


dataset = dataset.map(tokenize_dataset, batched=True)

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
