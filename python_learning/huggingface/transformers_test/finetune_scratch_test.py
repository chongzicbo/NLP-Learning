# -*-coding:utf-8 -*-

"""
# File       : finetune_scratch_test.py
# Time       ：2023/3/10 17:05
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import time

from datasets import load_dataset

dataset = load_dataset("yelp_review_full")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=5
)

from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=16)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=16)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))
model.train()

for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

import evaluate

metric = evaluate.load("accuracy")
model.eval()

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

text = "Text from the news article"
inputs = tokenizer(text, padding="max_length", truncation=True)
input_ids, token_type_ids, attention_mask = (
    torch.unsqueeze(torch.tensor(inputs["input_ids"]), 0).to(device),
    torch.unsqueeze(torch.tensor(inputs["token_type_ids"]), 0).to(device),
    torch.unsqueeze(torch.tensor(inputs["attention_mask"]), 0).to(device),
)
start_time = time.time()
model(input_ids, token_type_ids, attention_mask)
end_time = time.time()
print(end_time - start_time)

torch.onnx.export(
    model,
    (input_ids, token_type_ids, attention_mask),
    "/home/bocheng/data/model_saved/test.onnx",
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=["input_ids", "token_type_ids", "attention_mask"],
    output_names=["output"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "token_type_ids": {0: "batch_size", 1: "sequence"},
        "attention_mask": {0: "batch_size", 1: "sequence"},  # variable length axes
        "output": {0: "batch_size"},
    },
)
