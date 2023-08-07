from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model
import evaluate
import torch
import numpy as np


model_checkpoint = "roberta-large"
lr = 1e-3
batch_size = 16
num_epochs = 10
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
bionlp = load_dataset(path="/home/bocheng/data/corpus/bionlp2004")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # print(label)
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_bionlp = bionlp.map(tokenize_and_align_labels, batched=True)
