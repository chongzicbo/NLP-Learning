# -*-coding:utf-8 -*-

"""
# File       : hyperparam.py
# Time       ：2023/3/6 15:42
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from transformers import EvalPrediction
from transformers import (
    Trainer,
    BertTokenizer,
    TrainingArguments,
    BertForSequenceClassification,
    EarlyStoppingCallback
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["WANDB_START_METHOD"] = "thread"
# wandb.init(project="hyper_demo",
#            tags=["baseline", "hyperparam"],
#            group="group_2")


source_data_path = "../data/ChnSentiCorp_htl_all.csv"
ber_name_path = "hfl/chinese-bert-wwm-ext"

source_df = pd.read_csv(source_data_path, encoding="GBK")
source_df.dropna(how="any", axis=0, inplace=True)

# print(source_df)

source_df = shuffle(source_df)
eval_df = source_df[0:1600]
train_df = source_df[1600:-1]

le = LabelEncoder()
le.fit(source_df["label"].tolist())


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def df_2_dataset(df: DataFrame, le: LabelEncoder, text: str, label: str, tokenizer) -> Dataset:
    """Transform DataFrame into DataSet

    Args:
        df (DataFrame): Source Data
        le (LabelEncoder): LabelEncoder
        text (str): Column Name of Text
        label (str): Column Name of Label
        tokenizer ([type]): Tokenizer

    Returns:
        Dataset: DataSet
    """
    x = list(df[text])
    df["label_id"] = le.transform(df[label].tolist())
    y = list(df["label_id"])
    x_tokenized = tokenizer(x, padding=True, truncation=True, max_length=512)
    result = Dataset(x_tokenized, y)
    return result


tokenizer = BertTokenizer.from_pretrained(ber_name_path, do_lower_case=False)

train_dataset = df_2_dataset(train_df, le, "review", "label", tokenizer)
eval_dataset = df_2_dataset(eval_df, le, "review", "label", tokenizer)


def compute_metrics(results: EvalPrediction) -> Optional[dict]:
    """Calculate Metrics

    Args:
        results (EvalPrediction): Predictions & labels

    Returns:
        [type]: Dictionary of Metrics
    """
    pred, labels = results.predictions, results.label_ids
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="binary")
    precision = precision_score(y_true=labels, y_pred=pred, average="binary")
    f1 = f1_score(y_true=labels, y_pred=pred, average="binary")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train(train_dataset: Dataset,
          valid_dataset: Dataset,
          pre_model_path: str
          ) -> None:
    args = TrainingArguments(
        report_to="wandb",
        output_dir="/data/christmas.wang/project/classification_base_project/output",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        num_train_epochs=6,
        seed=0,
        load_best_model_at_end=True,
    )

    def model_init():
        return BertForSequenceClassification.from_pretrained(pre_model_path, num_labels=2)

    trainer = Trainer(
        model_init=model_init,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.3),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 8),
            "seed": trial.suggest_int("seed", 20, 40),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        }

    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        n_trials=10,
        hp_space=hp_space
    )
    print("*************************************")
    print(" Best run %s" % str(best_trial))
    print("*************************************")


if __name__ == '__main__':
    train(train_dataset, eval_dataset, ber_name_path)
