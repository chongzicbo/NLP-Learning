# -*-coding:gbk -*-

"""
# File       : prompt.py
# Time       ：2023/3/6 16:40
# Author     ：chengbo
# version    ：python 3.8
# Description：https://zhuanlan.zhihu.com/p/424888379
"""
import os
import pandas as pd
from tqdm import tqdm
import torch
import jieba
import codecs
import json
from sklearn.utils import shuffle
from typing import List, Dict
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from transformers import (
    Trainer,
    BertTokenizer,
    TrainingArguments,
    BertForMaskedLM,
    EarlyStoppingCallback
)

torch.cuda.empty_cache()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ber_name_path = "hfl/chinese-bert-wwm-ext"

source_data_path = "../data"

label_2_id = {"好": 1, "差": 0}
id_2_label = {"1": "好", "0": "差"}

data_dict = {
    "train": os.path.join(source_data_path, "hotel_review_few_shot_train.csv"),
    "test": os.path.join(source_data_path, "hotel_review_few_shot_test.csv")
}

z_or_f = "f"

tokenizer = BertTokenizer.from_pretrained(ber_name_path)
model = BertForMaskedLM.from_pretrained(ber_name_path)


def compute_metrics(pred):
    labels = pred.label_ids[:, 3]
    preds = pred.predictions[:, 3].argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


df_train = pd.read_csv(data_dict["train"], encoding="utf-8")
df_test = pd.read_csv(data_dict["test"], encoding="utf-8")

text = []
label = []
punc = "＂!＃＄％＆＇?（）()/＊＋，－／：；,.＜＝＞＠［＼］\"＾＿｀｛｜｝～?????　、〃〈〉《》「」『』【】〔〕〖〗?????〝〞????–—‘’?“”??…?﹏﹑﹔·！？?。"
"""
1.循环遍历每一对（text, label）
2.对text进行结巴分词
3.遍历分词后的词数组
4.在每个词语的间隙，插入mask作为新text，同时在词语的间隙插入标签作为新label
"""
for index, row in tqdm(iterable=df_train.iterrows(), total=df_train.shape[0]):
    sentence = row["review"]
    words = jieba.lcut(sentence)
    for i in range(len(words)):
        sentence_train = "".join(words[:i]) + "，酒店[MASK]，" + "".join(words[i:])
        sentence_test = "".join(words[:i]) + "，酒店" + id_2_label[str(row["label"])] + "，" + "".join(words[i:])
        text.append(sentence_train)
        label.append(sentence_test)
text, label = shuffle(text, label)
print(len(text))
print(text[:3])
print(label[:3])


def dataset_builder(x: List[str], y: List[str], tokenizer: BertTokenizer, max_len: int) -> Dataset:
    data_dict = {'text': x, 'label_text': y}
    result = Dataset.from_dict(data_dict)

    def preprocess_function(examples):
        text_token = tokenizer(examples['text'], padding=True, truncation=True, max_length=max_len)
        text_token['labels'] = np.array(
            tokenizer(examples['label_text'], padding=True, truncation=True, max_length=max_len)["input_ids"])
        return text_token

    result = result.map(preprocess_function, batched=True)
    return result


eval_dataset = dataset_builder(text[:130], label[:130], tokenizer, 512)
train_dataset = dataset_builder(text[130:], label[130:], tokenizer, 512)

args = TrainingArguments(
    output_dir="../data/christmas.wang/project/classification_base_project/output",
    evaluation_strategy="steps",
    eval_steps=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-5,
    num_train_epochs=6,
    seed=20,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer.train()

pred = []
true = []
external_words = []
df_test.dropna(how="any", axis=0, inplace=True)
for index, row in tqdm(iterable=df_test.iterrows(), total=df_test.shape[0]):
    text = "酒店[MASK]，" + row["review"]
    tokenized_text = tokenizer.tokenize(text)
    if len(tokenized_text) > 512:
        tokenized_text = tokenized_text[:512]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
    segments_tensors = torch.tensor([segments_ids]).to('cuda')

    masked_index = tokenized_text.index('[MASK]')

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
    predicted_index = torch.argmax(predictions[0][0][masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    # print(predicted_token+str(row["label"]))
    if predicted_token not in ["差", "好"]:
        external_words.append(predicted_token)
        predicted_token = "差"
    y_pred = label_2_id[predicted_token]
    pred.append(y_pred)
    true.append(row["label"])
precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary')
acc = accuracy_score(true, pred)
print({'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall})
print(external_words)
print(len(external_words))

good_words = ["美", "舒", "服", "豪", "华", "丽", "亮", "错", "大", "宜", "明"]


def get_label(words: List[str]) -> int:
    for key, val in label_2_id.items():
        if key in words:
            return val
    for word in words:
        if word in good_words:
            return 0
    return 1


pred = []
true = []

for index, row in tqdm(iterable=df_test.iterrows(), total=df_test.shape[0]):
    text = "酒店[MASK]，" + row["review"]
    tokenized_text = tokenizer.tokenize(text)
    if len(tokenized_text) > 512:
        tokenized_text = tokenized_text[:512]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
    segments_tensors = torch.tensor([segments_ids]).to('cuda')

    masked_index = tokenized_text.index('[MASK]')

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)
    top_k = torch.topk(predictions[0][0][masked_index].flatten(), 5).indices.tolist()
    words = []
    for word in top_k:
        predicted_token = tokenizer.convert_ids_to_tokens([word])[0]
        words.append(predicted_token)
    pred.append(get_label(words))
    true.append(row["label"])
precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='binary')
acc = accuracy_score(true, pred)
print({'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall})

