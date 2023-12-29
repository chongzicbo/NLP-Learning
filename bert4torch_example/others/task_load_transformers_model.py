#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: task_load_transformers_model.py
@time: 2022/8/11 18:49
"""

# 调用transformers库中的模型来调用
# 本脚本演示功能为主，实际训练建议两者取其一
# 少量可能使用到的场景：
# 1）bert4torch的fit过程可以轻松使用对抗训练，梯度惩罚，虚拟对抗训练等功能
# 2）就是临时直接用transformers库里面的模型文件
# 3）写代码时候用于校验两者结果
import os.path

from transformers import AutoModelForSequenceClassification
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import BaseModel
from bert4torch.snippets import sequence_padding, Callback, text_segmentate, ListDataset
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

maxlen = 128
batch_size = 16
root_model_path = (
    "/mnt/e/working/huada_bgi/data/pretrained_model/huggingface/bert-base-chinese"
)
dict_path = "/mnt/e/working/huada_bgi/data/pretrained_model/huggingface/bert-base-chinese/vocab.txt"
data_dir = "/mnt/e/opensource_data/分类/情感分析/sentiment/"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filenames):
        """加载数据，并尽量划分为不超过maxlen的句子"""
        D = []
        seps, strips = "\n。！？!?；;，, ", "；;，, "
        for filename in filenames:
            with open(filename, encoding="utf-8") as f:
                for l in f:
                    text, label = l.strip().split("\t")
                    for t in text_segmentate(text, maxlen - 2, seps, strips):
                        D.append((t, int(label)))
        return D


def collate_fn(batch):
    batch_token_ids, batch_segment_ids, batch_labels = [], [], []
    for text, label in batch:
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
        batch_labels.append([label])

    batch_token_ids = torch.tensor(
        sequence_padding(batch_token_ids), dtype=torch.long, device=device
    )
    batch_segment_ids = torch.tensor(
        sequence_padding(batch_segment_ids), dtype=torch.long, device=device
    )
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return [batch_token_ids, batch_segment_ids], batch_labels.flatten()


# 加载数据集
train_dataloader = DataLoader(
    MyDataset([os.path.join(data_dir, "sentiment.train.data")]),
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
valid_dataloader = DataLoader(
    MyDataset([os.path.join(data_dir, "sentiment.valid.data")]),
    batch_size=batch_size,
    collate_fn=collate_fn,
)
test_dataloader = DataLoader(
    MyDataset([os.path.join(data_dir, "sentiment.test.data")]),
    batch_size=batch_size,
    collate_fn=collate_fn,
)


class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            root_model_path, num_labels=2
        )

    def forward(self, token_ids, segment_ids):
        output = self.bert(input_ids=token_ids, token_type_ids=segment_ids)
        return output.logits


model = Model().to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters(), lr=2e-5),
    metrics=["accuracy"],
)


# 定义评价函数
def evaluate(data):
    total, right = 0.0, 0.0
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        total += len(y_true)
        right += (y_true == y_pred).sum().item()
    return right / total


class Evaluator(Callback):
    """评估与保存"""

    def __init__(self):
        self.best_val_acc = 0.0

    def on_epoch_end(self, global_step, epoch, logs=None):
        val_acc = evaluate(valid_dataloader)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            # model.save_weights('best_model.pt')
        print(f"val_acc: {val_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n")


if __name__ == "__main__":
    evaluator = Evaluator()
    model.fit(
        train_dataloader,
        epochs=20,
        steps_per_epoch=100,
        grad_accumulation_steps=2,
        callbacks=[evaluator],
    )
else:
    model.load_weights("best_model.pt")
