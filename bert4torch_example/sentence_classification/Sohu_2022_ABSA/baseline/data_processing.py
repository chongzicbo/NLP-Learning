#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: data_processing.py
@time: 2022/8/12 11:30
"""

from sklearn.model_selection import StratifiedKFold
import json

with open(
    "E:/Github/Sohu2022/Sohu2022_data/nlp_data/train.txt", "r", encoding="utf-8"
) as f:
    train_data = f.readlines()
tag2_index = []
for line in train_data:
    line = json.loads(line)
    if 2 in set(line["entity"].values()):
        tag2_index.append(1)
    else:
        tag2_index.append(0)
print(sum(tag2_index))

print("样本总量：", len(train_data))
file_id = 0
kfold = StratifiedKFold(n_splits=4).split(train_data, tag2_index)
for i, (train_idx, dev_idx) in enumerate(kfold):
    train, dev = [train_data[i] for i in train_idx], [train_data[i] for i in dev_idx]
    dev_type2 = [tag2_index[i] for i in dev_idx]
    with open(
        f"E:/Github/Sohu2022/Sohu2022_data/nlp_data/dev_{file_id}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.writelines(dev)
    with open(
        f"E:/Github/Sohu2022/Sohu2022_data/nlp_data/train_{file_id}.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.writelines(train)

    print(
        f"================================样本{file_id}, train: ",
        len(train),
        "dev: ",
        len(dev),
        "dev_type2: ",
        sum(dev_type2),
    )
    print(dev[:1])
    file_id += 1
