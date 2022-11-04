#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: trains.py
@time: 2022/11/3 16:50
"""
import json
import os.path

from tqdm import tqdm
import codecs

data_dir = "/mnt/e/opensource_data/信息抽取/关系抽取/BD_Knowledge_Extraction/"
schemas_path = os.path.join(data_dir, "all_50_schemas")
all_50_schemas = set()

with open(schemas_path) as f:
    for l in tqdm(f):
        a = json.loads(l)
        all_50_schemas.add(a['predicate'])

id2predicate = {i + 1: j for i, j in enumerate(all_50_schemas)}  # 0表示终止类别
predicate2id = {j: i for i, j in id2predicate.items()}

with codecs.open('all_50_schemas_me.json', 'w', encoding='utf-8') as f:
    json.dump([id2predicate, predicate2id], f, indent=4, ensure_ascii=False)

chars = {}
min_count = 2

train_data = []
train_data_path = os.path.join(data_dir, 'train_data.json')
dev_data_path = os.path.join(data_dir, 'dev_data.json')
with open(train_data_path) as f:
    for l in tqdm(f):
        a = json.loads(l)
        train_data.append(
            {
                'text': a['text'],
                'spo_list': [(i['subject'], i['predicate'], i['object']) for i in a['spo_list']]
            }
        )
        for c in a['text']:
            chars[c] = chars.get(c, 0) + 1

with codecs.open('train_data_me.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)

dev_data = []

with open(dev_data_path) as f:
    for l in tqdm(f):
        a = json.loads(l)
        dev_data.append(
            {
                'text': a['text'],
                'spo_list': [(i['subject'], i['predicate'], i['object']) for i in a['spo_list']]
            }
        )
        for c in a['text']:
            chars[c] = chars.get(c, 0) + 1

with codecs.open('dev_data_me.json', 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)

with codecs.open('all_chars_me.json', 'w', encoding='utf-8') as f:
    chars = {i: j for i, j in chars.items() if j >= min_count}
    id2char = {i + 2: j for i, j in enumerate(chars)}  # padding: 0, unk: 1
    char2id = {j: i for i, j in id2char.items()}
    json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)
