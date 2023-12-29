# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: data.py
@time: 2022/6/14 17:22
"""

import re
import numpy as np
import pandas as pd
import paddle


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    encoded_inputs = tokenizer(text=example["text"], max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array(example["label"], dtype="float32")
        return input_ids, token_type_ids, label
    return input_ids, token_type_ids


def create_dataloader(
    dataset, mode="train.json", batch_size=1, batchify_fn=None, trans_fn=None
):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train.json" else False

    if mode == "train.json":
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True,
    )


def read_custom_data(filename, is_test=False):
    data = pd.read_csv(filename)
    for line in data.values:
        if is_test:
            text = line[1]
            yield {"text": clean_text(text), "label": ""}
        else:
            text, label = line[1], line[2]
            yield {"text": clean_text(text), "label": label}


def clean_text(text):
    text = text.replace("\r", "").replace("\n", "")
    text = re.sub(r"\\n\n", ".", text)
    return text


def write_test_results(filename, results, label_info):
    data = pd.read_csv(filename)
    qids = [line[0] for line in data.values]

    results_dict = {k: [] for k in label_info}
    results_dict["id"] = qids

    results = list(map(list, zip(*results)))
    for key in results_dict:
        if key != "id":
            for result in results:
                results_dict[key] = result
    df = pd.DataFrame(results_dict)
    df.to_csv("sample_test.csv", index=False)
    print("Test results saved")
