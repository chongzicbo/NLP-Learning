# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: data.py
@time: 2022/6/23 12:27
"""

import paddle
import numpy as np

from paddlenlp.datasets import MapDataset


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


def read_text_pair(data_path, is_test=False):
    """Reads data."""
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = line.rstrip().split("\t")
            if is_test == False:
                if len(data) != 3:
                    continue
                yield {"query1": data[0], "query2": data[1], "label": data[2]}
            else:
                if len(data) != 2:
                    continue
                yield {"query1": data[0], "query2": data[1]}


def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    query, title = example["query1"], example["query2"]

    encoded_inputs = tokenizer(text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids
