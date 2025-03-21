# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: predict.py
@time: 2022/6/18 14:51
"""

import argparse
import os

import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.data import Tuple, Pad

from utils import convert_example

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, required=True, default="checkpoints/model_900/model_state.pdparams",
                    help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", type=int, default=128,
                    help="The maximum total input sequence length after tokenization. "
                         "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu', 'xpu', 'npu'], default="gpu",
                    help="Select which device to train.json model, defaults to gpu.")
args = parser.parse_args()


# yapf: enable


def predict(model, data, tokenizer, label_map, batch_size=1):
    """
    Predicts the data labels.
    Args:
        model (obj:`paddle.nn.Layer`): A model to classify texts.
        data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
            A Example object contains `text`(word_ids) and `seq_len`(sequence length).
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer`
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
        batch_size(obj:`int`, defaults to 1): The number of batch.
    Returns:
        results(obj:`dict`): All the predictions labels.
    """
    examples = []
    for text in data:
        example = {"text": text}
        input_ids, token_type_ids = convert_example(
            example, tokenizer, max_seq_length=args.max_seq_length, is_test=True
        )
        examples.append((input_ids, token_type_ids))

    # Seperates data into some batches.
    batches = [
        examples[idx : idx + batch_size] for idx in range(0, len(examples), batch_size)
    ]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
    ): fn(samples)

    results = []
    model.eval()
    for batch in batches:
        input_ids, token_type_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        token_type_ids = paddle.to_tensor(token_type_ids)
        logits = model(input_ids, token_type_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results


if __name__ == "__main__":
    paddle.set_device(args.device)

    data = [
        "这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般",
        "怀着十分激动的心情放映，可是看着看着发现，在放映完毕后，出现一集米老鼠的动画片",
        "作为老的四星酒店，房间依然很整洁，相当不错。机场接机服务很好，可以在车上办理入住手续，节省时间。",
    ]
    label_map = {0: "negative", 1: "positive"}

    model = ppnlp.transformers.ErnieForSequenceClassification.from_pretrained(
        "ernie-tiny", num_classes=len(label_map)
    )
    tokenizer = ppnlp.transformers.ErnieTinyTokenizer.from_pretrained("ernie-tiny")

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    results = predict(model, data, tokenizer, label_map, batch_size=args.batch_size)
    for idx, text in enumerate(data):
        print("Data: {} \t Lable: {}".format(text, results[idx]))
