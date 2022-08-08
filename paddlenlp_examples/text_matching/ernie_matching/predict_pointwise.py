from functools import partial
import argparse
import sys
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad

from data import create_dataloader, read_text_pair
from data import convert_pointwise_example as convert_example
from model import PointwiseMatching

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True, help="The full path of input file")
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=64, type=int,
                    help="The maximum total input sequence length after tokenization. "
                         "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu",
                    help="Select which device to train.json model, defaults to gpu.")
args = parser.parse_args()


def predict(model, data_loader):
    batch_probs = []
    model.eval()
    with paddle.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data
            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)
            batch_prob = model(input_ids=input_ids, token_type_ids=token_type_ids).numpy()
            batch_probs.append(batch_prob)
        batch_probs = np.concatenate(batch_probs, axis=0)
        return batch_probs


if __name__ == '__main__':
    paddle.set_device(args.device)
    pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(
        'ernie-gram-zh')
    tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained(
        'ernie-gram-zh')

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_test=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
    ): [data for data in fn(samples)]
    valid_ds = load_dataset(read_text_pair, data_files=args.input_file, lazy=False)
    valid_data_loader = create_dataloader(valid_ds, mode="predict", batch_size=args.batch_size, batchify_fn=batchify_fn,
                                          transfn=trans_func)
    model = PointwiseMatching(pretrained_model)
    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError("Please set --params_path with correct pretrained model file")

    y_probs = predict(model, valid_data_loader)
    y_preds = np.argmax(y_probs, axis=1)
    valid_ds = load_dataset(read_text_pair, data_files=args.input_file, lazy=False)
    for idx, y_pred in enumerate(y_preds):
        text_pair = valid_ds[idx]
        text_pair["pred_label"] = y_pred
        print(text_pair)
