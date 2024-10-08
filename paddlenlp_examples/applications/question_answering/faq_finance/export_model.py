#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: export_model.py
@time: 2022/7/28 17:16
"""

import argparse
import os
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import AutoTokenizer, AutoModel
from paddlenlp.data import Stack, Tuple, Pad

from model import SimCSE

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--params_path", type=str, required=True,
                    default='./checkpoint/model_50/model_state.pdparams',
                    help="The path to model parameters to be loaded.")
parser.add_argument("--output_path", type=str, default='./output',
                    help="The path of model parameter in static graph to be saved.")
args = parser.parse_args()
# yapf: enable

if __name__ == "__main__":
    output_emb_size = 256

    pretrained_model = AutoModel.from_pretrained("ernie-3.0-medium-zh")

    tokenizer = AutoTokenizer.from_pretrained("ernie-3.0-medium-zh")
    model = SimCSE(pretrained_model, output_emb_size=output_emb_size)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)

    model.eval()
    # Convert to static graph with specific input description
    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # input_ids
            paddle.static.InputSpec(shape=[None, None], dtype="int64"),  # segment_ids
        ],
    )
    # Save in static graph model.
    save_path = os.path.join(args.output_path, "inference")
    paddle.jit.save(model, save_path)
