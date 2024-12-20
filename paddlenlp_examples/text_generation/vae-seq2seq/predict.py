# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: predict.py
@time: 2022/6/21 21:10
"""

import io

import numpy as np
import paddle

from model import VAESeq2SeqInferModel
from args import parse_args
from data import create_data_loader


def infer(args):
    print(args)
    device = paddle.set_device(args.device)
    _, _, _, vocab, bos_id, eos_id, _ = create_data_loader(args)

    net = VAESeq2SeqInferModel(
        args.embed_dim, args.hidden_size, args.latent_size, len(vocab) + 2
    )

    model = paddle.Model(net)
    model.prepare()
    model.load(args.init_from_ckpt)

    infer_output = paddle.ones((args.batch_size, 1), dtype="int64") * bos_id

    space_token = " "
    line_token = "\n"
    with io.open(args.infer_output_file, "w", encoding="utf-8") as out_file:
        predict_lines = model.predict_batch(infer_output)[0]
        for line in predict_lines:
            end_id = -1
            if eos_id in line:
                end_id = np.where(line == eos_id)[0][0]
            new_line = [vocab.to_tokens(e[0]) for e in line[:end_id]]
            out_file.write(space_token.join(new_line))
            out_file.write(line_token)


if __name__ == "__main__":
    args = parse_args()
    infer(args)
