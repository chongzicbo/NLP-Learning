# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: train.json.py
@time: 2022/6/12 10:19
"""

from args import parse_args

import paddle
from paddlenlp.metrics import Perplexity

from data import create_train_loader
from model import Seq2SeqAttnModel, CrossEntropyCriterion


def do_train(args):
    device = paddle.set_device(args.device)

    # Define dataloader
    train_loader, vocab = create_train_loader(args.batch_size)
    vocab_size = len(vocab)
    pad_id = vocab[vocab.eos_token]

    model = paddle.Model(
        Seq2SeqAttnModel(
            vocab_size, args.hidden_size, args.hidden_size, args.num_layers, pad_id
        )
    )

    optimizer = paddle.optimizer.Adam(
        learning_rate=args.learning_rate, parameters=model.parameters()
    )
    ppl_metric = Perplexity()
    model.prepare(optimizer, CrossEntropyCriterion(), ppl_metric)

    print(args)
    model.fit(
        train_data=train_loader,
        epochs=args.max_epoch,
        eval_freq=1,
        save_freq=1,
        save_dir=args.model_path,
        log_freq=args.log_freq,
    )


if __name__ == "__main__":
    args = parse_args()
    do_train(args)
