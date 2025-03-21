# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: train.json.py
@time: 2022/6/23 20:34
"""

import argparse
from collections import defaultdict
import os
import random
from functools import partial
import time
import numpy as np
import paddle
import paddle.nn as nn
from paddle.metric import Accuracy
from paddle.optimizer import AdamW
from modeling import ErnieDocForSequenceClassification
from paddlenlp.transformers import ErnieDocTokenizer, ErnieDocBPETokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from paddlenlp.datasets import load_dataset
from paddlenlp.ops.optimizer.adamwdl import layerwise_lr_decay
from data import (
    ClassifierIterator,
    ImdbTextPreprocessor,
    HYPTextPreprocessor,
    to_json_file,
)
from metrics import F1

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--model_name_or_path", type=str, default="ernie-doc-base-zh",
                    help="Pretraining model name or path")
parser.add_argument("--max_seq_length", type=int, default=512,
                    help="The maximum total input sequence length after SentencePiece tokenization.")
parser.add_argument("--learning_rate", type=float, default=1.5e-4, help="Learning rate used to train.json.")
parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
parser.add_argument("--logging_steps", type=int, default=1, help="Log every X updates steps.")
parser.add_argument("--output_dir", type=str, default='checkpoints/', help="Directory to save model checkpoint")
parser.add_argument("--epochs", type=int, default=3, help="Number of epoches for training.")
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"],
                    help="Select cpu, gpu devices to train.json model.")
parser.add_argument("--seed", type=int, default=1, help="Random seed for initialization.")
parser.add_argument("--memory_length", type=int, default=128, help="Length of the retained previous heads.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Linear warmup proption over the training process.")
parser.add_argument("--dataset", default="iflytek", choices=["imdb", "iflytek", "thucnews", "hyp"], type=str,
                    help="The training dataset")
parser.add_argument("--layerwise_decay", default=1.0, type=float, help="Layerwise decay ratio")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
parser.add_argument("--test_results_file", default="./test_restuls.json", type=str,
                    help="The file path you would like to save the model ouputs on test dataset.")

args = parser.parse_args()
# yapf: enable

DATASET_INFO = {
    "imdb": (ErnieDocBPETokenizer, "test", "test", ImdbTextPreprocessor(), Accuracy()),
    "hyp": (ErnieDocBPETokenizer, "dev", "test", HYPTextPreprocessor(), F1()),
    "iflytek": (ErnieDocTokenizer, "dev", "test", None, Accuracy()),
    "thucnews": (ErnieDocTokenizer, "dev", "test", None, Accuracy()),
}


def set_seed(args):
    # Use the same data seed(for data shuffle) for all procs to guarantee data
    # consistency after sharding.
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Maybe different op seeds(for dropout) for different procs is better. By:
    # `paddle.seed(args.seed + paddle.distributed.get_rank())`
    paddle.seed(args.seed)


def init_memory(batch_size, memory_length, d_model, n_layers):
    return paddle.zeros([n_layers, batch_size, memory_length, d_model], dtype="float32")


@paddle.no_grad()
def evaluate(model, metric, data_loader, memories):
    model.eval()
    losses = []
    # copy the memory
    tic_train = time.time()
    eval_logging_step = 500

    probs_dict = defaultdict(list)
    label_dict = dict()
    global_steps = 0
    for step, batch in enumerate(data_loader, start=1):
        (
            input_ids,
            position_ids,
            token_type_ids,
            attn_mask,
            labels,
            qids,
            gather_idxs,
            need_cal_loss,
        ) = batch
        logits, memories = model(
            input_ids, memories, token_type_ids, position_ids, attn_mask
        )
        logits, labels, qids = list(
            map(lambda x: paddle.gather(x, gather_idxs), [logits, labels, qids])
        )
        # Need to collect probs for each qid, so use softmax_with_cross_entropy
        loss, probs = nn.functional.softmax_with_cross_entropy(
            logits, labels, return_softmax=True
        )
        losses.append(loss.mean().numpy())
        # Shape: [B, NUM_LABELS]
        np_probs = probs.numpy()
        # Shape: [B, 1]
        np_qids = qids.numpy()
        np_labels = labels.numpy().flatten()
        for i, qid in enumerate(np_qids.flatten()):
            probs_dict[qid].append(np_probs[i])
            label_dict[qid] = np_labels[i]  # Same qid share same label.

        if step % eval_logging_step == 0:
            logger.info(
                "Step %d: loss:  %.5f, speed: %.5f steps/s"
                % (step, np.mean(losses), eval_logging_step / (time.time() - tic_train))
            )
            tic_train = time.time()

    # Collect predicted labels
    preds = []
    labels = []
    for qid, probs in probs_dict.items():
        mean_prob = np.mean(np.array(probs), axis=0)
        preds.append(mean_prob)
        labels.append(label_dict[qid])

    preds = paddle.to_tensor(np.array(preds, dtype="float32"))
    labels = paddle.to_tensor(np.array(labels, dtype="int64"))

    metric.update(metric.compute(preds, labels))
    acc_or_f1 = metric.accumulate()
    logger.info(
        "Eval loss: %.5f, %s: %.5f"
        % (np.mean(losses), metric.__class__.__name__, acc_or_f1)
    )
    metric.reset()
    model.train()
    return acc_or_f1


def predict(model, test_dataloader, file_path, memories, label_list):
    label_dict = dict()
    model.eval()
    for _, batch in enumerate(test_dataloader, start=1):
        (
            input_ids,
            position_ids,
            token_type_ids,
            attn_mask,
            _,
            qids,
            gather_idxs,
            need_cal_loss,
        ) = batch
        logits, memories = model(
            input_ids, memories, token_type_ids, position_ids, attn_mask
        )
        logits, qids = list(
            map(lambda x: paddle.gather(x, gather_idxs), [logits, qids])
        )
        probs = nn.functional.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_list[i] for i in idx]
        for i, qid in enumerate(qids.numpy().flatten()):
            label_dict[str(qid)] = labels[i]
    to_json_file("iflytek", label_dict, file_path)


def do_train(args):
    set_seed(args)

    (
        tokenizer_class,
        eval_name,
        test_name,
        preprocess_text_fn,
        eval_metric,
    ) = DATASET_INFO[args.dataset]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    train_ds, eval_ds, test_ds = load_dataset(
        "clue", name=args.dataset, splits=["train.json", eval_name, test_name]
    )
    num_classes = len(train_ds.label_list)

    paddle.set_device(args.device)
    trainer_num = paddle.distributed.get_world_size()
    if trainer_num > 1:
        paddle.distributed.init_parallel_env()
    rank = paddle.distributed.get_rank()
    if rank == 0:
        if os.path.exists(args.model_name_or_path):
            logger.info("init checkpoint from %s" % args.model_name_or_path)
    model = ErnieDocForSequenceClassification.from_pretrained(
        args.model_name_or_path, num_classes=num_classes
    )
    model_config = model.ernie_doc.config
    if trainer_num > 1:
        model = paddle.DataParallel(model)

    train_ds_iter = ClassifierIterator(
        train_ds,
        args.batch_size,
        tokenizer,
        trainer_num,
        trainer_id=rank,
        memory_len=model_config["memory_len"],
        max_seq_length=args.max_seq_length,
        random_seed=args.seed,
        preprocess_text_fn=preprocess_text_fn,
    )
    eval_ds_iter = ClassifierIterator(
        eval_ds,
        args.batch_size,
        tokenizer,
        trainer_num,
        trainer_id=rank,
        memory_len=model_config["memory_len"],
        max_seq_length=args.max_seq_length,
        mode="eval",
        preprocess_text_fn=preprocess_text_fn,
    )
    test_ds_iter = ClassifierIterator(
        test_ds,
        args.batch_size,
        tokenizer,
        trainer_num,
        trainer_id=rank,
        memory_len=model_config["memory_len"],
        max_seq_length=args.max_seq_length,
        mode="test",
        preprocess_text_fn=preprocess_text_fn,
    )

    train_dataloader = paddle.io.DataLoader.from_generator(
        capacity=70, return_list=True
    )
    train_dataloader.set_batch_generator(train_ds_iter, paddle.get_device())
    eval_dataloader = paddle.io.DataLoader.from_generator(capacity=70, return_list=True)
    eval_dataloader.set_batch_generator(eval_ds_iter, paddle.get_device())
    test_dataloader = paddle.io.DataLoader.from_generator(capacity=70, return_list=True)
    test_dataloader.set_batch_generator(test_ds_iter, paddle.get_device())

    num_training_examples = train_ds_iter.get_num_examples()
    num_training_steps = (
        args.epochs * num_training_examples // args.batch_size // trainer_num
    )  # 训练步数
    logger.info("Device count: %d, trainer_id: %d" % (trainer_num, rank))
    logger.info("Num train.json examples: %d" % num_training_examples)
    logger.info("Max train.json steps: %d" % num_training_steps)
    logger.info(
        "Num warmup steps: %d" % int(num_training_steps * args.warmup_proportion)
    )

    lr_scheduler = LinearDecayWithWarmup(
        args.learning_rate, num_training_steps, args.warmup_proportion
    )

    decay_params = [
        p.name
        for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    # Construct dict
    name_dict = dict()
    for n, p in model.named_parameters():
        name_dict[p.name] = n

    simple_lr_setting = partial(
        layerwise_lr_decay,
        args.layerwise_decay,
        name_dict,
        model_config["num_hidden_layers"],
    )

    optimizer = AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params
        # ,
        # lr_ratio=simple_lr_setting
    )

    criterion = paddle.nn.loss.CrossEntropyLoss()
    metric = paddle.metric.Accuracy()

    global_steps = 0
    best_acc = -1
    create_memory = partial(
        init_memory,
        args.batch_size,
        args.memory_length,
        model_config["hidden_size"],
        model_config["num_hidden_layers"],
    )
    # Copy the memory
    memories = create_memory()
    tic_train = time.time()
    stop_training = False
    for epoch in range(args.epochs):
        train_ds_iter.shuffle_sample()
        train_dataloader.set_batch_generator(train_ds_iter, paddle.get_device())
        for step, batch in enumerate(train_dataloader, start=1):
            global_steps += 1
            (
                input_ids,
                position_ids,
                token_type_ids,
                attn_mask,
                labels,
                qids,
                gather_idx,
                need_cal_loss,
            ) = batch
            logits, memories = model(
                input_ids, memories, token_type_ids, position_ids, attn_mask
            )

            logits, labels = list(
                map(lambda x: paddle.gather(x, gather_idx), [logits, labels])
            )
            loss = criterion(logits, labels) * need_cal_loss
            mean_loss = loss.mean()
            mean_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            # Rough acc result, not a precise acc
            acc = metric.compute(logits, labels) * need_cal_loss
            metric.update(acc)

            if global_steps % args.logging_steps == 0:
                logger.info(
                    "train.json: global step %d, epoch: %d, loss: %f, acc:%f, lr: %f, speed: %.2f step/s"
                    % (
                        global_steps,
                        epoch,
                        mean_loss,
                        metric.accumulate(),
                        lr_scheduler.get_lr(),
                        args.logging_steps / (time.time() - tic_train),
                    )
                )
                tic_train = time.time()

            if global_steps % args.save_steps == 0:
                # Evaluate
                logger.info("Eval:")
                eval_acc = evaluate(
                    model, eval_metric, eval_dataloader, create_memory()
                )
                # Save
                if rank == 0:
                    output_dir = os.path.join(
                        args.output_dir, "model_%d" % (global_steps)
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model._layers
                        if isinstance(model, paddle.DataParallel)
                        else model
                    )
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    if eval_acc > best_acc:
                        logger.info("Save best model......")
                        best_acc = eval_acc
                        best_model_dir = os.path.join(output_dir, "best_model")
                        if not os.path.exists(best_model_dir):
                            os.makedirs(best_model_dir)
                        model_to_save.save_pretrained(best_model_dir)
                        tokenizer.save_pretrained(best_model_dir)

            if args.max_steps > 0 and global_steps >= args.max_steps:
                stop_training = True
                break
        if stop_training:
            break
    logger.info("Final test result:")
    eval_acc = evaluate(model, eval_metric, eval_dataloader, create_memory())
    logger.info("start predict the test data")

    create_memory = partial(
        init_memory,
        args.batch_size,
        args.memory_length,
        model_config["hidden_size"],
        model_config["num_hidden_layers"],
    )
    # Copy the memory
    memories = create_memory()
    predict(model, test_dataloader, args.file_path, memories, test_ds.label_list)
    logger.info(
        "Done Predicting the results has been saved in file: {}".format(args.file_path)
    )


if __name__ == "__main__":
    do_train(args)
