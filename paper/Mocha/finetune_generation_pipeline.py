# -*-coding:utf-8 -*-

"""
# File       : finetune_generation_pipeline.py
# Time       ：2023/2/16 16:41
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import json
import time
from tqdm import tqdm
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch
import os
from itertools import chain
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn import DataParallel
import numpy as np
from datetime import datetime
import transformers
import random


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true'):
        return True
    elif v.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class ArgenPlanFewShotS2sDataset(Dataset):
    def __init__(self, tokenizer, data_path, few_shot_rate, max_len=512, batch_first=True, is_mtl=True):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.is_mtl = is_mtl
        self.few_shot_rate = few_shot_rate

        self.examples = self.load_data()
        self.batch_first = batch_first
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id

    def process(self, line_json, with_eos=True):
        if line_json["task"] == "revise_shuffle" or line_json["task"] == "revise_kp":
            context = random.sample(line_json["input_list"], 1)[0]
            response = line_json["output"]
        elif line_json["task"] == "distingush":
            flag_int = random.randint(0, 9)
            if flag_int <= 4:  # positive
                context = line_json["pos"]["input"]
                response = line_json["pos"]["output"]
            else:
                sample_res = random.sample(line_json["neg_list"], 1)[0]
                context = sample_res["input"]
                response = sample_res["output"]
        else:
            context = line_json["input"]
            response = line_json["output"]

        src_id = self.tokenizer.encode(context)
        tgt_id = self.tokenizer.encode(response)

        instance = {}
        instance["input_ids"] = src_id[:self.max_len]
        instance["lm_labels"] = tgt_id[:self.max_len]
        instance["original_json"] = line_json
        return instance

    def load_data(self):
        data = [json.loads(ln) for ln in open(self.data_path).readlines()]
        all_ids = sorted(list(set([elem["id"] for elem in data])))
        random.shuffle(all_ids)
        few_shot_num = int(len(all_ids) * self.few_shot_rate)
        selected_ids = all_ids[:few_shot_num]
        print("selected ids: ", selected_ids[:10])
        selected_ids = set(selected_ids)
        data = [elem for elem in data if elem["id"] in selected_ids]

        data_filter = data
        if not self.is_mtl:
            data_filter = [elem for elem in data_filter if elem["task"] == "generation"]
        return data_filter

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        sample = self.examples[index]
        instance = self.process(sample)
        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad_id)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad_id)
        original_json = [instance["original_json"] for instance in batch]
        return input_ids, labels, original_json


class ArgenPlanS2sDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_len=512, batch_first=True, is_mtl=True):
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.is_mtl = is_mtl
        self.examples = self.load_data()
        self.batch_first = batch_first
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id

    def process(self, line_json, with_eos=True):
        if line_json["task"] == "revise_shuffle" or line_json["task"] == "revise_kp":
            context = random.sample(line_json["input_list"], 1)[0]
            response = line_json["output"]
        elif line_json["task"] == "distingush":
            flag_int = random.randint(0, 9)
            if flag_int <= 4:  # positive
                context = line_json["pos"]["input"]
                response = line_json["pos"]["output"]
            else:
                sample_res = random.sample(line_json["neg_list"], 1)[0]
                context = sample_res["input"]
                response = sample_res["output"]
        else:
            context = line_json["input"]
            response = line_json["output"]

        src_id = self.tokenizer.encode(context)
        tgt_id = self.tokenizer.encode(response)

        instance = {}
        instance["input_ids"] = src_id[:self.max_len]
        instance["lm_labels"] = tgt_id[:self.max_len]
        instance["original_json"] = line_json
        return instance

    def load_data(self):
        data = [json.loads(ln) for ln in open(self.data_path).readlines()]
        data_filter = data
        if not self.is_mtl:
            data_filter = [elem for elem in data_filter if elem["task"] == "generation"]

        # # ablation study:
        # # planning, surface_realization, revise_shuffle, revise_kp, distingush
        # # ablation_task = ["planning", "surface_realization"]
        # ablation_task = ["revise_shuffle", "revise_kp", "distingush"]
        # print("ablation: ", ablation_task)
        # data_filter = [elem for elem in data_filter if elem["task"] not in ablation_task]

        print("used task: ", set([elem["task"] for elem in data_filter]))

        return data_filter

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        sample = self.examples[index]
        instance = self.process(sample)
        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad_id)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad_id)
        original_json = [instance["original_json"] for instance in batch]
        return input_ids, labels, original_json


def generate(batch_size, model_path, tokenizer_path, test_path, write_path, max_length, fp16=False,
             fp16_opt_level='O1'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    print("load model from: ", model_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()

    print("load data from: ", test_path)
    print("write results to: ", write_path)
    print("decoding max length: ", max_length)

    f_w = open(write_path, "w")

    eval_data = ArgenPlanS2sDataset(
        tokenizer=tokenizer,
        data_path=test_path,
        batch_first=True,
        is_mtl=False
    )
    eval_dataloader = DataLoader(
        eval_data,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=eval_data.collate,
        shuffle=False,
        num_workers=8
    )

    print("number of test: {}".format(eval_data.__len__()))

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model = amp.initialize(model, opt_level=fp16_opt_level)

    print('starting decoding')
    for cur_epoch in range(1):
        for batch_step, cur_batch in enumerate(eval_dataloader):
            batch_input_ids = cur_batch[0].to("cuda")
            batch_labels = cur_batch[1].to("cuda")
            batch_context_mask = (batch_input_ids != tokenizer.pad_token_id).long()

            batch_original_json = cur_batch[2]

            # topp-sampling
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_context_mask,
                max_length=max_length,
                do_sample=True,
                top_k=10,
                top_p=0.9,
                use_cache=True,
            )

            for i in range(outputs.size()[0]):
                cur_gen = tokenizer.decode(outputs[i])
                cur_ref = tokenizer.decode(batch_labels[i])
                cur_input = tokenizer.decode(batch_input_ids[i])
                cur_json = batch_original_json[i]
                cur_json["input_src"] = cur_input.replace("<pad>", "").strip()
                cur_json["gen"] = cur_gen.replace("<pad>", "").strip()
                f_w.write(json.dumps(cur_json, ensure_ascii=False) + "\n")

    print("finish decoding")


def eval_model_loss(model, tokenizer, eval_dataloader, device, epoch_id):
    tot_loss = []
    tot_sample = []
    tot_ppl = []
    with torch.no_grad():
        for step, cur_batch in enumerate(eval_dataloader):
            batch_input_ids = cur_batch[0].to(device)
            batch_labels = cur_batch[1].to(device)
            batch_context_mask = (batch_input_ids != tokenizer.pad_token_id).long()

            outputs = model.forward(
                input_ids=batch_input_ids,
                attention_mask=batch_context_mask,
                labels=batch_labels,
            )

            loss = outputs["loss"]
            ppl = torch.exp(loss)

            n_sample = batch_input_ids.shape[0]
            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl.mean().item() * n_sample)
            tot_sample.append(n_sample)
    print("eval: ", len(tot_loss))
    print(
        f"Step {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)}")
    return np.sum(tot_loss) / np.sum(tot_sample)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1001, type=int, required=False)
    parser.add_argument('--epochs', default=15, type=int, required=False)
    parser.add_argument("--num_optim_steps", type=int, default=10000,
                        help="new API specifies num update steps")
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='batch size')
    parser.add_argument('--lr', default=5e-5, type=float, required=False)
    parser.add_argument('--warmup_steps', default=500, type=int, required=False)
    parser.add_argument('--save_step', default=2000, type=int, required=False)
    parser.add_argument('--log_step', default=100, type=int, required=False)
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    # for task
    parser.add_argument('--output_dir', default=f'save_model/cmv_mtl', type=str, required=False)
    parser.add_argument('--tokenizer_path', default=f'./model_card/t5',
                        type=str, required=False)
    parser.add_argument('--model_path', default=f'./model_card/t5',
                        type=str, required=False)
    parser.add_argument('--task', default=f'cmv', type=str, required=False)
    parser.add_argument('--is_mtl', default=False, type=str2bool, required=False)
    parser.add_argument('--few_shot_rate', default=-1, type=float, required=False)

    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    is_distributed = (args.local_rank != -1)
    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    batch_size = args.batch_size

    lr = args.lr
    gradient_accumulation = args.gradient_accumulation
    log_step = args.log_step * gradient_accumulation
    fp16 = args.fp16
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    output_dir = args.output_dir
    save_step = args.save_step

    # fix all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    assert log_step % gradient_accumulation == 0

    tokenizer_path = args.tokenizer_path
    model_path = args.model_path
    if args.local_rank == 0:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        print("save path: ", output_dir)
        print("initialize model from: ", model_path)

    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    model.train()
    model.to(device)

    task = args.task
    is_mtl = args.is_mtl
    print("is mtl train: ", is_mtl)

    if task == "cmv":  # argument generation
        train_path = "./data/cmv/cmv_train.jsonl"
        dev_path = "./data/cmv/cmv_dev.jsonl"
        test_path = "./data/cmv/cmv_test.jsonl"
        max_length = 200
    elif task == "nyt":  # article writing
        train_path = "./data/nyt/nyt_train.jsonl"
        dev_path = "./data/nyt/nyt_dev.jsonl"
        test_path = "./data/nyt/nyt_test.jsonl"
        max_length = 350
    elif task == "wiki_plot":  # story
        train_path = "./data/wikiplot/wikiplot_train.jsonl"
        dev_path = "./data/wikiplot/wikiplot_dev.jsonl"
        test_path = "./data/wikiplot/wikiplot_test.jsonl"
        max_length = 512
    else:
        print("task error: ", task)

    if args.few_shot_rate == -1:  # full shot training
        train_data = ArgenPlanS2sDataset(
            tokenizer=tokenizer,
            data_path=train_path,
            batch_first=True,
            is_mtl=is_mtl
        )
    else:  # few shot training
        train_data = ArgenPlanFewShotS2sDataset(
            tokenizer=tokenizer,
            data_path=train_path,
            batch_first=True,
            is_mtl=is_mtl,
            few_shot_rate=args.few_shot_rate
        )
        log_step = int(log_step * args.few_shot_rate)
        if save_step != -1:
            save_step = int(save_step * args.few_shot_rate)
        print(f"log step: {log_step}, save step: {save_step}")

    eval_data = ArgenPlanS2sDataset(
        tokenizer=tokenizer,
        data_path=dev_path,
        batch_first=True,
        is_mtl=is_mtl
    )

    data_sampler = DistributedSampler(train_data)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=train_data.collate,
        sampler=data_sampler,
        num_workers=8
    )

    eval_dataloader = DataLoader(
        eval_data,
        batch_size=batch_size,
        drop_last=True,
        collate_fn=eval_data.collate,
        shuffle=True,
        num_workers=4
    )

    if is_mtl:
        eval_gen_data = ArgenPlanS2sDataset(
            tokenizer=tokenizer,
            data_path=dev_path,
            batch_first=True,
            is_mtl=False
        )
        eval_gen_dataloader = DataLoader(
            eval_gen_data,
            batch_size=batch_size,
            drop_last=True,
            collate_fn=eval_data.collate,
            shuffle=True,
            num_workers=4
        )

    if args.local_rank == 0:
        print("load data from: ", train_path)
        print("number of train: {} dev: {}".format(train_data.__len__(), eval_data.__len__()))

    multi_gpu = False
    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if is_distributed:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                        find_unused_parameters=True)
        multi_gpu = True

    ##############################
    # training
    ##############################
    print('starting training')
    running_loss = 0
    running_ppl = 0

    current_step = 0
    cur_epoch = 0
    best_eval_loss = 10000
    best_eval_gen_loss = 100000
    for cur_epoch in range(args.epochs):
        if is_distributed:
            data_sampler.set_epoch(cur_epoch)
        for batch_step, cur_batch in enumerate(train_dataloader):
            current_step += 1
            batch_input_ids = cur_batch[0].to(device)
            batch_labels = cur_batch[1].to(device)
            batch_context_mask = (batch_input_ids != tokenizer.pad_token_id).long()

            outputs = model.forward(
                input_ids=batch_input_ids,
                attention_mask=batch_context_mask,
                labels=batch_labels,
            )

            loss = outputs["loss"]
            # note: this is only estimated value for ppl: needed to imporved
            ppl = torch.exp(loss)

            #  get loss
            if multi_gpu:
                loss = loss.mean()
                ppl = ppl.mean()
            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation
                ppl = ppl / gradient_accumulation

            #  loss backward
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            #  optimizer step
            if current_step % gradient_accumulation == 0:
                running_loss += loss.item()
                running_ppl += ppl.item()
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()

            if args.local_rank != 0:
                continue

            if current_step % log_step == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                print('now time: {}:{}. Step {} of of epoch {}, lr: {:.7f}, loss {:.3f}, ppl {:.3f}'.format(
                    datetime.now().hour,
                    datetime.now().minute,
                    current_step,
                    cur_epoch,
                    cur_lr,
                    running_loss * gradient_accumulation / (log_step / gradient_accumulation),
                    running_ppl * gradient_accumulation / (log_step / gradient_accumulation),
                ))
                running_loss = 0
                running_ppl = 0

            # save_step != -1: save with steps; otherwise save for each epoch
            if save_step != -1 and current_step % save_step == 0:
                print("start evaluation")
                model.eval()
                eval_loss = eval_model_loss(model, tokenizer, eval_dataloader, device, current_step)
                if is_mtl:
                    eval_gen_loss = eval_model_loss(model, tokenizer, eval_gen_dataloader, device, current_step)
                model.train()

                print('saving model for step {}'.format(current_step))
                if not os.path.exists(output_dir + '/model_step{}'.format(current_step)):
                    os.mkdir(output_dir + '/model_step{}'.format(current_step))
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir + '/model_step{}'.format(current_step))

                if eval_loss < best_eval_loss:
                    print("save best.....")
                    if not os.path.exists(output_dir + '/best_eval'):
                        os.mkdir(output_dir + '/best_eval')
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir + '/best_eval')
                    best_eval_loss = eval_loss

                if is_mtl:
                    if eval_gen_loss < best_eval_gen_loss:
                        print("save best gen model.....")
                        if not os.path.exists(output_dir + '/best_eval_gen'):
                            os.mkdir(output_dir + '/best_eval_gen')
                        model_to_save = model.module if hasattr(model, 'module') else model
                        model_to_save.save_pretrained(output_dir + '/best_eval_gen')
                        best_eval_gen_loss = eval_gen_loss

        if args.local_rank == 0 and save_step == -1:
            # do validation & save
            print("start evaluation")
            model.eval()
            eval_loss = eval_model_loss(model, tokenizer, eval_dataloader, device, current_step)
            if is_mtl:
                eval_gen_loss = eval_model_loss(model, tokenizer, eval_gen_dataloader, device, current_step)
            model.train()

            if eval_loss < best_eval_loss:
                print("save best.....")
                if not os.path.exists(output_dir + '/best_eval'):
                    os.mkdir(output_dir + '/best_eval')
                model_to_save = model.module if hasattr(model, 'module') else model
                model_to_save.save_pretrained(output_dir + '/best_eval')
                best_eval_loss = eval_loss

    print('training finished')

    ##############################
    # decoding
    ##############################
    del model
    print("##############################")
    print('starting decoding')
    print("##############################")

    if args.local_rank == 0:
        generate(
            batch_size=150,
            model_path=output_dir + '/best_eval',
            tokenizer_path=args.tokenizer_path,
            test_path=test_path,
            write_path=output_dir + f"/{task}_{is_mtl}_{args.few_shot_rate}_{args.seed}_output_best.jsonl",
            max_length=max_length,
        )


if __name__ == "__main__":
    train()