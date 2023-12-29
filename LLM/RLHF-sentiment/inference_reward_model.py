# -*-coding:utf-8 -*-

"""
# File       : inference_reward_model.py
# Time       ：2023/4/21 11:23
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch
from rich import print
from transformers import AutoTokenizer

device = "cpu"
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/model_best/")
model = torch.load("./checkpoints/model_best/model.pt")

model.to(device).eval()

texts = ["买过很多箱这个苹果了,一如既往的好,汁多味甜", "一台充电很慢，信号不好！退了！又买了一台竟然是次品。服了。。"]
inputs = tokenizer(texts, max_length=128, padding="max_length", return_tensors="pt")
r = model(**inputs)
print(r)
