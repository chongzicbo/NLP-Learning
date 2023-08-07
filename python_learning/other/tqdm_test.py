# -*-coding:utf-8 -*-

"""
# File       : tqdm_test.py
# Time       ：2023/3/13 11:04
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import time
from tqdm import tqdm, trange

for i in range(100):
    time.sleep(0.05)

for i in tqdm(range(100), desc="Processing"):
    time.sleep(0.05)

dic = ["a", "b", "c", "d", "e"]
pbar = tqdm(dic)
for i in pbar:
    pbar.set_description("Processing " + i)

time.sleep(0.2)

with tqdm(total=200) as pbar:
    pbar.set_description("Processing:")
    for i in range(20):
        time.sleep(0.1)
        pbar.update(10)
