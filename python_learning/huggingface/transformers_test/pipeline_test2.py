# -*-coding:utf-8 -*-

"""
# File       : pipeline_test2.py
# Time       ：2023/3/10 14:41
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""

from transformers import pipeline

generator = pipeline(task="automatic-speech-recognition")

text = generator(
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
)
print(text)
