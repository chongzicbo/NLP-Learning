# -*-coding:utf-8 -*-

"""
# File       : example02.py
# Time       ：2023/6/12 10:53
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import retrying
from diffusers import StableDiffusionPipeline
import torch


@retrying.retry
def get_image():
    model_id = "dreamlike-art/dreamlike-photoreal-1.0"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    prompt = "a sexy cute cat "
    image = pipe(prompt).images[0]

    image.save("./result1.jpg")


if __name__ == "__main__":
    get_image()
