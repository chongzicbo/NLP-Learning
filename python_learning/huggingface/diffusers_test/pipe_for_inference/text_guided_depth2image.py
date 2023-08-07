# -*-coding:utf-8 -*-

"""
# File       : text_guided_depth2image.py
# Time       ：2023/3/15 11:52
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch
import requests
from PIL import Image

from diffusers import StableDiffusionDepth2ImgPipeline

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16,
).to("cuda")


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
init_image = Image.open(requests.get(url, stream=True).raw)
prompt = "two tigers"
n_prompt = "bad, deformed, ugly, bad anatomy"
image = pipe(
    prompt=prompt, image=init_image, negative_prompt=n_prompt, strength=0.7
).images[0]
