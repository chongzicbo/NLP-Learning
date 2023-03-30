# -*-coding:utf-8 -*-

"""
# File       : example-01-stablediffusion.py
# Time       ：2023/3/8 9:38
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
print(torch.cuda.is_available())

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "a beautiful japan woman"
image = pipe(prompt).images[0]
plt.imshow(image)
plt.show()