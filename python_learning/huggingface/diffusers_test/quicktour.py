# -*-coding:utf-8 -*-

"""
# File       : quicktour.py
# Time       ：2023/3/13 15:39
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.to('cuda')
from diffusers import DPMSolverMultistepScheduler

pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
generator=torch.Generator('cuda').manual_seed(0)
image = pipeline("a photo of a cat flying in sky",generator=generator).images[0]
from matplotlib import pyplot as plt

plt.imshow(image)
plt.show()
