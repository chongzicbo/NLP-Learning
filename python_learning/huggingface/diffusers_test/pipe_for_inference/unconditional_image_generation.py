# -*-coding:utf-8 -*-

"""
# File       : unconditional_image_generation.py
# Time       ：2023/3/15 11:22
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from diffusers import DiffusionPipeline

generator = DiffusionPipeline.from_pretrained("google/ddpm-celebahq-256")
generator.to("cuda")
image = generator().images[0]

from matplotlib import pyplot as plt

plt.imshow(image)
plt.show()
