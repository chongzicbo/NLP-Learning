# -*-coding:utf-8 -*-

"""
# File       : text2image_generation.py
# Time       ：2023/3/15 11:33
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from diffusers import DiffusionPipeline

generator = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")
generator.to("cuda")

image = generator("An image of a squirrel in Picasso style").images[0]
from matplotlib import pyplot as plt

plt.imshow(image)
plt.show()
