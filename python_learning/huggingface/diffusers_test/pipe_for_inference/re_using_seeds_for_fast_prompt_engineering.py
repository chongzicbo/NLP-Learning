# -*-coding:utf-8 -*-

"""
# File       : re_using_seeds_for_fast_prompt_engineering.py
# Time       ：2023/3/15 12:02
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import torch

prompt = "Labrador in the style of Vermeer"
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
import torch

generator = [torch.Generator(device="cuda").manual_seed(i + 10) for i in range(4)]
images = pipe(prompt, generator=generator, num_images_per_prompt=4).images
pipe.safety_checker = lambda images, clip_input: (images, False)
import matplotlib.pyplot as plt

plt.figure()
for i in range(len(images)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i])
plt.show()
prompt = [
    prompt + t for t in [", highly realistic", ", artsy", ", trending", ", colorful"]
]
generator = [torch.Generator(device="cuda").manual_seed(0) for i in range(4)]
