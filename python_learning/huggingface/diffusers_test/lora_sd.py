# -*-coding:utf-8 -*-

"""
# File       : lora_sd.py
# Time       ：2023/3/17 15:00
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from huggingface_hub import model_info

# LoRA weights ~3 MB
model_path = "sayakpaul/sd-model-finetuned-lora-t4"

info = model_info(model_path)
model_base = info.cardData["base_model"]
print(model_base)  # CompVis/stable-diffusion-v1-4
