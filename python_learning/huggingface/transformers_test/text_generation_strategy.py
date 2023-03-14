# -*-coding:utf-8 -*-

"""
# File       : text_generation_strategy.py
# Time       ：2023/3/13 13:46
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from transformers import AutoTokenizer, AutoModelForCausalLM

checkpoint = "gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

prompt = "Hugging Face Company is"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, penalty_alpha=0.6, top_k=4, max_new_tokens=100)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))