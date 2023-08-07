#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: basic_language_model_transformer_xl.py
@time: 2022/8/12 9:20
"""

# 调用transformer_xl模型，该模型流行度较低，未找到中文预训练模型
# last_hidden_state目前是debug到transformer包中查看，经比对和本框架一致
# 用的是transformer中的英文预训练模型来验证正确性
# 转换脚本: convert_script/convert_transformer_xl.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

pretrained_model = "F:/Projects/pretrain_ckpt/transformer_xl/[english_hugging_face_torch]--transfo-xl-wt103"

# ----------------------transformers包----------------------
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
model = AutoModelForCausalLM.from_pretrained(pretrained_model)
model.eval()
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
with torch.no_grad():
    # 这里只能断点进去看
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.losses
print("transforms loss: ", loss)

# ----------------------bert4torch配置----------------------
from bert4torch.models import build_transformer_model

config_path = f"{pretrained_model}/bert4torch_config.json"
checkpoint_path = f"{pretrained_model}/bert4torch_pytorch_model.bin"

model = build_transformer_model(
    config_path,
    checkpoint_path=checkpoint_path,
    model="transformer_xl",
)

print("bert4torch last_hidden_state: ", model.predict([inputs["input_ids"]]))
