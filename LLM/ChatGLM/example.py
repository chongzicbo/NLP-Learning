# -*-coding:utf-8 -*-

"""
# File       : example.py
# Time       ：2023/3/15 9:03
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import os
import platform
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("/data/model/huggingface/models--THUDM--chatglm-6b/snapshots/220f772e9a2d7a55701d9b49bf2efc618acc3b56", trust_remote_code=True)
model = AutoModel.from_pretrained("/data/model/huggingface/models--THUDM--chatglm-6b/snapshots/220f772e9a2d7a55701d9b49bf2efc618acc3b56", trust_remote_code=True).half().cuda()
model = model.eval()

os_name = platform.system()

history = []
print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
while True:
    query = input("\n用户：")
    if query == "stop":
        break
    if query == "clear":
        history = []
        command = 'cls' if os_name == 'Windows' else 'clear'
        os.system(command)
        print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
        continue
    response, history = model.chat(tokenizer, query, history=history)
    print(f"ChatGLM-6B：{response}")