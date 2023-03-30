# -*-coding:utf-8 -*-

"""
# File       : interface_state.py
# Time       ：2023/3/17 12:13
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""

import gradio as gr

# 1. global state
# scores = []
#
#
# def track_score(score):
#     scores.append(score)
#     top_scores = sorted(scores, reverse=True)[:3]
#     return top_scores
#
#
# demo = gr.Interface(
#     track_score,
#     gr.Number(label="Score"),
#     gr.JSON(label="Top Scores")
# )
# demo.launch(server_name='188.188.1.250',server_port=9988)

# 2.session state

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-medium')


def user(message, history):
    return '', history + [[message, None]]


def bot(history):
    user_message = history[-1][0]
    new_user_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensor='pt')
    bot_input_ids = torch.cat([torch.LongTensor(history), new_user_input_ids], dim=-1)
    history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id).tolist()
    response = tokenizer.decode(history[0]).split("<|endoftext|>")
    response = [(response[i], response[i + 1]) for i in range(0, len(response) - 1, 2)]  # convert to tuples of list
    return history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("clear")
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(server_name='188.188.1.250',server_port=9988)
