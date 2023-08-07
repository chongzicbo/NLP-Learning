# -*-coding:utf-8 -*-

"""
# File       : components_attributes.py
# Time       ：2023/3/17 10:27
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import gradio as gr


def greet(name):
    return "hello " + name + "!"


demo = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(lines=2, placeholder="Name here ..."),
    outputs="text",
)
demo.launch(server_name="188.188.1.250", server_port=9988)
