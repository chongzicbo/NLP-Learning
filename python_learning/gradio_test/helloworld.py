# -*-coding:utf-8 -*-

"""
# File       : helloworld.py
# Time       ：2023/3/17 10:11
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import gradio as gr


def greet(name):
    return 'hello ' + name + " !"

demo=gr.Interface(fn=greet,inputs='text',outputs='text')

demo.launch(server_name='188.188.1.250',server_port=9988)

