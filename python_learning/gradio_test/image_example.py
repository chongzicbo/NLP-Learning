# -*-coding:utf-8 -*-

"""
# File       : image_example.py
# Time       ：2023/3/17 10:52
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import numpy as np
import gradio as gr

def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    return sepia_img

demo = gr.Interface(sepia, gr.Image(), "image")
demo.launch(server_name='188.188.1.250',server_port=9988)
