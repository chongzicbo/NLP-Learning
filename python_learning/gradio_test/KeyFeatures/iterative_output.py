# -*-coding:utf-8 -*-

"""
# File       : iterative_output.py
# Time       ：2023/3/17 11:46
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import gradio as gr
import numpy as np
import time


def fake_diffusion(steps):
    for _ in range(steps):
        time.sleep(1)
        image = np.random.random((600, 600, 3))
        yield image
    image = "https://gradio-builds.s3.amazonaws.com/diffusion_image/cute_dog.jpg"
    yield image


demo = gr.Interface(fake_diffusion, inputs=gr.Slider(1, 10, 3), outputs="image")

# define queue - required for generators
demo.queue()

demo.launch(server_name="188.188.1.250", server_port=9988)
