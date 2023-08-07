# -*-coding:utf-8 -*-

"""
# File       : multi_input_output.py
# Time       ：2023/3/17 10:46
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import gradio as gr


def greet(name, is_morning, temperature):
    salutation = "Good morning" if is_morning else "Good evening"
    greeting = f"{salutation} {name}. it is {temperature} degrees today"
    celsius = (temperature - 32) * 5 / 9
    return greeting, round(celsius, 2)


demo = gr.Interface(
    fn=greet, inputs=["text", "checkbox", gr.Slider(0, 100)], outputs=["text", "number"]
)
demo.launch(server_name="188.188.1.250", server_port=9988)
