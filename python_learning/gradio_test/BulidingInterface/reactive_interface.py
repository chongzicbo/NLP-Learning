# -*-coding:utf-8 -*-

"""
# File       : reactive_interface.py
# Time       ：2023/3/17 15:54
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import gradio as gr

def calculator(num1, operation, num2):
    if operation == "add":
        return num1 + num2
    elif operation == "subtract":
        return num1 - num2
    elif operation == "multiply":
        return num1 * num2
    elif operation == "divide":
        return num1 / num2

demo = gr.Interface(
    calculator,
    [
        "number",
        gr.Radio(["add", "subtract", "multiply", "divide"]),
        "number"
    ],
    "number",
    live=True,
)
demo.launch(server_name='188.188.1.250',server_port=9988)
