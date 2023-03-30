# -*-coding:utf-8 -*-

"""
# File       : example_inputs.py
# Time       ：2023/3/17 11:27
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import gradio as gr
def calculator(num1,operation,num2):
    if operation=='add':
        return num1+num2
    elif operation=='substract':
        return num1-num2
    elif operation=='multiply':
        return num1*num2
    elif operation=='divide':
        if num2==0:
            raise gr.Error('Cannot divide by zero!')
        return num1/num2

demo=gr.Interface(
    calculator,
    [
        'number',
        gr.Radio(['add','substract','multiply','divide']),
        'number'
    ],
    'number',
    examples=[
        [5, "add", 3],
        [4, "divide", 2],
        [-4, "multiply", 2.5],
        [0, "subtract", 1.2],
    ],
    title="Toy Calculator",
    description="Here's a sample toy calculator. Allows you to calculate things like $2+2=4$",
)
demo.launch(server_name='188.188.1.250',server_port=9988)