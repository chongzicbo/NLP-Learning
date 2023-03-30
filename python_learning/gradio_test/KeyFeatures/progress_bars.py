# -*-coding:utf-8 -*-

"""
# File       : progress_bars.py
# Time       ：2023/3/17 11:50
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import gradio as gr
import time

def slowly_reverse(word, progress=gr.Progress()):
    progress(0, desc="Starting")
    time.sleep(1)
    progress(0.05)
    new_string = ""
    for letter in progress.tqdm(word, desc="Reversing"):
        time.sleep(0.25)
        new_string = letter + new_string
    return new_string

demo = gr.Interface(slowly_reverse, gr.Text(), gr.Text())

if __name__ == "__main__":
    demo.queue(concurrency_count=10).launch(server_name='188.188.1.250',server_port=9988)