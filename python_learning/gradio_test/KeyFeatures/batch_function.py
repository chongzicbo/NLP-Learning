# -*-coding:utf-8 -*-

"""
# File       : batch_function.py
# Time       ：2023/3/17 11:58
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import time
import gradio as gr


def trim_words(words, lens):
    trimmed_words = []
    time.sleep(5)
    for w, l in zip(words, lens):
        trimmed_words.append(w[: int(l)])
    return [trimmed_words]


#
# demo = gr.Interface(trim_words, ["textbox", "number"], ["output"],
#                     batch=True, max_batch_size=16)

with gr.Blocks() as demo:
    with gr.Row():
        word = gr.Textbox(label="word")
        leng = gr.Number(label="leng")
        output = gr.Textbox(label="Output")
    with gr.Row():
        run = gr.Button()

    event = run.click(trim_words, [word, leng], output, batch=True, max_batch_size=16)

demo.queue()
demo.launch(server_name="188.188.1.250", server_port=9988)
