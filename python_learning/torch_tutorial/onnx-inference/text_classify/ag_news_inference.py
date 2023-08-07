#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: ag_news_inference.py
@time: 2022/11/10 11:31
"""
import torch
from ag_news_classify import (
    TextClassificationModel,
    train_iter,
    build_vocab_from_iterator,
    text_pipeline,
    yield_tokens,
    tokenizer,
    ag_news_label,
)

import torch

text = "Text from the news article"
text = torch.tensor(text_pipeline(text))
offsets = torch.tensor([0])
print(text)
print(offsets)
# Export the model
model = torch.load("ag_new.pth")
torch.onnx.export(
    model,  # model being run
    (text, offsets),  # model input (or a tuple for multiple inputs)
    "ag_news_model.onnx",  # where to save the model (can be a file or file-like object)
    export_params=True,  # store the trained parameter weights inside the model file
    opset_version=10,  # the ONNX version to export the model to
    do_constant_folding=True,  # whether to execute constant folding for optimization
    input_names=["input", "offsets"],  # the model's input names
    output_names=["output"],  # the model's output names
    dynamic_axes={
        "input": {0: "batch_size"},  # variable length axes
        "output": {0: "batch_size"},
    },
)

import onnx

onnx_model = onnx.load("ag_news_model.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime as ort
import numpy as np

ort_sess = ort.InferenceSession("ag_news_model.onnx")
outputs = ort_sess.run(
    None, {"input": text.numpy(), "offsets": torch.tensor([0]).numpy()}
)
# Print Result
result = outputs[0].argmax(axis=1) + 1
print("This is a %s news" % ag_news_label[result[0]])
