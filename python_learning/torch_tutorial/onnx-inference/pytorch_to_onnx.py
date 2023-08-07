#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: pytorch_to_onnx.py
@time: 2022/11/10 10:58
"""
import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
model = torch.load("model.pt")
torch.onnx.export(
    model,
    torch.randn(1, 28, 28).to(device),
    "fashion_mnist_model.onnx",
    input_names=["input"],
    output_names=["output"],
)

import onnx

onnx_model = onnx.load("fashion_mnist_model.onnx")
onnx.checker.check_model(onnx_model)

print(onnx_model)
