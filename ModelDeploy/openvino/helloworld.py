# -*-coding:utf-8 -*-

"""
# File       : helloworld.py
# Time       ：2023/3/14 9:33
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from openvino.runtime import Core, PartialShape

ie = Core()
#
onnx_model_path = "/home/bocheng/data/model_saved/test.onnx"
#
# model_onnx = ie.read_model(model=onnx_model_path)
# print(model_onnx.inputs)
# compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")
# # print(compiled_model_onnx)
#
# input_layer = model_onnx.input(0)
# print(input_layer.any_name)
# print(f"input precision: {input_layer.element_type}")
# print(f"input shape: {input_layer.shape}")
# print(model_onnx.outputs)
#
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
#
# text = "Text from the news article Text from the news article Text from the news article Text from the news article Text from the news article Text from the news article Text from the news article Text from the news article Text from the news article Text from the news article Text from the news article Text from the news article Text from the news article Text from the news article"
# inputs = tokenizer(text, padding='max_length', truncation=True)
# input_ids, token_type_ids, attention_mask = torch.unsqueeze(torch.tensor(inputs["input_ids"]), 0), torch.unsqueeze(
#     torch.tensor(
#         inputs['token_type_ids']), 0), torch.unsqueeze(torch.tensor(inputs['attention_mask']), 0)
# inputs = {
#     'input_ids': input_ids.numpy(),
#     'token_type_ids': token_type_ids.numpy(),
#     'attention_mask': attention_mask.numpy()
# }
# result = compiled_model_onnx(inputs)
# for k,v in result.items():
#     print(k,v)
# output_layer = compiled_model_onnx.output(0)
# request = compiled_model_onnx.create_infer_request()
# start_time=time.time()
# request.infer(inputs=inputs)
# result = request.get_output_tensor(output_layer.index).data
# print(result)
# end_time=time.time()
# print(end_time-start_time)
# from openvino.runtime import serialize
# serialize(model_onnx,xml_path='/home/bocheng/data/model_saved/test.xml')

model_onnx = ie.read_model(model='/home/bocheng/data/model_saved/test.onnx')



new_shape = PartialShape([-1, 512])
print(model_onnx.output(0).any_name)
model_onnx.reshape({model_onnx.input(0).any_name: new_shape, model_onnx.input(1).any_name: new_shape,
                    model_onnx.input(2).any_name: new_shape})

compiled_model_onnx = ie.compile_model(model=model_onnx, device_name="CPU")

text = ["Text from the news article", "today is a good day"]

inputs = tokenizer(text, padding='max_length', truncation=True)
input_ids, token_type_ids, attention_mask = torch.tensor(inputs["input_ids"]),torch.tensor(
        inputs['token_type_ids']), torch.tensor(inputs['attention_mask'])
inputs = {
    'input_ids': input_ids.numpy(),
    'token_type_ids': token_type_ids.numpy(),
    'attention_mask': attention_mask.numpy()
}
print(input_ids.shape)

output_layer = compiled_model_onnx.output(0)
request = compiled_model_onnx.create_infer_request()
request.infer(inputs=inputs)
result = request.get_output_tensor(output_layer.index).data
print(result)