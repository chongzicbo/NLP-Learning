# -*-coding:utf-8 -*-

"""
# File       : bert_onnx_test.py
# Time       ：2023/3/13 17:14
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""

import onnx
import torch
from transformers import AutoTokenizer

# onnx_model = onnx.load("/home/bocheng/data/model_saved/test.onnx")
# onnx.checker.check_model(onnx_model)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
import onnxruntime as ort

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

ort_sess = ort.InferenceSession("/home/bocheng/data/model_saved/test.onnx", providers=['CPUExecutionProvider'])
text = "Text from the news article"
inputs = tokenizer(text, padding='max_length', truncation=True)
input_ids, token_type_ids, attention_mask = torch.unsqueeze(torch.tensor(inputs["input_ids"]), 0), torch.unsqueeze(
    torch.tensor(
        inputs['token_type_ids']), 0), torch.unsqueeze(torch.tensor(inputs['attention_mask']), 0)

{
    'input_ids': input_ids.numpy(),
    'token_type_ids': token_type_ids.numpy(),
    'attention_mask': attention_mask.numpy()
}
outputs = ort_sess.run(None,{
    'input_ids': input_ids.numpy(),
    'token_type_ids': token_type_ids.numpy(),
    'attention_mask': attention_mask.numpy()
} )
print(outputs)