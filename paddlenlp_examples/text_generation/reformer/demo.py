"""
@author: chengbo
@software: PyCharm
@file: demo.py
@time: 2022/6/10 12:27
"""

import paddle
from paddlenlp.transformers import ReformerModelWithLMHead


# encoding
def encode(list_of_strings, pad_token_id=0):
    max_length = max([len(string) for string in list_of_strings])
    attention_masks = paddle.zeros((len(list_of_strings), max_length), dtype="int64")
    input_ids = paddle.full((len(list_of_strings), max_length), pad_token_id, dtype="int64")
    for idx, string in enumerate(list_of_strings):
        if not isinstance(string, bytes):
            string = str.encode(string)
        input_ids[idx, :len(string)] = paddle.to_tensor([x + 2 for x in string], dtype="int64")
        attention_masks[idx, :len(string)] = 1
    return input_ids, attention_masks


# Decoding
def decode(outputs_ids):
    decoded_outputs = []
    for output_ids in outputs_ids.tolist():
        decoded_outputs.append("".join([chr(x - 2) if x > 1 else "" for x in output_ids]))
    return decoded_outputs


if __name__ == '__main__':
    model = ReformerModelWithLMHead.from_pretrained("reformer-enwik8")
    model.eval()
    encoded, attention_masks = encode(["In 1965, Brooks left IBM to found the Department of"])
    output = decode(model.generate(encoded, decode_strategy="greedy_search", max_length=150, repetition_penalty=1.2)[0])
    print(output)
