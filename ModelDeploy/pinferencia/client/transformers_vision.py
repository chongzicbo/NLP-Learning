#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: example_01.py
@time: 2022/10/28 11:03
"""

import requests

response = requests.post(
    url="http://localhost:8000/v1/models/vision/predict",
    json={
        "data": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"  # noqa
    },
)
print("Prediction:", response.json()["data"])