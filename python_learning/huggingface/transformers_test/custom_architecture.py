# -*-coding:utf-8 -*-

"""
# File       : custom_architecture.py
# Time       ：2023/3/13 11:22
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from transformers import DistilBertConfig,DistilBertModel

config = DistilBertConfig()
print(config)

myconfig = DistilBertConfig(activation='relu', attention_dropout=0.4)
print(myconfig)
model=DistilBertModel.from_pretrained("distilbert-base-uncased",config=myconfig)
print(model)

from transformers import ViTImageProcessor
vit_extrator=ViTImageProcessor()
print(vit_extrator)
my_vit_extractor = ViTImageProcessor(resample="PIL.Image.BOX", do_normalize=False, image_mean=[0.3, 0.3, 0.3])

