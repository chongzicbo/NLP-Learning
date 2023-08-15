import timm
import torch

# models=timm.list_models(pretrained=True)
# print(models)

import towhee
from towhee import pipeline

embedding_pipeline = pipeline("towhee/image-embedding-swinbase")
image_path = "/home/bocheng/data/images/articles/pharmaceuticals-16-00117/test/pharmaceuticals-16-00117-g005.png"
embedding = embedding_pipeline(image_path)
print(len(embedding))
