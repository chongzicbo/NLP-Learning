import os

# 修改环境变量要在导入datasets或者transformers模块之前
os.environ["XDG_CACHE_HOME"] = "/data/sshadmin/bocheng/.cache"
import torchvision
from datasets import load_dataset
from torchvision import transforms

dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
