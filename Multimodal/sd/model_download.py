from diffusers import StableDiffusionPipeline
import os
import torch

os.environ["XDG_CACHE_HOME"] = "/data/sshadmin/bocheng/.cache"
# https://huggingface.co/sd-dreambooth-library ，这里有来自社区的各种模型
model_id = "sd-dreambooth-library/mr-potato-head"
StableDiffusionPipeline.from_pretrained(
    model_id,
    cache_dir="/data/sshadmin/bocheng/.cache/huggingface/hub/",
    torch_dtype=torch.float16,
).to("cuda")

import os

os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/sshadmin/bocheng/.cache/huggingface/hub/"
import torch


from diffusers import StableDiffusionPipeline

# os.environ["HUGGINGFACE_HUB_CACHE"] = ""
# https://huggingface.co/sd-dreambooth-library ，这里有来自社区的各种模型

model_id = "sd-dreambooth-library/mr-potato-head"
StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
