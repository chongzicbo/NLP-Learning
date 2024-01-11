# -*-coding:utf-8 -*-

"""
# File       : pipeline_test.py
# Time       ：2023/3/14 16:17
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from diffusers import (
    DiffusionPipeline,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

repo_id = "runwayml/stable-diffusion-v1-5"

scheduler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
# or
# scheduler = DPMSolverMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")

stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, scheduler=scheduler)
