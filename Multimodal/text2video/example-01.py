# -*-coding:utf-8 -*-

"""
# File       : example-01.py
# Time       ：2023/6/12 11:33
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import retrying
from diffusers import DiffusionPipeline
from diffusers import TextToVideoSDPipeline
from diffusers.utils import export_to_video
@retrying.retry
def get_model():
	pipeline = DiffusionPipeline.from_pretrained("vdo/animov-512x")
	return pipeline


if __name__ == '__main__':
	pipeline = get_model()
	pipe = pipeline.to("cuda")
	video_frames=pipe('a cute cat').frames
	video_path = export_to_video(video_frames)
	print(video_path)