# -*-coding:utf-8 -*-

"""
# File       : ttskit_test.py
# Time       ：2023/5/30 10:12
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from ttskit import sdk_api

wav = sdk_api.tts_sdk('文本', audio='24')