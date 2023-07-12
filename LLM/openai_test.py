# -*-coding:utf-8 -*-

"""
# File       : openai_test.py
# Time       ：2023/6/8 14:20
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import os
import openai
openai.organization = "org-p1SkoHd7HEWYaP5gYqNXPPYf"
openai.api_key = os.getenv("OPENAI_API_KEY","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")
openai.Model.list()