# -*-coding:utf-8 -*-

"""
# File       : pathlib_test.py
# Time       ：2023/3/7 10:30
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from pathlib import Path
import os.path

# 老方式
two_dirs_up = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 新方式，可读性强
two_dirs_up = Path(__file__).resolve().parent.parent
print(two_dirs_up)
