#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: clean_data.py
@time: 2022/10/31 10:53
"""
import re


def filter_word(raw_word):
    if raw_word.strip() in ['\t', '']:
        return False
    if not re.search(r'^[\u4e00-\u9fa5_a-zA-Z\d]+$', raw_word):
        return False
    else:
        return True


def filter_char(char, remove_sp=True):
    if char.strip() in ['\t', '']:
        return False
    if remove_sp:
        if re.search(r'[\u4e00-\u9fa5_a-zA-Z\d]', char):
            return True
        else:
            return False
    else:
        return True
