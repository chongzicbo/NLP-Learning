#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: main.py
@time: 2022/10/8 17:32
"""
import sys
from chineseproofread import proofread
from checkproof import proofcheck
import importlib


def main():
    ptarget = proofread()
    ptarget.proofreadAndSuggest("天汽")


if __name__ == "__main__":
    importlib.reload(sys)
    sys.setdefaultencoding('utf-8')
    main()
