# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: download.py
@time: 2022/6/24 14:54
"""

import os
import sys
import argparse

from paddle.utils.download import get_path_from_url

URL = "https://bj.bcebos.com/paddlenlp/datasets/lexical_analysis_dataset_tiny.tar.gz"


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--data_dir',
                        help='directory to save data to',
                        type=str,
                        default='data')
    args = parser.parse_args(arguments)
    get_path_from_url(URL, args.data_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
