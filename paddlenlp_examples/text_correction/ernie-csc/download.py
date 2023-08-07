# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: download.py
@time: 2022/6/18 16:30
"""

import os
import sys
import argparse

from paddle.utils.download import get_path_from_url

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--data_dir", help="directory to save data to", type=str, default="./"
)
parser.add_argument(
    "-u",
    "--url",
    help="URL of target",
    type=str,
    default="https://bj.bcebos.com/paddlenlp/datasets/sighan_test.zip",
)
args = parser.parse_args()


def main():
    get_path_from_url(args.url, args.data_dir)


if __name__ == "__main__":
    sys.exit(main())
