# -*-coding:utf-8 -*-

"""
# File       : with_test.py
# Time       ：2023/2/9 16:02
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""


class Sample:
    def __enter__(self):
        print("in __enter__")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("in __exit__")
        print("type:", exc_type)
        print("val:", exc_val)
        print("tb", exc_tb)

    def do_something(self):
        bar = 1 / 0
        return bar + 10


def get_sample():
    return Sample()


with get_sample() as sample:
    print("Sample:", sample)
    sample.do_something()
