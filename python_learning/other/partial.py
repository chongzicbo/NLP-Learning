#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: partial.py
@time: 2022/7/1 10:25
"""


def add(*args):
    return sum(args)


print(add(1, 2, 3) + 100)
print(add(5, 5, 5) + 100)


def add(*args):
    return sum(args) + 100


print(add(1, 2, 3))
print(add(5, 5, 5))

from functools import partial


def add(*args):
    return sum(args)


add_100 = partial(add, 100)
print(add_100(1, 2, 3))

add_101 = partial(add, 101)
print(add_101(1, 2, 3))
