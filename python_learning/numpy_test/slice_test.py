#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: slice_test.py
@time: 2022/10/8 15:37
"""
import numpy as np

a = np.arange(1, 25)
a = a.reshape(2, 3, 4)
print(a)
print(a[:, 0])
print(a[:, 0, :])
print(a[:, :, 0])
print(a[..., 0])
print(a[None, 0])
