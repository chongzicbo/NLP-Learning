#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: test.py
@time: 2022/9/29 19:05
"""
import tensorflow as tf

x = tf.constant([5, 4, 6])
print(tf.greater(tf.cast(x, tf.float32), 0.5))
