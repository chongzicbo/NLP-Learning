#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: tf_function.py
@time: 2022/9/26 15:57
"""

import tensorflow as tf
import timeit
from datetime import datetime


def a_regular_function(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x


a_function_that_uses_a_graph = tf.function(a_regular_function)

x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

orig_value = a_regular_function(x1, y1, b1).numpy()
tf_function_value = a_function_that_uses_a_graph(x1, y1, b1).numpy()

assert orig_value == tf_function_value


def inner_function(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x


# 使用tf.function将`outer_function`变成一个tensorflow `Function`。注意，之前的代码是将tf.function当作是函数来使用，这样是被当作了修饰符来使用。这两种方式都是被支持的。
@tf.function
def outer_function(x):
    y = tf.constant([[2.0], [3.0]])
    b = tf.constant(4.0)

    return inner_function(x, y, b)


# tf.function构建的graph中不仅仅包含了 `outer_function`还包含了它里面调用的`inner_function`。
print(outer_function(tf.constant([[1.0, 2.0]])).numpy())


@tf.function
def my_relu(x):
    return tf.maximum(0.0, x)


# 下面对`my_relu` 的3次调用的数据类型都不同，所以生成了3个graph。这3个graph都被保存在my_relu这个tenforflow function中。
print(my_relu(tf.constant(5.5)))
print(my_relu([1, -1]))  # python数组
print(my_relu(tf.constant([3.0, -3.0])))  # tf数组

# 下面这两个调用就不会生成新的graph.
print(my_relu(tf.constant(-2.5)))  # 这个数据类型与上面的 `tf.constant(5.5)`一样.
print(my_relu(tf.constant([-1.0, 1.0])))  # 这个数据类型与上面的 `tf.constant([3., -3.])`一样。
