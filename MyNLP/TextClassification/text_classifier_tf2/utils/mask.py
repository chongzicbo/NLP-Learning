#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: mask.py
@time: 2022/10/31 9:52
"""
import tensorflow as tf


def create_padding_mask(seq):
    """
    生成mask，mask值为1
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # [batch_size, 1, 1, seq_len]
    # 执行attention计算时，attention_matrix=[batch_size, num_head, seq_len_q, seq_len_k]
    return seq[:, tf.newaxis, tf.newaxis, :]

