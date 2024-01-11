#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: custom_loss.py
@time: 2022/8/19 11:02
"""
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python import keras
import tensorflow as tf


class CrossEntropy(Layer):
    """特殊的层，用来定义复杂loss
    继承Layer的方式自定义损失
    """

    def __init__(self, output_axis=None, **kwargs):
        super(CrossEntropy, self).__init__(**kwargs)
        self.output_axis = output_axis

    def call(self, inputs, mask=None):
        loss = self.compute_loss(inputs, mask)
        self.add_loss(loss, inputs=inputs)
        if self.output_axis is None:
            return inputs
        elif isinstance(self.output_axis, list):
            return [inputs[i] for i in self.output_axis]
        else:
            return inputs[self.output_axis]

    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name="accuracy")
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def compute_output_shape(self, input_shape):
        if self.output_axis is None:
            return input_shape
        elif isinstance(self.output_axis, list):
            return [input_shape[i] for i in self.output_axis]
        else:
            return input_shape[self.output_axis]

    def compute_mask(self, inputs, mask):
        if mask is not None:
            if self.output_axis is None:
                return mask
            elif isinstance(self.output_axis, list):
                return [mask[i] for i in self.output_axis]
            else:
                return mask[self.output_axis]

    def get_config(self):
        config = {
            "output_axis": self.output_axis,
        }
        base_config = super(CrossEntropy, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


# https://github.com/mkocabas/focal-loss-keras/blob/master/focal_loss.py
# 装饰器的方式实现复杂损失函数：focal_loss
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(
            alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1 + K.epsilon())
        ) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0 + K.epsilon()))
