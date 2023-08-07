# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: custom_layer.py
@time: 2022/6/26 15:55
"""
import tensorflow as tf
from tensorflow.python import keras


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        print("enter __init__ method...")
        self.num_outputs = num_outputs

    def build(self, input_shape):
        print("enter build method...")
        self.kernel = self.add_weight(
            "kernel", shape=[int(input_shape[-1]), self.num_outputs]
        )

    def call(self, inputs):
        print("enter call method...")
        return tf.matmul(inputs, self.kernel)


# layer = MyDenseLayer(10)
# _ = layer(tf.zeros([10, 5]))
# print([var.name for var in layer.trainable_variables])


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name="")
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding="same")
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


# block = ResnetIdentityBlock(1, [1, 2, 3])
# _ = block(tf.zeros([1, 2, 3, 3]))
#
# print(block.layers)
class Linear(keras.models.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs, *args, **kwargs):
        return tf.matmul(inputs, self.w) + self.b


class Linear1(keras.models.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Linear1, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs, *args, **kwargs):
        return tf.matmul(inputs, self.w) + self.b


class Linear(keras.models.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs, *args, **kwargs):
        return tf.matmul(inputs, self.w) + self.b


class ComputeSum(keras.models.Layer):
    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        # 不可训练权重
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs, *args, **kwargs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total


class MLPBlock(keras.models.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear1 = Linear(32)
        self.linear2 = Linear(32)
        self.linear3 = Linear(1)

    def call(self, inputs):
        x = self.linear1(inputs)
        x = tf.nn.relu(x)
        x = self.linear2(x)
        x = tf.nn.relu(x)
        return self.linear3(x)


class ActivityRegularizationLayer(keras.models.Layer):
    def __init__(self, rate=1e-2):
        super(ActivityRegularizationLayer, self).__init__()
        self.rate = rate

    def call(self, inputs, *args, **kwargs):
        self.add_loss(self.rate * tf.reduce_sum(inputs))
        return inputs


class OuterLayer(keras.models.Layer):
    def __init__(self):
        super(OuterLayer, self).__init__()
        self.activity_reg = ActivityRegularizationLayer(1e-2)

    def call(self, inputs, *args, **kwargs):
        return self.activity_reg(inputs)


class OuterLayerWithKernelRegularizer(keras.layers.Layer):
    def __init__(self):
        super(OuterLayerWithKernelRegularizer, self).__init__()
        self.dense = keras.layers.Dense(
            32, kernel_regularizer=tf.keras.regularizers.l2(1e-3)
        )

    def call(self, inputs):
        return self.dense(inputs)


class LogisticEndpoint(keras.models.Layer):
    def __init__(self, name=None):
        super(LogisticEndpoint, self).__init__(name=name)
        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.accuracy_fn = keras.metrics.BinaryAccuracy()

    def call(self, targets, logits, sample_weights=None, *args, **kwargs):
        loss = self.loss_fn(targets, logits, sample_weights)
        self.add_loss(loss)

        acc = self.accuracy_fn(targets, logits, sample_weights)
        self.add_metric(acc, name="accuracy")

        return tf.nn.softmax(logits)


class Linear(keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({"units": self.units})
        return config


if __name__ == "__main__":
    x = tf.ones((2, 2))
    linear_layer = Linear1(4, 2)
    y = linear_layer(x)
    print(y)

    assert linear_layer.weights == [linear_layer.w, linear_layer.b]

    # x = tf.ones((2, 2))
    # my_sum = ComputeSum(2)
    # y = my_sum(x)
    # print(y.numpy())
    # y = my_sum(x)
    # print(y.numpy())
    #
    # print(len(my_sum.weights))
    # print(len(my_sum.non_trainable_weights))
    # print(len(my_sum.trainable_weights))
    #
    # linear_layer = Linear(32)
    # y = linear_layer(x)
    #
    # print(y)
    #
    # mlp = MLPBlock()
    # y = mlp(tf.ones(shape=(3, 64)))
    # print("weights:", len(mlp.weights))
    # print("trainable weights:", len(mlp.trainable_weights))

    # layer = OuterLayer()
    # assert len(layer.losses) == 0
    # _ = layer(tf.zeros(1, 1))
    # assert len(layer.losses) == 1
    #
    # _ = layer(tf.zeros(1, 1))
    # assert len(layer.losses) == 1
    #
    # layer = OuterLayerWithKernelRegularizer()
    # _ = layer(tf.zeros((1, 1)))
    #
    # # This is `1e-3 * sum(layer.dense.kernel ** 2)`,
    # # created by the `kernel_regularizer` above.
    # print(layer.losses)
    #
    # optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #
    # # for x_batch_train,y_batch_train in train_da
    # import numpy as np
    #
    # inputs = keras.Input(shape=(3,))
    # outputs = ActivityRegularizationLayer()(inputs)
    # model = keras.Model(inputs, outputs)
    #
    # # If there is a loss passed in `compile`, the regularization
    # # losses get added to it
    # model.compile(optimizer="adam", loss="mse")
    # model.fit(np.random.random((2, 3)), np.random.random((2, 3)))
    #
    # # It's also possible not to pass any loss in `compile`,
    # # since the model already has a loss to minimize, via the `add_loss`
    # # call during the forward pass!
    # model.compile(optimizer="adam")
    # model.fit(np.random.random((2, 3)), np.random.random((2, 3)))
    # import numpy as np
    #
    # layer = LogisticEndpoint()
    # targets = tf.ones((2, 2))
    # logits = tf.ones((2, 2))
    # y = layer(targets, logits)
    # print("layer.metrics:", layer.metrics)
    # print("current accuracy value:", float(layer.metrics[0].result()))
    #
    # inputs = keras.Input(shape=(3,), name="inputs")
    # targets = keras.Input(shape=(10,), name="targets")
    # logits = keras.layers.Dense(10)(inputs)
    # predictions = LogisticEndpoint(name="predictions")(logits, targets)
    #
    # model = keras.Model(inputs=[inputs, targets], outputs=predictions)
    # model.compile(optimizer="adam")
    #
    # data = {
    #     "inputs": np.random.random((3, 3)),
    #     "targets": np.random.random((3, 10)),
    # }
    # model.fit(data)

    layer = Linear(64)
    config = layer.get_config()
    print(config)
    new_layer = Linear.from_config(config)
