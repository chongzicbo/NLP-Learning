#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: keras_pruning_comprehensive.py
@time: 2022/11/2 19:54

https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide?hl=zh-cn#modelfit

"""

import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot

import tempfile

input_shape = [20]
x_train = np.random.randn(1, 20).astype(np.float32)
y_train = tf.keras.utils.to_categorical(np.random.randn(1), num_classes=20)


def setup_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, input_shape=input_shape),
        tf.keras.layers.Flatten()
    ])
    return model


def setup_pretrained_weights():
    model = setup_model()

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy']
    )

    model.summary()
    model.fit(x_train, y_train)

    _, pretrained_weights = tempfile.mkstemp('.tf')

    model.save_weights(pretrained_weights)

    return pretrained_weights


def get_gzipped_model_size(model):
    # Returns size of gzipped model, in bytes.
    import os
    import zipfile

    _, keras_file = tempfile.mkstemp('.h5')
    model.save(keras_file, include_optimizer=False)

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(keras_file)

    return os.path.getsize(zipped_file)


setup_model()
pretrained_weights = setup_pretrained_weights()

base_model = setup_model()
base_model.load_weights(pretrained_weights).expect_partial()  # optional but recommended.

model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

model_for_pruning.summary()

# Create a base model
base_model = setup_model()
base_model.load_weights(pretrained_weights).expect_partial()  # optional but recommended for model accuracy


# Helper function uses `prune_low_magnitude` to make only the
# Dense layers train with pruning.
def apply_pruning_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer


# Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense`
# to the layers of the model.
model_for_pruning = tf.keras.models.clone_model(
    base_model,
    clone_function=apply_pruning_to_dense,
)

model_for_pruning.summary()

print(base_model.layers[0].name)
print("开始自定义训练")
# 自定义训练循环
base_model = setup_model()
base_model.load_weights(pretrained_weights).expect_partial()
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

# Boilerplate
loss = tf.keras.losses.categorical_crossentropy
optimizer = tf.keras.optimizers.Adam()
log_dir = tempfile.mkdtemp()
unused_arg = -1
epochs = 2
batches = 1

model_for_pruning.optimizer = optimizer
step_callback = tfmot.sparsity.keras.UpdatePruningStep()
step_callback.set_model(model_for_pruning)
log_callback = tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
log_callback.set_model(model_for_pruning)
step_callback.on_train_begin()

for _ in range(epochs):
    log_callback.on_epoch_begin(epoch=unused_arg)
    for _ in range(batches):
        step_callback.on_train_batch_begin(batch=unused_arg)
        with tf.GradientTape() as tape:
            logits = model_for_pruning(x_train, training=True)
            loss_value = loss(y_train, logits)
            print("损失为%s" % loss_value.numpy())
            grads = tape.gradient(loss_value, model_for_pruning.trainable_variables)
            optimizer.apply_gradients(zip(grads, model_for_pruning.trainable_variables))
    step_callback.on_epoch_end(batch=unused_arg)

print("自定义训练完成")

# 模型剪枝后压缩
# Define the model.
base_model = setup_model()
base_model.load_weights(pretrained_weights).expect_partial()  # optional but recommended for model accuracy
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model)

# Typically you train the model here.

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

print("final model")
model_for_export.summary()

print("\n")
print("Size of gzipped pruned model without stripping: %.2f bytes" % (get_gzipped_model_size(model_for_pruning)))
print("Size of gzipped pruned model with stripping: %.2f bytes" % (get_gzipped_model_size(model_for_export)))
