#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: keras_pruning_inference_example_03.py
@time: 2022/11/3 11:06
"""
import tempfile

import tensorflow as tf
import numpy as np

from tensorflow import keras
import tensorflow_datasets as tfds
import tensorflow_model_optimization as tfmot

(ds_train, ds_val, ds_test), ds_info = tfds.load('cifar10',
                                                 split=['train[:90%]', 'train[90%:]', 'test'],
                                                 as_supervised=True,
                                                 with_info=True, )


def normalize_img(image, label):
    return tf.image.convert_image_dtype(image, tf.float32), label


# for x in ds_train:
#     print(x)
#     break

batch_size = 128


def prepare_dataset(ds, buffer_size=None):
    ds = ds.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.cache()
    if buffer_size:
        ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

ds_train = prepare_dataset(ds_train,
                           buffer_size=ds_info.splits['train'].num_examples)
ds_val = prepare_dataset(ds_val)
ds_test = prepare_dataset(ds_test)

# Build the dense baseline model.
dense_model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(32, 32, 3)),
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid'),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Conv2D(filters=16, kernel_size=(1, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3), strides=(2, 2), padding='valid'),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Conv2D(filters=32, kernel_size=(1, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])

# Compile and train the dense model for 10 epochs.
dense_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

dense_model.fit(
    ds_train,
    epochs=2,
    validation_data=ds_val)

# Evaluate the dense model.
_, dense_model_accuracy = dense_model.evaluate(ds_test, verbose=0)

print(dense_model_accuracy)
#
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
#
# # Compute end step to finish pruning after after 5 epochs.
end_epoch = 5

num_iterations_per_epoch = len(ds_train)
end_step =  num_iterations_per_epoch * end_epoch

# Define parameters for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.25,
                                                               final_sparsity=0.75,
                                                               begin_step=0,
                                                               end_step=end_step),
      'pruning_policy': tfmot.sparsity.keras.PruneForLatencyOnXNNPack()
}
#
# # Try to apply pruning wrapper with pruning policy parameter.
# try:
#   model_for_pruning = prune_low_magnitude(dense_model, **pruning_params)
# except ValueError as e:
#   print(e)
#
fixed_dense_model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(32, 32, 3)),
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(
        filters=8,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding='valid'),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Conv2D(filters=16, kernel_size=(1, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3), strides=(2, 2), padding='valid'),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.Conv2D(filters=32, kernel_size=(1, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.ReLU(),
    keras.layers.GlobalAveragePooling2D(keepdims=True),
    keras.layers.Flatten(),
    keras.layers.Dense(10)
])

# Use the pretrained model for pruning instead of training from scratch.
fixed_dense_model.set_weights(dense_model.get_weights())

# Try to reapply pruning wrapper.
model_for_pruning = prune_low_magnitude(fixed_dense_model, **pruning_params)

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

model_for_pruning.fit(
  ds_train,
  epochs=15,
  validation_data=ds_val,
  callbacks=callbacks)

# Evaluate the dense model.
_, pruned_model_accuracy = model_for_pruning.evaluate(ds_test, verbose=0)

print('Dense model test accuracy:', dense_model_accuracy)
print('Pruned model test accuracy:', pruned_model_accuracy)