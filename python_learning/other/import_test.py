import numpy as np

a = np.random.random(size=(4, 2, 3))
b = a[:, 0]
print(b.shape)

from tensorflow.python import keras
