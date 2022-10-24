# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# import torch
#
# a = torch.tensor([1, 2, 3])
# mask = torch.tensor([True,True,True])
# print(a)
# print(mask)
# print(a[mask])

# import numpy as np
#
# a = np.random.randint(low=1, high=100, size=(2, 3, 5))
# b = a[:, 1, [2, 4]]
# print(a)
# print(b)
# print(np.argmax(b, axis=1))
#
# import tensorflow as tf
# tf.config.run_functions_eagerly()
import numpy as np
start=[(1,4),(6,9),()]
end=[]
start = np.random.choice(start)  # 如果有多个subject ，start
end = np.random.choice(end[end >= start])  # 随机选择一个end