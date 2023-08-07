# -*-coding:utf-8 -*-

"""
# File       : 线程池.py
# Time       ：2023/3/31 14:47
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from concurrent.futures import ThreadPoolExecutor
import threading
import time


def action(max):
    my_sum = 0
    for i in range(max):
        print(threading.current_thread().name + " " + str(i))
        my_sum += i
    return my_sum


# #创建一个包含两个线程的线程池
# pool=ThreadPoolExecutor(2)
# #向线程池再提交一个task
# future1=pool.submit(action,50)
# future2=pool.submit(action,100)
# print(future1.done())
# time.sleep(3)
#
# print(future2.done())
# print(future1.result())
# print(future2.result())
# pool.shutdown()

# 创建一个包含2条线程的线程池
# 线程池支持上下文管理协议，用with避免忘记写shutdown方法

with ThreadPoolExecutor(2) as pool:
    # 后面元组有3个元素，因此程序启动3次线程来执行action函数
    results = pool.map(action, (50, 100, 150))
    print("------------------")
    for result in results:
        print(result)
