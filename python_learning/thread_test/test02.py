#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: test02.py
@time: 2022/8/10 14:42
"""
import time
from threading import Thread, Lock

lock = Lock()  # 创建锁对象
n = 0


def task1():
    global n
    global lock
    lock.acquire()
    for i in range(800000):
        n += 1
    lock.release()


def task2():
    global n
    lock.acquire()
    print("n is {}".format(n))
    lock.release()


if __name__ == '__main__':
    print("这里是主线程")
    t1 = Thread(target=task1)
    t2 = Thread(target=task2)
    t1.start()
    t2.start()
    print("main: n is {}".format(n))
    time.sleep(0.3)
    print("主线程结束啦")
