#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: test01.py
@time: 2022/8/10 14:20
"""

import time
from threading import Thread


def task():
    print("开始做一个任务")
    time.sleep(1)
    print("这个任务结束啦")


def task1():
    print("开始做任务1啦")
    time.sleep(1)
    print("任务1结束啦")


def task2():
    print("开始做任务2啦")
    for i in range(5):
        print("任务2-{}".format(i))
        time.sleep(1)
    print("任务2结束啦")


class NewThread(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        print("开始做一个任务啦")
        time.sleep(1)
        print("这个任务结束啦")


if __name__ == "__main__":
    print("这里是主线程")
    # t1 = Thread(target=task)
    # t1.start()
    # time.sleep(0.3)
    # t1 = NewThread()
    # t1.start()
    # time.sleep(0.3)

    # t1 = Thread(target=task1)
    # t2 = Thread(target=task2)
    # t2.setDaemon(True)
    #
    # t1.start()
    # t2.start()
    # time.sleep(0.3)

    t1 = Thread(target=task1)
    t1.start()
    t1.join()

    print("主线程结束了")

    # print("主线程依然可以干别的事")
