# -*-coding:utf-8 -*-

"""
# File       : 线程共享资源.py
# Time       ：2023/3/31 14:24
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import time
from threading import Thread, Lock

lock = Lock()

n = 0


def task1():
    global n
    global lock
    lock.acquire()
    for i in range(800000):  # 将n循环加800000
        n += 1
    lock.release()


def task2():
    global n
    lock.acquire()
    print("n is {}".format(n))  # 访问n
    lock.release()


if __name__ == "__main__":
    print("这里是主线程")
    # 创建线程对象
    t1 = Thread(target=task1)
    t2 = Thread(target=task2)
    # 启动
    t1.start()
    t2.start()
    time.sleep(0.3)
    print("主线程结束了")
