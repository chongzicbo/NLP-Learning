# -*-coding:utf-8 -*-

"""
# File       : threading_test.py
# Time       ：2023/3/31 14:11
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import time
from threading import Thread


def task():
    print("开始做一个任务")
    time.sleep(1)
    print("这个任务结束啦")


class NewThread(Thread):
    def __init__(self):
        Thread.__init__(self)

    def run(self) -> None:
        print("开始做一个任务啦")
        time.sleep(1)
        print("这个任务结束啦")


def task1():
    print("开始做任务1啦")
    time.sleep(1)
    print("任务1结束啦")

def task2():
    print("开始做任务2��")
    for i in range(5):
        print("任务2-{}".format(i))
        time.sleep(1)
    print("任务2结束��")

def task3():
    print("开始做任务1啦")
    time.sleep(3)  # 用time.sleep模拟任务耗时
    print("任务1结束啦")


if __name__ == '__main__':
    print("这里是主进程")
    t3=Thread(target=task3)
    t3.start()
    t3.join()#让主线程等待
    time.sleep(0.3)
    print("主线程结束")
