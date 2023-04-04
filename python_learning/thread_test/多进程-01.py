# -*-coding:utf-8 -*-

"""
# File       : 多进程-01.py
# Time       ：2023/3/31 15:52
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from multiprocessing import Pool
import os, time, random


def long_time_task(name):
    print("run task %s (%s)..." % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print("task %s (%s) run time: %s" % (name, os.getpid(), (end - start)))


if __name__ == '__main__':
    print("Parent process id: %s" % (os.getpid()))
    p = Pool(processes=5)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print("Waiting for all tasks to complete...")
    p.close()
    p.join()
    print("all tasks completed!")
