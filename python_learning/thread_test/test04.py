#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: test04.py
@time: 2022/10/14 15:42
"""

from multiprocessing import Process

import time


def add(procname, n):
    sum = 0
    while sum <= n:
        sum += 1
    print(f"process name:{procname}")
    print(f"sum:{sum}")


if __name__ == '__main__':
    start = time.time()
    n = 500000000
    p1 = Process(target=add, args=("Proc-1", n // 2))
    p2 = Process(target=add, args=("Proc-2", n // 2))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print("run time:%s " % str(time.time() - start))
