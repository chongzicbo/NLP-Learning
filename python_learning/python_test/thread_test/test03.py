#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: test03.py
@time: 2022/10/14 15:31
"""
import threading
import time


def add(n):
    sum = 0
    while sum <= n:
        sum += 1
    print(f"sum:{sum}")


def add(n):
    sum = 0
    while sum <= n:
        sum += 1
    print(f"sum={sum}")


if __name__ == "__main__":
    start = time.time()
    # add(500000000)
    n = 500000000
    t1 = threading.Thread(target=add, args=[n // 2])
    t2 = threading.Thread(target=add, args=[n // 2])
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print("run time:%s" % str(time.time() - start))
