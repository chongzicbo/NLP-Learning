#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: 超时退出.py
@time: 2022/8/15 16:48
"""
import functools
import concurrent
from concurrent import futures

executor = futures.ThreadPoolExecutor(1)


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            future = executor.submit(func, *args, **kw)
            return future.result(timeout=seconds)

        return wrapper

    return decorator


import time


@timeout(1)
def task(a, b):
    time.sleep(1.2)
    executor.shutdown()
    return a + b


try:
    task(2, 3)
except concurrent.futures._base.TimeoutError:
    print("你好")
