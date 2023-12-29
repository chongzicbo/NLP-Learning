#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: utils.py
@time: 2022/8/18 17:24
"""
import logging
import sys
import time
from functools import wraps

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(filename="../log.txt")


def log_filter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = 1000 * time.time()
        logger.info(f"========== Begin: {func.__name__} ==========")
        logger.info(f"Args:{kwargs}")
        try:
            rsp = func(*args, **kwargs)
            logger.info(f"Response: {rsp}")
            end = 1000 * time.time()
            logger.info(f"Time consuming: {end - start}ms")
            logger.info(f"========== End: {func.__name__} ========== \n")
            return rsp
        except Exception as e:
            logger.error(repr(e))
            raise e

    return wrapper


@log_filter
def test1():
    a = 0
    for i in range(100000):
        a += 1
    return a


if __name__ == "__main__":
    test1()
