#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: timeout_quit.py
@time: 2022/8/16 15:33
"""
import time
import threading
import queue


def call_back_func(message):
    """运行成功回调"""
    pass


def error_back_func():
    """超时回调"""
    pass


def warp(*args, **kwargs):
    q = kwargs.pop("queue")
    f = kwargs.pop("function")
    result = f(*args, **kwargs)
    q.put(result)


def time_out(interval, call_back=None, error_back=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            q = queue.Queue()
            if "function" in kwargs:
                raise ValueError('不允许有名为"function"的参数')
            kwargs["function"] = func
            if "queue" in kwargs:
                raise ValueError('不允许有名为"queue"的参数')
            kwargs["queue"] = q
            t = threading.Thread(target=warp, args=args, kwargs=kwargs)
            t.setDaemon(True)  # 设置主线程技术子线程立刻结束
            t.start()
            try:
                result = q.get(timeout=interval)
                if call_back:
                    threading.Timer(0, call_back, args=(result,)).start()
                return result
            except queue.Empty:
                kwargs.pop("function")
                kwargs.pop("queue")
                print(f"运行超时，func:{func.__name__},args:{args}, kwargs:{kwargs}")

        return wrapper

    return decorator


@time_out(2, call_back=call_back_func, error_back=error_back_func)
def task1(name):
    print("**********task1****************")
    time.sleep(1)
    return name + "你好"


@time_out(2, call_back=call_back_func, error_back=error_back_func)
def task2(name):
    print("**********task****************")
    time.sleep(3)
    return name + "你好"


if __name__ == "__main__":
    # a = task1('小明')
    # print(a)

    b = task2("小红")
    print(b)
