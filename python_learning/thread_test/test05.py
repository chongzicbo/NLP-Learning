#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: test05.py
@time: 2022/10/14 16:26
"""

import os
from multiprocessing import Process

print("Process (%s) start..." % os.getpid())

# pid = os.fork()
# if pid == 0:
#     print('I am child process (%s) and my parent is %s.' % (os.getpid(), os.getppid()))
# else:
#     print('I (%s) just created a child process (%s).' % (os.getpid(), pid))


def run_proc(name):
    print("Run child process %s (%s)..." % (name, os.getpid()))


if __name__ == "__main__":
    print("Parent process %s." % os.getpid())
    p = Process(target=run_proc, args=("test",))
    print("Child process will start.")
    p.start()
    p.join()
    print("Child process end.")
