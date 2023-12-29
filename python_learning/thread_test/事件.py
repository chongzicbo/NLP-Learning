# -*-coding:utf-8 -*-

"""
# File       : 事件.py
# Time       ：2023/3/31 15:30
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import time
import threading

ExamBegEvent = threading.Event()


def student(id):
    print("学生[{}]等待考试开始".format(id))
    ExamBegEvent.wait()
    print("学生[{}]开始考试".format(id))


def teacher():
    print("老师：开始考试".format(id))
    ExamBegEvent.set()


for i in range(6):
    threading.Thread(target=student, args=(i,)).start()
time.sleep(3)
threading.Thread(target=teacher).start()
