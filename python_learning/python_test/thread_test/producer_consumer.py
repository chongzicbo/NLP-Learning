# -*-coding:utf-8 -*-

"""
# File       : producer_consumer.py
# Time       ：2023/3/31 14:35
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import time, random
from threading import Thread, currentThread
from queue import Queue, Empty

foods = (
    "蒸羊羔",
    "蒸熊掌",
    "蒸鹿尾儿",
    "烧花鸭",
    "烧雏鸡",
    "烧子鹅",
    "卤猪",
    "卤鸭",
    "酱鸡",
    "腊肉",
    "松花",
    "小肚儿",
    "晾肉",
    "香肠",
    "什锦苏盘",
)  # 食物列表


def producer(queue: Queue):
    print("[{}]厨师来了".format(currentThread().name))
    global foods
    for i in range(10):
        food = random.choice(foods)
        print("[{}]正在加工：{}".format(currentThread().name, food))
        time.sleep(0.8)
        print("[{}]上菜了".format(currentThread().name))
        queue.put(food)


def consumer(queue: Queue):
    print("[{}]客人来了".format(currentThread().name))
    while True:
        try:
            food = queue.get(timeout=0.5)
            print("[{}]正在享受美食：{}".format(currentThread().name, food))
            time.sleep(0.5)
            print("[{}]下菜了".format(currentThread().name))
        except Empty:
            print("没菜吃了，[{}]走了".format(currentThread().name))
            break


if __name__ == "__main__":
    queue = Queue()
    pds = []
    csm = []
    for i in range(4):
        t = Thread(target=producer, args=(queue,))
        t.start()
        pds.append(t)

    time.sleep(1)
    for i in range(2):
        t = Thread(target=consumer, args=(queue,))
        t.start()
        csm.append(t)
