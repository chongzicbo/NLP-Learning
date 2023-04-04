# -*-coding:utf-8 -*-

"""
# File       : 信号量.py
# Time       ：2023/3/31 14:58
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from concurrent.futures import ThreadPoolExecutor
import threading
import time


def action(max):
    for i in range(max):
        print(threading.current_thread().name + " " + str(i))


beg=time.perf_counter()
futures=[]
with   ThreadPoolExecutor(max_workers=3) as executor:
    for i in range(6):
        futures.append(executor.submit(action,5))
for i in futures:
    i.result()
end=time.perf_counter()

print("time use: ",end-beg)
print("----------")


## 信号量使用
sem = threading.Semaphore(3)  # 定义一个有三个信号的信号量
# 定义一个准备作为线程任务的函数
def action2(max):
    sem.acquire()  # 需要手动获得信号
    for i in range(max):
        print(threading.current_thread().name + '  ' + str(i))
        time.sleep(0.1)
    sem.release()  # 需要手动释放信号

beg = time.perf_counter()
# 创建6个线程，都开始
threads = []
for i in range(6):
    t = threading.Thread(target=action2,args=(5,))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
end = time.perf_counter()
print("time use: ",end-beg)