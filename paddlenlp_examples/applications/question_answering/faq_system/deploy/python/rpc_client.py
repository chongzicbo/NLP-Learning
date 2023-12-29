#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: rpc_client.py
@time: 2022/7/27 19:47
"""
import time
import numpy as np

from paddle_serving_server.pipeline import PipelineClient

client = PipelineClient()
client.connect(["127.0.0.1:8080"])

list_data = ["湖北省为什么鼓励缴费人通过线上缴费渠道缴费？", "佛山市救助站有多少个救助床位"]
feed = {}
for i, item in enumerate(list_data):
    feed[str(i)] = item

print(feed)
start_time = time.time()
ret = client.predict(feed_dict=feed)
end_time = time.time()
print("time to cost :{} seconds".format(end_time - start_time))

result = np.array(eval(ret.value[0]))
print(ret.key)
print(result.shape)
print(result)
