#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: base_demo.py
@time: 2022/10/22 11:53
"""

from dialogbot import Bot

bot = Bot()

response = bot.answer("姚明多高？")
print(response)
