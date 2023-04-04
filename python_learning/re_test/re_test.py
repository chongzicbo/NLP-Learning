#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: re_test.py
@time: 2022/8/11 15:08
"""

import re

text = "1p34.3 - pter microdeletion."

re_text = r"(\d+|[XY])[pq](ter)*(\+)?\d*(\.){0,1}\d*(\s*)(((-\s*>|-->|\-|\~|—|–|_|→|)|(\s*)(to)*(\s*))?(\(\d+\.*\d*\s*[-_–~]\s*\d+\.*\d*\)|)[pq]?(ter)*\d*){0,2}(\.){0,1}\d*(\s*)[pq]?\d*(\.){0,1}\d*(\(\d*(,\d*|)(,\d*|)(\s*)[_-](\s*)\d*(,\d*|)(,\d*|)\)(x\s*\d*|amp)|)"

print(re.search(re_text, text))
print(re.findall(re_text, text))


text="\na\n"
print(text)
print(re.findall("\n",text))

import re
re_str_patt = "\\\\"
reObj = re.compile(re_str_patt)
str_test = "abc\\cd\\hh"
print (reObj.findall(str_test))
import re
re_str_patt = "\\\\"
reObj = re.compile(re_str_patt)
str_test = "abc\\cd\\hh"
print (reObj.findall(str_test))

text="\\2345"
print(re.findall(r"\\|\d",text))