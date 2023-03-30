# -*-coding:utf-8 -*-

"""
# File       : example-02.py
# Time       ：2023/3/6 9:56
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import re

pattern = "a.c"
print(re.match(pattern, "abc"))

pattern = r"a\.c"
print(re.match(pattern, "a.c"))
pattern = r"a\\.c"
print(re.match(pattern, "a\\.c").group())

pattern = "a[bcd]e"
print(re.match(pattern, "abe"))

string = "d:\\abc\\n"
print(string)

tran_string1 = re.match("d:\\\\abc", string).group()
print(tran_string1)

tran_string2 = re.match(r"d:\\", string).group()  # 添加r表示解释器不会进行转义，只有正则表达式会进行转义
print(tran_string2)
