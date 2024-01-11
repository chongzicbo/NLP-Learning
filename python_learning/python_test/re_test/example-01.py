# -*-coding:utf-8 -*-

"""
# File       : example-01.py
# Time       ：2023/3/3 11:35
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
import re

input_text = "Hello,my name is Ben.Please visit my website at http://www.forta.com/"
pattern = "Hello"
pattern1 = "my"

# re.match()只会匹配目标字符串开头是否满足正则表达式，若开头不满足则匹配失败，函数返回None
print(re.match(pattern, input_text).group())
print(re.match(pattern1, input_text))

# findall找出文本中包含的所有匹配的结果
print(re.findall(pattern1, input_text))

# 搜索第一个匹配的结果
print(re.search(pattern1, input_text))

text = "生于杭州市余杭区，成长经历地杭州市余杭区，居住较长的地区杭州市余杭区69年，无疫区居住史，无冶游史，有饮酒习惯，酒类：白酒，每天500-600ml，已饮30年，未戒。，有吸烟习惯，种类：纸烟，每天20-30支，已吸30年，已戒6年。，否认毒物及放射性物质接触史。既往在矿场工作。"

# 使用小括号“()”的作用在这里是捕获分组的意思,也就是在正则匹配的时候，将小括号中匹配到的文本编号并存储到内存中以供后续使用
pattern = r"(有吸烟习惯)[，,]种类[：:]纸烟[，,](每[天日周月]\d{1,2}-\d{1,2}[支根])[，,]已吸(\d{1,2}年)[，,](已戒)(\d{1,2}年)。"
result = re.search(pattern, text)
print(result.group(0))  # group()方法是用来获得相应的分组结果, “.group()”和“.group(0)”是完全一样的。
print(type(result))
print(result[1], result.group(1))  # group(1),表示第一个括号对应的内容
print(type(result[1]))
print(result[2])
print(result[3])
print(result[4])
print(result[5])
print(result.group())
print(type(result.group()))
print(result.groups())
print(type(result.groups()))
print(result.group(1))
print(type(result.group(1)))
print(result.group(2))
print(result.group(3))
print(result.group(4))
print(result.group(5))

print("==========================================")
line = "Cats are smarter than dogs"

matchObj = re.match(r"(.*) are (.*?) .*", line, re.M | re.I)

if matchObj:
    print("matchObj.group() : ", matchObj.group())

    print("matchObj.group(1) : ", matchObj.group(1))

    print("matchObj.group(2) : ", matchObj.group(2))

else:
    print("No match!!")

phone = "2004-959-559 # 这是一个国外电话号码"
num = re.sub(r"#.*$", "", phone)
print("电话号码是：", num)

num = re.sub(r"\D", "", phone)
print("电话号码是:", num)


def double(matched: re.Match):
    value = int(matched.group("value"))
    return str(value * 2)


s = "A23G4HFD567"
# print(re.sub('(?P<value>\d+)'), double, s)
# print(re.sub('(\d+)', double, s))
print(re.sub("(?P<value>\d+)", double, s))  # 为分组指定别名
pattern = re.compile(r"\d+")
m = pattern.match("one12twothree34four", 3)
print(m.group())
text = "runoob,89 we 1 runoob, runoob. +"
print(re.split("[^\d]", text))
print(re.findall("[a-z]+", text))
print(re.findall("[a-z+]", text))
print(re.findall(r"[a-z\\+]", text))

print("=================================")
print(re.findall("ru|[+]", text))
print(re.findall("(?P<v>\d+)", text))
print(re.findall("(?P<value>\d+)", s))
print(re.findall("\d+", s))

print("=====================")

text = "abcabcabcabc"
print(re.findall("(?P<id>\w+)c(?P=id)", text))

print(re.findall("(?:abc)", text))

print("========================================")
txt = "011002200811"
print(re.findall("(?:(?<!\d)\d{12}(?!\d))", txt))
txt = "011002200811"
print(re.findall("(?:(?<!\d)\d{12}(?!\d))", txt))
