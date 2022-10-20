#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: excel_file.py
@time: 2022/7/22 14:57
"""
import xlrd

filename = "E:\\working\\huada_bgi\\data\\other_data\\Dataset_CNV Disease_V1.0(1).xlsx"

data = xlrd.open_workbook(filename)
table = data.sheets()[0]
nrows = table.nrows  # 获取sheet行数
ncols = table.ncols  # 获取sheey列数
print(nrows, ncols)
