# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: file_find.py
@time: 2022/6/24 15:46
"""
import os
import shutil

source_dir = "E:\\working\\huada_bgi\\data\\train_data\\train_data_110\\test"  # 源目录
target_dir = "E:\\working\\huada_bgi\\data\\test_data"  # 目标目录，清单中的文件会拷贝到该目录
file_list_path = "E:\\working\\huada_bgi\\data\\test_data\\Filename-list.txt"  # 清单路径


def find_file(source_dir, target_dir):
    with open(file_list_path, encoding="utf-8") as fr:
        files = fr.readlines()
    files = [file.strip() for file in files]
    # print(files)
    for file in os.listdir(source_dir):
        if file.strip() in files:
            shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, file))
            # print(file)
        # print(file)


if __name__ == '__main__':
    find_file(source_dir, target_dir)
