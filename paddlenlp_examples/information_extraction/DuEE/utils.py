# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: utils.py
@time: 2022/6/23 13:44
"""

import hashlib


def cal_md5(str):
    """calculate string md5"""
    str = str.decode("utf-8", "ignore").encode("utf-8", "ignore")
    return hashlib.md5(str).hexdigest()


def read_by_lines(path):
    """read the data by line"""
    result = list()
    with open(path, "r", encoding="utf8") as infile:
        for line in infile:
            result.append(line.strip())
    return result


def write_by_lines(path, data):
    """write the data"""
    with open(path, "w", encoding="utf8") as outfile:
        [outfile.write(d + "\n") for d in data]


def text_to_sents(text):
    """text_to_sents"""
    deliniter_symbols = ["。", "？", "！"]
    paragraphs = text.split("\n")
    ret = []
    for para in paragraphs:
        if para == "":
            continue
        sents = [""]
        for s in para:
            sents[-1] += s
            if s in deliniter_symbols:
                sents.append("")
        if sents[-1] == "":
            sents = sents[:-1]
        ret.extend(sents)
    return ret


def load_dict(dict_path):
    """load_dict"""
    vocab = {}
    for line in open(dict_path, "r", encoding="utf-8"):
        value, key = line.strip("\n").split("\t")
        vocab[key] = int(value)
    return vocab


def extract_result(text, labels):
    """extract_result"""
    ret, is_start, cur_type = [], False, None
    if len(text) != len(labels):
        # 韩文回导致label 比 text要长
        labels = labels[: len(text)]
    for i, label in enumerate(labels):
        if label != "O":
            _type = label[2:]
            if label.startswith("B-"):
                is_start = True
                cur_type = _type
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif _type != cur_type:
                """
                # 如果是没有B-开头的，则不要这部分数据
                cur_type = None
                is_start = False
                """
                cur_type = _type
                is_start = True
                ret.append({"start": i, "text": [text[i]], "type": _type})
            elif is_start:
                ret[-1]["text"].append(text[i])
            else:
                cur_type = None
                is_start = False
        else:
            cur_type = None
            is_start = False
    return ret


if __name__ == "__main__":
    s = "xxdedewd"
    print(cal_md5(s.encode("utf-8")))
