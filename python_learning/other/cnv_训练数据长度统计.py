#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: cnv_训练数据长度统计.py
@time: 2022/10/20 11:13
"""

import re
import codecs

from transformers import BertTokenizer

model_name_or_path = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True, )


def countLength():
    trainDataPath = "./bert_wordpiece_train__fsh.txt"
    with open(trainDataPath) as f:
        f = f.read()
        lens = []
        for l in f.split('\n\n'):
            n = 0
            if not l:
                continue
            id, label = [], []
            for i, c in enumerate(l.split('\n')):
                if len(re.split("\\s", c)) != 2:
                    print("=======", c)
                    continue
                word, flag = re.split("\\s", c)
                id.append(word)
            lens.append(len(id))
    print(lens)

    print(len(lens))
    print(len([l for l in lens if l > 254]))
    print(len([l for l in lens if l > 128]))
    print(len([l for l in lens if l > 100]))
    print(len([l for l in lens if l > 64]))


"""
对单词级的数据进行wordpice级别的划分
超过 max_len 长度的句子直接进行分割
Args:
    max_len: 最大句子长度，默认126，不包括 [CLS]和[SEP]两个字符
Returns:

"""


def wordpiece():
    file_path = "/mnt/e/working/huada_bgi/data/train_data/train_data_all/train_data_all.json1"
    wordpiece_train_path = "./bert_wordpiece_train__fsh.txt"
    with codecs.open(wordpiece_train_path, "w", encoding="utf-8") as fw:
        with codecs.open(file_path) as f:
            f = f.read()
            # ids,labels=[],[]
            for l in f.split('\n\n'):
                n = 0
                if not l:
                    continue
                id, label = [], []
                for i, c in enumerate(l.split('\n')):
                    if len(re.split("\\s", c)) != 2:
                        print("=======", c)
                        continue
                    word, flag = re.split("\\s", c)
                    # if word.strip().isalpha():
                    #     fw.write(word + "\t" + flag + "\n")
                    # else:
                    tokenized_word = tokenizer.tokenize(word.lower())
                    if flag.startswith("B") and len(tokenized_word) > 1:
                        tmp = [flag]
                        for t in tokenized_word[1:]:
                            tmp.append("I-" + flag[2:])
                        flags = tmp
                    else:
                        flags = [flag] * len(tokenized_word)
                    for w, l in list(zip(tokenized_word, flags)):
                        n += 1
                        fw.write(w + "\t" + l + "\n")
                fw.write("\n")


if __name__ == '__main__':
    countLength()
