#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: test01.py
@time: 2022/9/19 9:20
"""
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
a = nltk.word_tokenize("partial Yq deletion and partial duplication of")
print(a)

text = "A de novo derivative Y chromosome (partial Yq deletion and partial duplication of Yp and Yq) in a female with disorders of sex development Key Clinical Message We report an atypical disorders of sex development (DSD) case with no mutation of SYR gene but partial Yq deletion and partial duplication of Yp and Yq. This case emphasizes duplicated region Yp11.2 Yq11.223 with partial deletion of Yq11.223 Yqter most probably perturbed the sex differentiation and led to female phenotype. "
sentences = sent_tokenize(text)
print(sentences)

sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = sentence_tokenizer.tokenize(text)
print(sentences)
