#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: word2vec.py
@time: 2022/10/31 10:38
"""
import sys

sys.path.append("../../text_classifier_tf2")

from config import word2vec_config
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
import jieba
import multiprocessing
import pandas as pd
import os


class Word2VecUtils:
    def __init__(self, logger):
        self.logger = logger
        self.stop_words = word2vec_config['stop_words']
        self.train_data = word2vec_config['train_data']
        model_dir = word2vec_config['model_dir']
        model_name = word2vec_config['model_name']
        self.model_path = os.path.join(model_dir, model_name)
        self.dim = word2vec_config['word2vec_dim']
        self.min_count = word2vec_config['min_count']
        # skip-gram(sg=1)是CBOW(sg=0)训练时间的2.5 ～3倍左右
        self.sg = 1 if word2vec_config['sg'] == 'skip-gram' else 0
        # 使用分级softmax(1)是采用负采样(0)训练时间的1.8～2倍左右
        self.hs = 1

    @staticmethod
    def processing_sentence(x, stop_words):
        cut_word = jieba.cut(str(x).strip())
        if stop_words:
            words = [word for word in cut_word if word not in stop_words and word != ' ']
        else:
            words = list(cut_word)
            words = [word for word in words if word != ' ']
        return words

    def get_stop_words(self):
        stop_words_list = []
        try:
            with open(self.stop_words, 'r', encoding='utf-8') as stop_words_file:
                for line in stop_words_file:
                    stop_words_list.append(line.strip())
        except FileNotFoundError:
            return stop_words_list
        return stop_words_list

    def train_word2vec(self):
        train_df = pd.read_csv(self.train_data, encoding='utf-8')
        stop_words = self.get_stop_words()
        # 切词
        self.logger.info('Cut sentence...')
        tqdm.pandas(desc='Cut sentence...')
        train_df['sentence'] = train_df.sentence.progress_apply(self.processing_sentence, args=(stop_words,))
        # 删掉缺失的行
        train_df.dropna(inplace=True)
        all_cut_sentence = train_df.sentence.to_list()
        # 训练词向量
        self.logger.info('Training word2vec...')
        w2v_model = Word2Vec(size=self.dim, workers=multiprocessing.cpu_count(), min_count=self.min_count, sg=self.sg)
        w2v_model.build_vocab(all_cut_sentence)
        w2v_model.train(all_cut_sentence, total_examples=w2v_model.corpus_count, epochs=100)
        w2v_model.save(self.model_path)