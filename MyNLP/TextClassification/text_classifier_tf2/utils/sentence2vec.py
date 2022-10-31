#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: sentence2vec.py
@time: 2022/10/31 10:39
"""
import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
from utils.word2vec import Word2VecUtils
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm


class Sentence2VecUtils:
    def __init__(self, logger):
        self.w2v_utils = Word2VecUtils(logger)
        self.w2v_model = Word2Vec.load(self.w2v_utils.model_path)
        model_dir = word2vec_config['model_dir']
        self.pca_vec_path = os.path.join(model_dir, 'pca_u.npy')
        self.count_num = 0
        for key, value in self.w2v_model.wv.vocab.items():
            self.count_num += value.count
        self.logger = logger
        self.a = 1e-3
        self.u = None

    def calculate_weight(self, sentence):
        vs = np.zeros(self.w2v_utils.dim)  # add all word2vec values into one vector for the sentence
        sentence_length = len(sentence)
        for word in sentence:
            if word in self.w2v_model.wv.vocab:
                p_w = self.w2v_model.wv.vocab[word].count / self.count_num
                a_value = self.a / (self.a + p_w)  # smooth inverse frequency, SIF
                vs = np.add(vs, np.multiply(a_value, np.array(self.w2v_model[word])))  # vs += sif * word_vector
        if sentence_length != 0:
            vs = np.divide(vs, sentence_length)  # weighted average
        else:
            vs = None
        return vs

    def train_pca(self):
        sentence_set = []
        # 切词
        stop_words = self.w2v_utils.get_stop_words()
        train_df = pd.read_csv(self.w2v_utils.train_data, encoding='utf-8')
        self.logger.info('cut sentence...')
        train_df['sentence'] = train_df.sentence.apply(self.w2v_utils.processing_sentence, args=(stop_words,))
        # 删掉缺失的行
        train_df.dropna(inplace=True)
        sentence_list = train_df.sentence.to_list()
        for sentence in tqdm(sentence_list):
            vs = self.calculate_weight(sentence)
            if vs is not None:
                sentence_set.append(vs)  # add to our existing re-calculated set of sentences
            else:
                continue

        # calculate PCA of this sentence set
        pca = PCA(n_components=self.w2v_utils.dim)
        self.logger.info('train pca vector...')
        pca.fit(np.array(sentence_set))
        u = pca.components_[0]  # the PCA vector
        u = np.multiply(u, np.transpose(u))  # u x uT
        self.logger.info('save pca vector...')
        np.save(self.pca_vec_path, u)

    def load_pca_vector(self):
        if not os.path.isfile(self.pca_vec_path):
            self.logger.info('pca vector not exist...')
            raise Exception('pca vector not exist...')
        else:
            self.u = np.load(self.pca_vec_path)

    def get_sif_vector(self, sentence):
        vs = self.calculate_weight(sentence)
        sub = np.multiply(self.u, vs)
        result_vec = np.subtract(vs, sub)
        return result_vec

    def get_average_vector(self, sentence):
        vs = np.zeros(self.w2v_utils.dim)  # add all word2vec values into one vector for the sentence
        sentence_length = len(sentence)
        for word in sentence:
            if word in self.w2v_model.wv.vocab:
                vs = np.add(vs, self.w2v_model[word])
            vs = np.divide(vs, sentence_length)  # weighted average
        return vs

    def similar_words(self, word):
        rtn_list = []
        rtn = self.w2v_model.similar_by_word(word, topn=5)
        for item in rtn:
            rtn_list.append({item[0]: item[1]})
        return rtn_list
