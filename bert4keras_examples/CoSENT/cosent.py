#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: cosent.py
@time: 2022/7/3 15:42
"""
import json
import numpy as np
import scipy.stats
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import open
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.optimizers import Adam
from tqdm import tqdm
import sys

task_name = sys.argv[1]
assert task_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B']


def load_data(filename):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                D.append((l[0], l[1], int(l[2])))
    return D


# 基本参数
maxlen = 64 if task_name != 'PAWSX' else 128
batch_size = 32
epochs = 5

# 模型路径
config_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 加载数据集
data_path = '/root/senteval_cn/'
datasets = [
    load_data('%s%s/%s.%s.data' % (data_path, task_name, task_name, f))
    for f in ['train.json', 'valid', 'test']
]
train_data, valid_data, test_data = datasets


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            for text in [text1, text2]:
                token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def cosent_loss(y_true, y_pred):
    """排序交叉熵
    y_true：标签/打分，y_pred：句向量
    """
    y_true = y_true[::2, 0]
    y_true = K.cast(y_true[:, None] < y_true[None, :], K.floatx())
    y_pred = K.l2_normalize(y_pred, axis=1)
    y_pred = K.sum(y_pred[::2] * y_pred[1::2], axis=1) * 20
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = K.reshape(y_pred - (1 - y_true) * 1e12, [-1])
    y_pred = K.concatenate([[0], y_pred], axis=0)
    return K.logsumexp(y_pred)


# 构建模型
base = build_transformer_model(config_path, checkpoint_path)
output = keras.layers.Lambda(lambda x: x[:, 0])(base.output)
# output = keras.layers.GlobalAveragePooling1D()(base.output)
encoder = keras.models.Model(base.inputs, output)

model = encoder
model.compile(loss=cosent_loss, optimizer=Adam(2e-5))


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def l2_normalize(vecs):
    """l2标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


class Evaluator(keras.callbacks.Callback):
    """保存验证集分数最好的模型
    """
    def __init__(self):
        self.best_val_score = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_score = self.evaluate(valid_generator)
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            model.save_weights('%s.cosent.weights' % task_name)
        print(
            u'val_score: %.5f, best_val_score: %.5f\n' %
            (val_score, self.best_val_score)
        )

    def evaluate(self, data):
        Y_true, Y_pred = [], []
        for x_true, y_true in data:
            Y_true.extend(y_true[::2, 0])
            x_vecs = encoder.predict(x_true)
            x_vecs = l2_normalize(x_vecs)
            y_pred = (x_vecs[::2] * x_vecs[1::2]).sum(1)
            Y_pred.extend(y_pred)
        return compute_corrcoef(Y_true, Y_pred)


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
    model.load_weights('%s.cosent.weights' % task_name)
    test_score = evaluator.evaluate(test_generator)
    print(u'test_score: %.5f' % test_score)

else:

    model.load_weights('%s.cosent.weights' % task_name)