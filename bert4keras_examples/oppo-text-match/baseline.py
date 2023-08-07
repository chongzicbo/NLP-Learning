#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: baseline.py
@time: 2022/7/3 17:58
"""
# 句子对分类任务，脱敏数据
# 比赛链接：https://tianchi.aliyun.com/competition/entrance/531851
import os

os.environ["TF_KERAS"] = "1"  # 必须使用tf.keras
import tensorflow as tf

tf.config.run_functions_eagerly(True)  # 启动eager模式方便debug
# tf.compat.v1.enable_eager_execution
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from bert4keras.backend import keras, K
from bert4keras.tokenizers import load_vocab
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import truncate_sequences
from tqdm import tqdm

min_count = 5
maxlen = 32
batch_size = 10
bert_dir = (
    "/mnt/e/working/huada_bgi/data/pretrained_model/bert/chinese_L-12_H-768_A-12/"
)
config_path = os.path.join(bert_dir, "bert_config.json")
checkpoint_path = os.path.join(bert_dir, "bert_model.ckpt")
dict_path = os.path.join(bert_dir, "vocab.txt")

data_dir = "/mnt/e/opensource_data/文本匹配/小布助手语义匹配/tcdata/oppo_breeno_round1_data"


def load_data(filename):
    """加载数据
    单条格式：(文本1 ids, 文本2 ids, 标签id)
    """
    D = []
    with open(filename) as f:
        for l in f:
            l = l.strip().split("\t")  # sentence1,sentence2
            if len(l) == 3:
                a, b, c = l[0], l[1], int(l[2])  # 第三个字段，0表示不匹配
            else:
                a, b, c = l[0], l[1], -5  # 未标注数据，标签为-5
            a = [int(i) for i in a.split(" ")]  # 对sentence1进行split
            b = [int(i) for i in b.split(" ")]
            truncate_sequences(maxlen, -1, a, b)  # 对sentence进行截断和padding
            D.append((a, b, c))
    return D[:40]  # 取4条样本，方便debug


# 加载数据集
data = load_data(os.path.join(data_dir, "gaiic_track3_round1_train_20210228.tsv"))
train_data = [d for i, d in enumerate(data) if i % 10 != 0]
valid_data = [d for i, d in enumerate(data) if i % 10 == 0]
test_data = load_data(os.path.join(data_dir, "gaiic_track3_round1_testA_20210228.tsv"))

# 统计训练数据和测试数据的词频，以便跟加密数据进行对齐
tokens = {}
for d in data + test_data:
    for i in d[0] + d[1]:
        tokens[i] = tokens.get(i, 0) + 1

tokens = {i: j for i, j in tokens.items() if j >= min_count}  # 去掉频数小于5的单词
tokens = sorted(tokens.items(), key=lambda s: -s[1])  # 根据单词频数进行倒序排序
# 根据单词频数得到每个单词的位置序号 {word1:7,word2:8,word3:9}
tokens = {
    t[0]: i + 7  # 前七个为预留字符0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 5: no, 6: yes
    for i, t in enumerate(tokens)
}  # 0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 5: no, 6: yes

# BERT百度百科词频
counts = json.load(open("counts.json"))
del counts["[CLS]"]
del counts["[SEP]"]
token_dict = load_vocab(dict_path)  # bert词典
freqs = [counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])]
keep_tokens = list(np.argsort(freqs)[::-1])

# 模拟未标注
for d in valid_data + test_data:
    train_data.append((d[0], d[1], -5))


def random_mask(text_ids):
    """随机mask"""
    input_ids, output_ids = [], []
    rands = np.random.random(len(text_ids))
    for r, i in zip(rands, text_ids):
        if r < 0.15 * 0.8:
            input_ids.append(4)  # 4表示[MASK],15%的概率进行mask，其中0.8的概率被MASK
            output_ids.append(i)
        elif r < 0.15 * 0.9:  # 0.10的概率不变
            input_ids.append(i)
            output_ids.append(i)
        elif r < 0.15:
            input_ids.append(np.random.choice(len(tokens)) + 7)  # 0.10的概率 随机选择一个id
            output_ids.append(i)
        else:  # 0.85的概率 不进行mask
            input_ids.append(i)
            output_ids.append(0)  # id为0 表示 [PAD]
    return input_ids, output_ids


def sample_convert(text1, text2, label, random=False):
    """转换为MLM格式"""
    text1_ids = [tokens.get(t, 1) for t in text1]  # 1表示UNK，
    text2_ids = [tokens.get(t, 1) for t in text2]
    if random:
        if np.random.random() < 0.5:
            text1_ids, text2_ids = text2_ids, text1_ids  # 如果随机数小于0.5，则交换sen1和sen2位置
        text1_ids, out1_ids = random_mask(text1_ids)  # 对输入进行随机mask
        text2_ids, out2_ids = random_mask(text2_ids)
    else:
        out1_ids = [0] * len(text1_ids)
        out2_ids = [0] * len(text2_ids)
    token_ids = [2] + text1_ids + [3] + text2_ids + [3]  # 2表示[CLS],3表示[SEP],拼接两个输入文本
    segment_ids = [0] * len(token_ids)
    output_ids = (
        [label + 5] + out1_ids + [0] + out2_ids + [0]
    )  # label+5：如果label为1，+5=6，6表示yes,5表示no.对于测试数据，其label为-5，+5后=0，0表示[PAD].通过[CLS]向量预测yes或者no
    return token_ids, segment_ids, output_ids


class data_generator(DataGenerator):
    """数据生成器"""

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids, output_ids = sample_convert(
                text1, text2, label, random
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(output_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [batch_token_ids, batch_segment_ids], batch_output_ids
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_mlm=True,
    keep_tokens=[0, 100, 101, 102, 103, 100, 100]
    + keep_tokens[: len(tokens)],  # keep_tokens为要保留的词id列表
)


def masked_crossentropy(y_true, y_pred):
    """mask掉非预测部分"""
    y_true = K.reshape(y_true, K.shape(y_true)[:2])
    y_mask = K.cast(K.greater(K.cast(y_true, K.floatx()), 0.5), K.floatx())
    loss = K.sparse_categorical_crossentropy(y_true, y_pred)
    loss = K.sum(loss * y_mask) / K.sum(y_mask)
    return loss[None, None]


model.compile(loss=masked_crossentropy, optimizer=Adam(1e-5))
model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    """线下评测函数"""
    Y_true, Y_pred = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true)
        y_pred = y_pred[:, 0, 5:7]  # 0,表示取[CLS]位置对应的概率， 位置5、6，yes或者no的预测值
        y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
        y_true = y_true[:, 0] - 5
        Y_pred.extend(y_pred)
        Y_true.extend(y_true)
    return roc_auc_score(Y_true, Y_pred)  # 计算auc分数


class Evaluator(keras.callbacks.Callback):
    """评估与保存"""

    def __init__(self):
        self.best_val_score = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_score = evaluate(valid_generator)
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            model.save_weights("best_model.weights")
        print(
            "val_score: %.5f, best_val_score: %.5f\n" % (val_score, self.best_val_score)
        )


def predict_to_file(out_file):
    """预测结果到文件"""
    F = open(out_file, "w")
    for x_true, _ in tqdm(test_generator):
        y_pred = model.predict(x_true)[:, 0, 5:7]
        y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
        for p in y_pred:
            F.write("%f\n" % p)
    F.close()


if __name__ == "__main__":
    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=100,
        callbacks=[evaluator],
    )

else:
    model.load_weights("best_model.weights")
