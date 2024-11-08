#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: cls.py
@time: 2022/7/2 21:59
"""
# bert做Seq2Seq任务，采用UNILM方案
# 通过R-Drop增强模型的泛化性能
# 介绍链接：https://kexue.fm/archives/6933
# 数据集：https://github.com/CLUEbenchmark/CLGE 中的CSL数据集

from __future__ import print_function
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from keras.losses import kullback_leibler_divergence as kld
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 基本参数
maxlen = 256
batch_size = 16
epochs = 50  # 训练更多的epoch还能进一步有提升

# bert配置
config_path = "/root/kg/bert/chinese_wwm_L-12_H-768_A-12/bert_config.json"
checkpoint_path = "/root/kg/bert/chinese_wwm_L-12_H-768_A-12/bert_model.ckpt"
dict_path = "/root/kg/bert/chinese_wwm_L-12_H-768_A-12/vocab.txt"


def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding="utf-8") as f:
        for l in f:
            title, content = l.strip().split("\t")
            D.append((title, content))
    return D


# 加载数据集
train_data = load_data("/root/csl/train.json.tsv")
valid_data = load_data("/root/csl/val.tsv")
test_data = load_data("/root/csl/test.tsv")

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器"""

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (title, content) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(content, title, maxlen=maxlen)
            for i in range(2):
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分"""

    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        alpha = 4
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss1 = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss1 = K.sum(loss1 * y_mask) / K.sum(y_mask)
        loss2 = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
        loss2 = K.sum(loss2 * y_mask[::2]) / K.sum(y_mask[::2])
        return loss1 + loss2 / 4 * alpha


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application="unilm",
    dropout_rate=0.3,
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器"""

    @AutoRegressiveDecoder.wraps(default_rtype="probas")
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(model).predict([token_ids, segment_ids])

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search(
            [token_ids, segment_ids], topk=topk
        )  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)


class Evaluator(keras.callbacks.Callback):
    """评估与保存"""

    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.0

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)  # 评测模型
        if metrics["bleu"] > self.best_bleu:
            self.best_bleu = metrics["bleu"]
            model.save_weights("./best_model.weights")  # 保存模型
        metrics["best_bleu"] = self.best_bleu
        print("valid_data:", metrics)

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        for title, content in tqdm(data):
            total += 1
            title = " ".join(title).lower()
            pred_title = " ".join(autotitle.generate(content, topk)).lower()
            if pred_title.strip():
                scores = self.rouge.get_scores(hyps=pred_title, refs=title)
                rouge_1 += scores[0]["rouge-1"]["f"]
                rouge_2 += scores[0]["rouge-2"]["f"]
                rouge_l += scores[0]["rouge-l"]["f"]
                bleu += sentence_bleu(
                    references=[title.split(" ")],
                    hypothesis=pred_title.split(" "),
                    smoothing_function=self.smooth,
                )
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            "rouge-1": rouge_1,
            "rouge-2": rouge_2,
            "rouge-l": rouge_l,
            "bleu": bleu,
        }


if __name__ == "__main__":
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator],
    )

else:
    model.load_weights("./best_model.weights")
