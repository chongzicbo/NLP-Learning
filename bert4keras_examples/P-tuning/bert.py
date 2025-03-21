#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: bert.py
@time: 2022/8/18 22:06
"""
# 情感分析例子，利用MLM+P-tuning
import os

os.environ.setdefault("TF_KERAS", "1")
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss, Embedding
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model, BERT
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense
import tensorflow as tf

# keras默认是执行图模式，debug困难；使用eager模式方便debug
tf.config.run_functions_eagerly(True)

maxlen = 128
batch_size = 2
bert_dir = (
    "/mnt/e/working/huada_bgi/data/pretrained_model/bert/chinese_L-12_H-768_A-12/"
)
config_path = os.path.join(bert_dir, "bert_config.json")
checkpoint_path = os.path.join(bert_dir, "bert_model.ckpt")
dict_path = os.path.join(bert_dir, "vocab.txt")


def load_data(filename):
    D = []
    with open(filename, encoding="utf-8") as f:
        for l in f:
            text, label = l.strip().split("\t")
            D.append((text, int(label)))
    return D[:400]  # 取4条用于debug


# 加载数据集
data_dir = "/mnt/e/opensource_data/分类/情感分析/sentiment/"
train_data = load_data(os.path.join(data_dir, "sentiment.train.data"))
valid_data = load_data(os.path.join(data_dir, "sentiment.valid.data"))
test_data = load_data(os.path.join(data_dir, "sentiment.test.data"))

# 模拟标注和非标注数据
train_frac = 0.01  # 标注数据的比例
num_labeled = int(len(train_data) * train_frac)  # 取1%的数据作为标注数据
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
train_data = train_data[:num_labeled]
# train_data = train_data + unlabeled_data

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
mask_idx = 5
desc = ["[unused%s]" % i for i in range(1, 9)]
desc.insert(mask_idx - 1, "[MASK]")
desc_ids = [tokenizer.token_to_id(t) for t in desc]
pos_id = tokenizer.token_to_id("很")
neg_id = tokenizer.token_to_id("不")


def random_masking(token_ids):
    """对输入进行随机mask"""
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:  # 15%里的80%被mask
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:  # 15%里的10%保持不变
            source.append(t)
            target.append(t)
        elif r < 0.15:  # 剩下的10%进行随机替换
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:  # 剩下的85%不进行mask,修改target id为0
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    """数据生成器"""

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            # 增加模板前缀
            if label != 2:
                token_ids = token_ids[:1] + desc_ids + token_ids[1:]
                segment_ids = [0] * len(desc_ids) + segment_ids
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            # 把模板中的mask_idx位置id修改为[MASK]对应的id，对应的target_id修改为给定模板提示的Id
            if label == 0:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = neg_id
            elif label == 1:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = pos_id
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [batch_token_ids, batch_segment_ids, batch_output_ids], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分"""

    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name="accuracy")
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


class PtuningEmbedding(Embedding):
    """新定义Embedding层，只优化部分Token"""

    def call(self, inputs, mode="embedding"):
        embeddings = self.embeddings
        embeddings_sg = K.stop_gradient(embeddings)
        mask = np.zeros((K.int_shape(embeddings)[0], 1))
        mask[1:9] += 1  # 只优化id为1～8的token
        self.embeddings = embeddings * mask + embeddings_sg * (1 - mask)
        outputs = super(PtuningEmbedding, self).call(inputs, mode)
        self.embeddings = embeddings
        return outputs


class PtuningBERT(BERT):
    """替换原来的Embedding"""

    def apply(self, inputs=None, layer=None, arguments=None, **kwargs):
        if layer is Embedding:
            layer = PtuningEmbedding
        return super(PtuningBERT, self).apply(inputs, layer, arguments, **kwargs)


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model=PtuningBERT,
    with_mlm=True,
)

for layer in model.layers:
    if layer.name != "Embedding-Token":
        layer.trainable = False

# 训练用模型
y_in = keras.layers.Input(shape=(None,))
output = keras.layers.Lambda(lambda x: x[:, :10])(model.output)
outputs = CrossEntropy(1)(
    [y_in, model.output]
)  # CrossEntropy(1)  1:是经过损失层后，输出的还是model.output

train_model = keras.models.Model(model.inputs + [y_in], outputs)
train_model.compile(optimizer=Adam(6e-4))
train_model.summary()

# 预测模型
model = keras.models.Model(model.inputs, output)

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights("best_model_bert.weights")
        test_acc = evaluate(test_generator)
        print(
            "val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n"
            % (val_acc, self.best_val_acc, test_acc)
        )


def evaluate(data):
    total, right = 0.0, 0.0
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2]
        y_pred = model.predict(x_true)
        y_pred = y_pred[:, mask_idx, [neg_id, pos_id]].argmax(
            axis=1
        )  # 把mask_idx位置对应的行取出来，再把行对应的neg_id,pos_id位置预测值取出来，得到对应的预测结果
        y_true = (y_true[:, mask_idx] == pos_id).astype(int)
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


if __name__ == "__main__":
    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator) * 50,
        epochs=1000,
        callbacks=[evaluator],
    )

else:
    model.load_weights("best_model_bert.weights")
