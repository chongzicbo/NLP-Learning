#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: deepfm_test.py
@time: 2022/8/15 19:51
"""
import sys
# import pytest
from util import get_test_data, SAMPLE_SIZE, check_model, get_device

sys.path.append("..")
from models.deepfm import DeepFM


# @pytest.mark.parametrize(
#     'use_fm,hidden_size,sparse_feature_num,dense_feature_num',
#     [(True, (32,), 3, 3),
#      (False, (32,), 3, 3),
#      (False, (32,), 2, 2),
#      (False, (32,), 1, 1),
#      (True, (), 1, 1),
#      (False, (), 2, 2),
#      (True, (32,), 0, 3),
#      (True, (32,), 3, 0),
#      (False, (32,), 0, 3),
#      (False, (32,), 3, 0),
#      ]
# )
def test_DeepFM(use_fm, hidden_size, sparse_feature_num, dense_feature_num):
    model_name = "DeepFM"
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num=sparse_feature_num, dense_feature_num=dense_feature_num)

    model = DeepFM(feature_columns, feature_columns, use_fm=use_fm,
                   dnn_hidden_units=hidden_size, dnn_dropout=0.5, device=get_device())
    check_model(model, model_name, x, y)

    # no linear part
    model = DeepFM([], feature_columns, use_fm=use_fm,
                   dnn_hidden_units=hidden_size, dnn_dropout=0.5, device=get_device())
    check_model(model, model_name + '_no_linear', x, y)


if __name__ == "__main__":
    sample_size = SAMPLE_SIZE
    x, y, feature_columns = get_test_data(
        sample_size, sparse_feature_num=1, dense_feature_num=1)
    model = DeepFM(feature_columns, feature_columns,
                   dnn_dropout=0.5, device=get_device())
    model.compile('adam', 'binary_crossentropy',
                  metrics=['binary_crossentropy', 'acc'])
    model.fit(x, y, batch_size=100, epochs=1, validation_split=0.5)
    # print(x)
    # # y_pred = model(x)
