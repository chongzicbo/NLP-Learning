#! -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: config.py
@time: 2022/10/31 9:54
"""
# [train_classifier, interactive_predict, test, save_model, train_word2vec, train_sif_sentence_vec]
mode = 'train_classifier'

word2vec_config = {
    'stop_words': 'data/w2v_data/stop_words.txt',  # 停用词(可为空)
    'train_data': 'data/w2v_data/dataset.csv',  # 词向量训练用的数据
    'model_dir': 'model/word2vec_model',  # 词向量模型的保存文件夹
    'model_name': 'word2vec_model.pkl',  # 词向量模型名
    'word2vec_dim': 300,  # 词向量维度
    'min_count': 3,  # 最低保留词频大小
    # 选择skip-gram和cbow
    'sg': 'cbow'
}

CUDA_VISIBLE_DEVICES = -1
# int, -1:CPU, [0,]:GPU
# coincides with tf.CUDA_VISIBLE_DEVICES

classifier_config = {
    # 模型选择
    # 传统模型：TextCNN/TextRNN/TextRCNN/Transformer
    # 预训练模型：Bert/DistilBert/AlBert/RoBerta/Electra/XLNet
    'classifier': 'TextCNN',
    # 若选择Bert系列微调做分类，请在pretrained指定预训练模型的版本
    'pretrained': 'bert-base-chinese',
    # 训练数据集
    'train_file': 'data/train_dataset.csv',
    # 验证数据集
    'val_file': 'data/val_dataset.csv',
    # 测试数据集
    'test_file': 'data/test_dataset.csv',
    # 引入外部的词嵌入,可选word2vec、Bert
    # word2vec:使用word2vec词向量做特征增强
    # 不填写则随机初始化的Embedding
    'embedding_method': '',
    # token的粒度,token选择字粒度的时候，词嵌入(embedding_method)无效
    # 词粒度:'word'
    # 字粒度:'char'
    'token_level': 'word',
    # 去停用词，路径需要在上面的word2vec_config中配置，仅限非预训练微调使用
    'stop_words': True,
    # 是否去掉特殊字符
    'remove_special': True,
    # 不外接词嵌入的时候需要自定义的向量维度
    'embedding_dim': 300,
    # 存放词表的地方
    'token_file': 'data/word-token2id',
    # 类别列表
    'classes': ['家居', '时尚', '教育', '财经', '时政', '娱乐', '科技', '体育', '游戏', '房产'],
    # 模型保存的文件夹
    'checkpoints_dir': 'model/textcnn',
    # 模型保存的名字
    'checkpoint_name': 'textcnn',
    # 使用Textcnn模型时候设定卷集核的个数
    'num_filters': 64,
    # 学习率
    # 微调预训练模型时建议更小，设置5e-5
    'learning_rate': 0.0005,
    # 优化器选择
    # 可选：Adagrad/Adadelta/RMSprop/SGD/Adam/AdamW
    'optimizer': 'Adam',
    # 训练epoch
    'epoch': 100,
    # 最多保存max_to_keep个模型
    'max_to_keep': 1,
    # 每print_per_batch打印
    'print_per_batch': 100,
    # 是否提前结束
    'is_early_stop': True,
    # 是否引入attention
    # 注意:textrcnn不支持
    'use_attention': False,
    # attention大小
    'attention_size': 300,
    'patient': 8,
    'batch_size': 256,
    'max_sequence_length': 300,
    # 遗忘率
    'dropout_rate': 0.5,
    # 隐藏层维度
    # 使用textrcnn、textrnn和transformer中需要设定
    # 使用transformer建议设定为2048
    'hidden_dim': 256,
    # 编码器个数(使用transformer需要设定)
    'encoder_num': 1,
    # 多头注意力的个数(使用transformer需要设定)
    'head_num': 12,
    # 若为二分类则使用binary
    # 多分类使用micro或macro
    'metrics_average': 'micro',
    # 类别样本比例失衡的时候可以考虑使用
    'use_focal_loss': True,
    # focal loss的各个标签权重
    'weight': None,
    # 使用标签平滑
    # 主要用在预训练模型微调，直接训练小模型使用标签平滑会带来负面效果，慎用
    'use_label_smoothing': False,
    'smooth_factor': 0.1,
    # 是否使用GAN进行对抗训练
    'use_gan': True,
    # 目前支持FGM和PGD两种方法
    # fgm:Fast Gradient Method
    # pgd:Projected Gradient Descent
    'gan_method': 'pgd',
    # 对抗次数
    'attack_round': 3,
    # 使用对比学习，不推荐和对抗方法一起使用，效率慢收益不大
    'use_r_drop': False
}
