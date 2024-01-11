# -*- coding:utf-8 -*-
"""
@author: chengbo
@software: PyCharm
@file: ann_util.py
@time: 2022/6/24 16:26
"""

import numpy as np
import hnswlib
from paddlenlp.utils.log import logger


def build_index(args, data_loader, model):
    index = hnswlib.Index(space="ip", dim=args.output_emb_size)

    # Initializing index
    # max_elements - the maximum number of elements (capacity). Will throw an exception if exceeded
    # during insertion of an element.
    # The capacity can be increased by saving/loading the index, see below.
    #
    # ef_construction - controls index search speed/build speed tradeoff
    #
    # M - is tightly connected with internal dimensionality of the data. Strongly affects memory consumption (~M)
    # Higher M leads to higher accuracy/run_time at fixed ef/efConstruction
    index.init_index(
        max_elements=args.hnsw_max_elements, ef_construction=args.hnsw_ef, M=args.hnsw_m
    )

    # Controlling the recall by setting ef:
    # higher ef leads to better accuracy, but slower search
    index.set_ef(args.hnsw_ef)

    # Set number of threads used during batch search/construction
    # By default using all available cores
    index.set_num_threads(16)

    logger.info("start build index..........")

    all_embeddings = []

    for text_embeddings in model.get_semantic_embedding(data_loader):
        all_embeddings.append(text_embeddings.numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    index.add_items(all_embeddings)

    logger.info("Total index number:{}".format(index.get_current_count()))

    return index
