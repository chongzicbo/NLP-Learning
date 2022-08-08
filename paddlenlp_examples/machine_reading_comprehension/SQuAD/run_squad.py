import os
import random
import time
import json
import math
from functools import partial
import numpy as np
import paddle

from paddle.io import DataLoader
from args import parse_args
import paddlenlp as ppnlp
from paddlenlp.data import default_data_collator, DataCollatorWithPadding
from paddlenlp.transformers import BertForQuestionAnswering, BertTokenizer, \
    ErnieForQuestionAnswering, ErnieTokenizer, FunnelForQuestionAnswering, FunnelTokenizer

from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction

from datasets import load_dataset
