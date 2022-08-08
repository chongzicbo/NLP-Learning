import os

import numpy as np
import paddle
from paddlenlp.datasets import load_dataset


def create_dataloader(dataset, mode="train.json", batch_size=1, batchify_fn=None, trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == "train.json" else False
    if mode == "train.json":
        sampler = paddle.io.DistributedBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    dataloader = paddle.io.DataLoader(dataset, batch_sampler=sampler, collate_fn=batchify_fn)
    return dataloader


def preprocess_prediction_data(data, tokenizer, pad_token_id=0, max_ngram_filter_size=3):
    examples = []
    for text in data:
        ids = tokenizer.encode(text)
        seq_len = len(ids)
        if seq_len < max_ngram_filter_size:
            ids.extend([pad_token_id] * (max_ngram_filter_size - seq_len))
        examples.append(ids)
    return examples


def convert_example(example, tokenizer):
    input_ids = tokenizer.encode(example["text"])
    input_ids = np.array(input_ids, dtype="int64")
    label = np.array(example["label"], dtype="int64")
    return input_ids, label


def read_custom_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        # skip head
        next(f)
        for line in f:
            data = line.strip().split("\t")
            label, text = data
            yield {"text": text, "label": label}


if __name__ == '__main__':
    from functools import partial
    import numpy as np
    import paddle
    from paddlenlp.datasets import load_dataset
    from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab

    data_dir = "E:\\opensource_data\\分类\\情感分析\\RobotChat"
    vocab_path = os.path.join(data_dir, "robot_chat_word_dict.txt")
    vocab = Vocab.load_vocabulary(vocab_path, unk_token='[UNK]', pad_token='[PAD]')
    # print(vocab.idx_to_token)
    dataset_names = ["train.json.tsv", "dev.tsv", "test.tsv"]
    train_ds = load_dataset(read_custom_data, filename=os.path.join(data_dir, dataset_names[0]), lazy=False)
    # for x in train_ds:
    #     print(x)
    # for x in data:
    #     print(x)
    #     break

    tokenizer = JiebaTokenizer(vocab)
    trans_fn = partial(convert_example, tokenizer=tokenizer)
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=vocab.token_to_idx.get('[PAD]', 0)),
        Stack(dtype='int64')  # label
    ): [data for data in fn(samples)]
    train_loader = create_dataloader(train_ds,
                                     batch_size=5,
                                     mode="train.json",
                                     batchify_fn=batchify_fn,
                                     trans_fn=trans_fn
                                     )
    for x in train_loader:
        print(x)
        break
    data = [
        [[1, 2, 3, 4], [1]],
        [[5, 6, 7], [0]],
        [[8, 9], [1]],
    ]
    # # batchify_fn = Tuple(Pad(pad_val=0), Stack())
    data = batchify_fn(data)
    print(data)
