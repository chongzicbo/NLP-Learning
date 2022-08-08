from paddlenlp.datasets import load_dataset

# raw_dataset = load_dataset("msra_ner")
# train_ds = raw_dataset["train.json"]
# for x in train_ds:
#     print(x)
#     break
# print(train_ds.label_list)
# print(train_ds.features["ner_tags"])
# for x in train_ds:
#     print(x.features["ner_tags"])
#     break
train_ds, eval_ds = load_dataset("msra_ner", split=("train.json", "test"))
label_list = train_ds.features["ner_tags"].feature.names
