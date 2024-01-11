# -*-coding:utf-8 -*-

"""
# File       : pipeline_test.py
# Time       ：2023/3/7 17:09
# Author     ：chengbo
# version    ：python 3.8
# Description：
"""
from transformers import pipeline

# classifier = pipeline("sentiment-analysis")
# print(classifier("We are very happy to show you the 🤗 Transformers library."))
# results = classifier(["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."])
# for result in results:
#     print(f"label: {result['label']}, with score: {round(result['score'], 4)}")

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print(
    classifier(
        "Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers."
    )
)

encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)

pt_batch = tokenizer(
    [
        "We are very happy to show you the 🤗 Transformers library.",
        "We hope you don't hate it.",
    ],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)

print(pt_batch)

pt_outputs = model(**pt_batch)
from torch import nn

print(pt_outputs)
pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
print(pt_predictions)

# pt_save_directory = "./pt_save_pretrained"
# tokenizer.save_pretrained(pt_save_directory)
# model.save_pretrained(pt_save_directory)

from transformers import AutoConfig

my_config = AutoConfig.from_pretrained("distilbert-base-uncased", n_heads=12)

from transformers import AutoModel

my_model = AutoModel.from_config(my_config)
