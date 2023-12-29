import os

os.environ["XDG_CACHE_HOME"] = "/data/sshadmin/bocheng/.cache"
import torch
import transformers
import accelerate
import peft

print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")
model_checkpoint = "google/vit-base-patch16-224-in21k"

1.0  # 准备数据
from datasets import load_dataset

dataset = load_dataset(
    "food101",
    cache_dir="/data/sshadmin/bocheng/.cache/huggingface/datasets",
    split="train[:5000]",
)
labels = dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label
print(id2label[2])

# 2.加载图片处理器
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transformers = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transformers = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)


def preprocess_train(example_batch):
    example_batch["pixel_values"] = [
        train_transformers(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


def preprocess_val(example_batch):
    example_batch["pixel_values"] = [
        val_transformers(image.convert("RGB")) for image in example_batch["image"]
    ]
    return example_batch


splits = dataset.train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]

train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)


# 3.加载模型
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params:{trainable_params} || all params:{all_param} || trainable%:{100*trainable_params/all_param:.2f}"
    )


from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint, label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True
)

print_trainable_parameters(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)

lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

# 4.定义训练参数
model_name = model_checkpoint.split("/")[-1]
batch_size = 128
args = TrainingArguments(
    f"{model_name}-finetuned-lora-food101",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=5,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    label_names=["labels"],
)

# 5.模型评估方法
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

"""
The compute_metrics function takes a named tuple as input: predictions, which are the logits of the model as Numpy arrays, and label_ids, which are the ground-truth labels as Numpy arrays.
"""


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


import torch


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# 6.模型训练评估
# trainer = Trainer(
#     lora_model,
#     args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     tokenizer=image_processor,
#     compute_metrics=compute_metrics,
#     data_collator=collate_fn,
# )

# train_results = trainer.train()

# print(trainer.evaluate(val_ds))

# 7.模型推理
from peft import PeftConfig, PeftModel

model_id = f"{model_name}-finetuned-lora-food101/checkpoint-20"
config = PeftConfig.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(
    config.base_model_name_or_path,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
image_processor = AutoImageProcessor.from_pretrained(model_id)
inference_model = PeftModel.from_pretrained(model, model_id)
from PIL import Image
import requests

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

encoding = image_processor(image.convert("RGB"), return_tensors="pt")
with torch.no_grad():
    outputs = inference_model(**encoding)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", inference_model.config.id2label[predicted_class_idx])
"Predicted class: beignets"
