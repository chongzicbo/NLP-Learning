import os

# 修改环境变量要在导入datasets或者transformers模块之前
os.environ["XDG_CACHE_HOME"] = "/data/sshadmin/bocheng/.cache"
from peft import LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/mt0-large")

from peft import get_peft_model

model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
)
