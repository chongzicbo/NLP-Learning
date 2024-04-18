import os, sys

os.environ["XDG_CACHE_HOME"] = "/data/bocheng/data/.cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/bocheng/data/.cache/huggingface/hub/"
import torch
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    GenerationConfig,
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model

### config ###
model_id = "NousResearch/Llama-2-7b-hf"  # optional meta-llama/Llama-2–7b-chat-hf
max_length = 512
device_map = "auto"
batch_size = 128
micro_batch_size = 32
gradient_accumulation_steps = batch_size // micro_batch_size

# nf4" use a symmetric quantization scheme with 4 bits precision
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# load model from huggingface
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, use_cache=False, device_map=device_map
)

# load tokenizer from huggingface
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# 输出可训练参数量
def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(
        f"trainable model parameters: {trainable_model_params}. All model parameters: {all_model_params}"
    )
    return trainable_model_params


ori_p = print_number_of_trainable_model_parameters(model)

# 配置LoRa参数
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

peft_p = print_number_of_trainable_model_parameters(model)
print(
    f"# Trainable Parameter \n Before: {ori_p} \n After :{peft_p} \n Percentage: {round(peft_p/ori_p * 100,2)}"
)

## 看看原始LLaMA2的生成效果
prompt = "Write me a poem about Singapore"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=64)
print("\nAnswer: ", tokenizer.decode(generate_ids[0]))
res = tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(res)

# 微调
max_length = 256
dataset = datasets.load_dataset("databricks/databricks-dolly-15k", split="train")
### generate prompt based on template ###
prompt_template = {
    "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context.\
    Write a response that appropriately completes the request.\
    \n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
    "prompt_no_input": "Below is an instruction that describes a task.\
    Write a response that appropriately completes the request.\
    \n\n### Instruction:\n{instruction}\n\n### Response:\n",
    "response_split": "### Response:",
}


def generate_prompt(
    instruction, input=None, label=None, prompt_template=prompt_template
):
    if input:
        res = prompt_template["prompt_input"].format(
            instruction=instruction, input=input
        )
    else:
        res = prompt_template["prompt_no_input"].format(instruction=instruction)
    if label:
        res = f"{res}{label}"

    return res


def tokenize(tokenizer, prompt, max_length=max_length, add_eos_token=False):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(
        data_point["instruction"], data_point["context"], data_point["response"]
    )
    tokenized_full_prompt = tokenize(tokenizer, full_prompt)
    user_prompt = generate_prompt(data_point["instruction"], data_point["context"])
    tokenized_user_prompt = tokenize(tokenizer, user_prompt)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    mask_token = [-100] * user_prompt_len
    tokenized_full_prompt["labels"] = (
        mask_token + tokenized_full_prompt["labels"][user_prompt_len:]
    )
    return tokenized_full_prompt


dataset = dataset.train_test_split(test_size=1000, shuffle=True, seed=42)
cols = ["instruction", "context", "response", "category"]
train_data = (
    dataset["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)
)
val_data = (
    dataset["test"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)
)

# 模型训练
args = TrainingArguments(
    output_dir="./llama-7b-int4-dolly",
    num_train_epochs=20,
    max_steps=200,
    fp16=True,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="constant",
    per_device_train_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    group_by_length=False,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    disable_tqdm=False,
)


trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
    ),
)
model.config.use_cache = False
trainer.train()
model.save_pretrained("llama-7b-int4-dolly")

# model path and weight
model_id = "NousResearch/Llama-2-7b-hf"
peft_path = "./llama-7b-int4-dolly"

# loading model
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, use_cache=False, device_map="auto"
)

# loading peft weight
model = PeftModel.from_pretrained(
    model,
    peft_path,
    torch_dtype=torch.float16,
)
model.eval()

# generation config
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,  # beam search
)

# generating reply
with torch.no_grad():
    prompt = "Write me a poem about Singapore."
    inputs = tokenizer(prompt, return_tensors="pt")
    generation_output = model.generate(
        input_ids=inputs.input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
        max_new_tokens=64,
    )
    print("\nAnswer: ", tokenizer.decode(generation_output.sequences[0]))
