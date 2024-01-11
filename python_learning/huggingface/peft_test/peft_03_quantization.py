import os

os.environ["XDG_CACHE_HOME"] = "/data/sshadmin/bocheng/.cache"
import torch
from transformers import BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", quantization_config=config
)

from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

from peft import LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
from peft import get_peft_model

model = get_peft_model(model, config)

from peft import AutoModelForCausalLM, LoftQConfig, LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
loftq_config = LoftQConfig(loftq_bits=4)

lora_config = LoraConfig(
    init_lora_weights="loftq",
    loftq_config=loftq_config,
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
