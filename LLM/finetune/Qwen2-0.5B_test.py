from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to the large language model"

messages = [
    {
        "role": "system",
        "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    },
    {"role": "user", "content": prompt},
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
for x in model_inputs:
    print(x, model_inputs[x].shape)
generated_ids = model.generate(**model_inputs, max_new_tokens=512)
# print(generated_ids, generated_ids.shape)
# print(tokenizer.decode(generated_ids[0], skip_special_tokens=False))
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
]
# print(generated_ids)
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
