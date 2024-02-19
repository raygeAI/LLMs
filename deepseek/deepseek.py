# -*- coding: utf-8 -*-


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "../model/deepseek-moe-16b-chat"
# model_name = "deepseek-ai/deepseek-moe-16b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

messages = [
    {"role": "user", "content": "Who are you?"}
]
input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

# 如果必须使用 sdp_kernel 上下文管理器，请使用 memory efficient 或 math 内核
with torch.backends.cuda.sdp_kernel(
    enable_flash=False,
    enable_math=True,
    enable_mem_efficient=True
):
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)
    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
    print(result)

