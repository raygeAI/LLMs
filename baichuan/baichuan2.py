# -*- coding: utf-8 -*-
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

model_file = "../model/baichuan-inc/baichuan2-7b-chat"

tokenizer = AutoTokenizer.from_pretrained(model_file, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_file, device_map="auto",
                                             torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_file)
messages = []
messages.append({"role": "user", "content": "默写长恨歌"})
response = model.chat(tokenizer, messages)
print(response)
