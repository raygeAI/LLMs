import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# model_dir = "E:\model\Llama-2-7b-chat-hf"
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='auto',
#                                              torch_dtype=torch.float16)
# model = model.eval()
# tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
# input_ids = tokenizer(['<s> 鸡和兔在一个笼子里，共有35个头，94只脚，那么鸡有多少只，兔有多少只？ <s>Assistant: '], return_tensors="pt",
#                       add_special_tokens=False).input_ids.to('cuda')
# generate_input = {
#     "input_ids": input_ids,
#     "max_new_tokens": 512,
#     "do_sample": True,
#     "top_k": 50,
#     "top_p": 0.95,
#     "temperature": 0.3,
#     "repetition_penalty": 1.3,
#     "eos_token_id": tokenizer.eos_token_id,
#     "bos_token_id": tokenizer.bos_token_id,
#     "pad_token_id": tokenizer.pad_token_id
# }
# generate_ids = model.generate(**generate_input)
# text = tokenizer.decode(generate_ids[0])
# print(text)

from llama2_for_langchain import Llama2

# 这里以调用4bit量化压缩的Llama2-Chinese参数FlagAlpha/Llama2-Chinese-13b-Chat-4bit为例
llm = Llama2(model_name_or_path='FlagAlpha/Llama2-Chinese-13b-Chat-4bit', bit4=True)

while True:
    human_input = input("Human: ")
    response = llm(human_input)
    print(f"Llama2: {response}")