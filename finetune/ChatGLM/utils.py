# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
from transformers import set_seed


# set_random_seed 设置随机数种子
def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# save_lora_model 保存微调之后 lora 部分文件
def save_lora_model(model, tokenizer, lora_file: str, state_dict=None):
    if state_dict is None:
        model.save_pretrained(lora_file, torch_dtype=torch.float16)
    else:
        model.save_pretrained(lora_file, state_dict=state_dict, torch_dtype=torch.float16)
    tokenizer.save_pretrained(lora_file)


# print_trainable_parameters 输出可训练参数
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print("trainable params: {} || all params: {} || trainable%: {}".format(trainable_params, all_param, 100 * trainable_params / all_param))
