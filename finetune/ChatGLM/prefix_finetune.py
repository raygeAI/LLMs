# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import DataLoader
from typing import List, Dict
from tqdm import tqdm
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import get_cosine_schedule_with_warmup
from peft import PrefixTuningConfig, get_peft_model, TaskType
from peft import PeftModel, PeftConfig
from dataset.dataset import ChatGLM3PrefixDataSet, DataCollate, ChatGLM3PromptDataSet
from utils import set_random_seed, print_trainable_parameters


# PrefixTuneConfig 微调相关配置
@dataclass
class PrefixTuneConfig:
    # 文本最大长度
    max_length: int = 256
    # 最大输入长度+
    max_src_length: int = 128
    batch_size: int = 15
    num_epochs: int = 100
    lr: float = 2e-2
    pre_seq_len: int = 32
    prefix_projection: bool = False
    # train_file: str = "./data/AdvertiseGen/dev.json"
    train_file: str = "data/hotel_train.jsonl"
    device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    peft_prefix_file: str = "./model/finetune/ChatGLM3/prefix_tuning"
    base_model_file: str = "../../model/ChatGLM/chatGLM3"


# ChatGLM3PrefixTuning  基于 Prefix 的 ChatGLM3 微调过程
class ChatGLM3PrefixTuning:
    def __init__(self):
        # 设置随机数种子
        set_random_seed(29)
        self.tokenizer = AutoTokenizer.from_pretrained(PrefixTuneConfig.base_model_file, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(PrefixTuneConfig.base_model_file, trust_remote_code=True)
        self.config.pre_seq_len = PrefixTuneConfig.pre_seq_len
        self.config.prefix_projection = PrefixTuneConfig.prefix_projection
        self.base_model = AutoModel.from_pretrained(PrefixTuneConfig.base_model_file, config=self.config,
                                                    trust_remote_code=True)

    # __load_dataset 加载 lora 微调并训练数据
    def __load_dataset(self) -> DataLoader:
        dataset = ChatGLM3PromptDataSet(PrefixTuneConfig.train_file, self.tokenizer, PrefixTuneConfig.max_length, PrefixTuneConfig.max_src_length, True)
        data_collate = DataCollate(self.tokenizer)
        dataloader = DataLoader(dataset, collate_fn=data_collate, batch_size=PrefixTuneConfig.batch_size)
        return dataloader

    # __get_prefix_model 基于基础模型，依据 PrefixTuningConfig  配置，构建 prefixTuning 模型
    def __get_prefix_model(self) -> PeftModel:
        model = self.base_model
        model.train()
        print(model)
        model.to(PrefixTuneConfig.device)
        print_trainable_parameters(model)
        return model

    # __save_prefix_tuning_mode 保存模型
    def __save_prefix_tuning_model(self, peft_prefix_file: str):
        os.makedirs(PrefixTuneConfig.peft_prefix_file, exist_ok=True)
        prefix_encoder_params = {
            k: v.to("cpu") for k, v in self.base_model.named_parameters() if v.requires_grad
        }
        torch.save(prefix_encoder_params, os.path.join(PrefixTuneConfig.peft_prefix_file, "pytorch_model.bin"))
        self.tokenizer.save_pretrained(peft_prefix_file)

    # prefix_finetune 执行 prefix_tuning 微调
    def prefix_finetune(self):
        dataloader = self.__load_dataset()
        model = self.__get_prefix_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=PrefixTuneConfig.lr, eps=1e-8)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=15,
            num_training_steps=(len(dataloader) * PrefixTuneConfig.num_epochs),
        )

        for epoch in range(PrefixTuneConfig.num_epochs):
            model.train()
            train_loss = 0
            for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), unit="batch"):
                optimizer.zero_grad()
                outputs = model(**batch, use_cache=True)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                train_loss += loss.item()
                # 每10次打印一次
                if step % 10 == 0:
                    print("\n......loss......", loss.item())
            print("\n Epoch: {} Train Loss: {}".format(epoch, train_loss))
        # 保存 prefix tuning 微调之后的模型以及 tokenize 文件
        self.__save_prefix_tuning_model(PrefixTuneConfig.peft_prefix_file)

    # merge_prefix_tuned_model 加载经过 prefix 微调和原始 base mode 模型进行合并
    def merge_prefix_tuned_model(self):
        prefix_state_dict = torch.load(os.path.join(PrefixTuneConfig.peft_prefix_file, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        self.base_model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        self.base_model.half().to(PrefixTuneConfig.device)
        # 切换到推理模式
        self.base_model.eval()
        return self.base_model

    # chat 聊天会话
    def chat(self, query: str, history: List[Dict] = None):
        model = self.merge_prefix_tuned_model()
        responses, past = model.chat(self.tokenizer, query, history)
        return responses, past


if __name__ == "__main__":
    prefixTune = ChatGLM3PrefixTuning()
    prefixTune.prefix_finetune()
    query = "介绍一下你自己？"
    response, history = prefixTune.chat(query)
    print(response, "\n", history)
