# -*- coding: utf-8 -*-
from typing import List, Dict
from tqdm import tqdm
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel, PeftConfig
from dataset.dataset import ChatGLM3PromptDataSet, DataCollate
from utils import set_random_seed, save_lora_model


# LoRATuneConfig lora 微调相关配置
@dataclass
class LoRATuneConfig:
    peft_lora_file: str = "./model/finetune/ChatGLM3/lora.pth"
    base_model_file: str = "../../model/ChatGLM/chatGLM3"
    train_file: str = "./data/hotel_train.json"
    # 文本最大长度
    max_length: int = 2048
    # 最大输入长度+
    max_src_length: int = 1024
    batch_size: int = 3
    num_epochs: int = 20
    lr: float = 5e-4
    device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ChatGLM3LoRaTuning  基于 Lora 的 ChatGLM3 微调过程
class ChatGLM3LoRATuning:
    def __init__(self: str):
        set_random_seed(29)
        self.base_model = AutoModel.from_pretrained(LoRATuneConfig.base_model_file, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(LoRATuneConfig.base_model_file, trust_remote_code=True)

    # load_dataset 加载 lora 微调并训练数据
    def _load_dataset(self) -> DataLoader:
        dataset = ChatGLM3PromptDataSet(LoRATuneConfig.train_file, self.tokenizer, LoRATuneConfig.max_length,
                                        LoRATuneConfig.max_src_length, True)
        data_collate = DataCollate(self.tokenizer)
        dataloader = DataLoader(dataset, collate_fn=data_collate, batch_size=LoRATuneConfig.batch_size)
        return dataloader

    # get_lora_model 基于基础模型，依据lora 配置，构建 lora 模型
    def get_lora_model(self) -> PeftModel:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.01,
            target_modules=["query_key_value"]
        )
        model = get_peft_model(self.base_model, config)
        model.to(LoRATuneConfig.device)
        return model

    # lora_finetune 执行 lora 微调
    def lora_finetune(self):
        dataloader = self._load_dataset()
        model = self.get_lora_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LoRATuneConfig.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=10,
            num_training_steps=(len(dataloader) * LoRATuneConfig.num_epochs),
        )
        t_loss = 0
        for epoch in range(LoRATuneConfig.num_epochs):
            model.train()
            for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), unit="batch"):
                optimizer.zero_grad()
                outputs = model(**batch, use_cache=False)
                loss = outputs.loss
                print("\n......loss......", loss.item())
                t_loss += loss.item()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if (step + 1) % 10 == 0:
                    print(f"\n mini-batch {step + 1} average loss: {round(t_loss / 10, 5)}")
                    t_loss = 0.0
        # 保存 lora 微调之后的模型以及tokenize 文件
        save_lora_model(model, self.tokenizer, LoRATuneConfig.peft_lora_file)

    # merge_lora_tuned_model 加载经过 lora 微调和原始base mode 模型进行合并
    def merge_lora_tuned_model(self):
        lora_model = PeftModel.from_pretrained(self.base_model, LoRATuneConfig.peft_lora_file,
                                               torch_dtype=torch.float16)
        merged_model = lora_model.merge_and_unload()
        # 移动到GPU
        merged_model.cuda()
        # 切换到推理模式
        merged_model.eval()
        return merged_model

    # chat 聊天会话
    def chat(self, query: str, history: List[Dict] = None):
        model = self.merge_lora_tuned_model()
        responses, past = model.chat(self.tokenizer, query, history)
        return responses, past


if __name__ == "__main__":
    loraTune = ChatGLM3LoRATuning()
    # loraTune.lora_finetune()
    query = "介绍一下你自己?"
    response, history = loraTune.chat(query)
    print(response, "\n", response)
