# -*- coding: utf-8 -*-

import torch
import bitsandbytes
from dataclasses import dataclass
from typing import Dict, List
from trl import DPOTrainer
from datasets import Dataset, load_dataset
from transformers import TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from utils import set_random_seed


def get_dataset():
    dataset = load_dataset("json", data_files="./data/harmless_base_cn_train.jsonl")
    print(dataset)
    train_val = dataset["train"].train_test_split(test_size=2000, shuffle=True, seed=42)
    train_data = train_val["train"]
    val_data = train_val["test"]

    def extract_anthropic_prompt(prompt_and_response):
        final = ""
        for sample in prompt_and_response:
            if sample["role"] == "human":
                sample["role"] = "<|user|>"
            final += sample["role"] + "\n" + sample["text"]
        final += "\n"
        return final

    def get_hh(dataset, sanity_check: bool = False) -> Dataset:
        """
        The dataset is converted to a dictionary with the following structure:
        {
            'prompt': List[str],
            'chosen': List[str],
            'rejected': List[str],
        }
        """
        if sanity_check:
            dataset = dataset.select(range(min(len(dataset), 1000)))

        def split_prompt_and_responses(sample) -> Dict[str, str]:
            prompt = extract_anthropic_prompt(sample["context"])
            if sample["chosen"]["role"] == "assistant":
                sample["chosen"]["role"] = "<|assistant|>"
            # rejected
            if sample["rejected"]["role"] == "assistant":
                sample["rejected"]["role"] = "<|assistant|>"

            return {
                "prompt": prompt,
                "chosen": sample["chosen"]["role"] + "\n" + sample["chosen"]["text"],
                "rejected": sample["rejected"]["role"] + "\n" + sample["rejected"]["text"],
            }

        return dataset.map(split_prompt_and_responses)

    train_dataset = get_hh(train_data, sanity_check=True)
    eval_dataset = get_hh(val_data, sanity_check=True)
    return train_dataset, eval_dataset


# DPOTuningConfig 配置
@dataclass
class DPOTuningConfig:
    train_datafile: str = "./data/dpo/hotel_train.jsonl"
    test_datafile: str = "./data/dpo/hotel_test.jsonl"
    base_model_file: str = "../../model/ChatGLM/chatGLM3"
    peft_dpo_lora_file: str = "./model/finetune/ChatGLM3/dpo_tuning"
    training_args: TrainingArguments = TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=250,
        remove_unused_columns=False,
        gradient_accumulation_steps=2,
        learning_rate=1.5e-3,
        evaluation_strategy="steps",
        eval_steps=10,
        # 参考bert 预热率10%，一般在 0.1~0.25 之间
        warmup_ratio=0.15,
        weight_decay=0.0001,
        log_level="info",
        logging_steps=2,
        # label_smoothing_factor=0.001,
        output_dir=peft_dpo_lora_file,
    )
    lora_config: LoraConfig = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.005,
        target_modules=[
            "query_key_value",
            "dense_h_to_4h",
            # "dense_4h_to_h",
            # "dense",
        ],
        task_type="CAUSAL_LM",
        # bias="none",
    )


# DPOTuning RLHF 算法的改进算法 DPO (Direct Preference Optimization)
class DPOTuning:
    def __init__(self):
        self.base_model = self.__load_base_model()
        self.tokenizer = AutoTokenizer.from_pretrained(DPOTuningConfig.base_model_file, trust_remote_code=True)

    # __load_base_model 加载基础模型
    @staticmethod
    def __load_base_model():
        # 设置量化配置
        base_model = AutoModelForCausalLM.from_pretrained(
            DPOTuningConfig.base_model_file,
            # load_in_8bit=True,
            trust_remote_code=True,
            device_map="auto")
        # base_model.half()
        return base_model

    # __load_dataset 构建训练数据集合
    @staticmethod
    def __load_dataset(data_file):
        # 准备训练数据
        data = load_dataset("json", data_files=data_file, split="train")
        items = []
        for d in data:
            item = [{k: v[i] for k, v in d.items() if i < len(v)} for i in range(max(len(v) for v in d.values()))]
            items.extend(item)
        dataset = Dataset.from_list(items)

        # transform
        def transform(ele):
            for k, text in ele.items():
                if k == "prompt":
                    ele[k] = "<|user|>" + " \n " + text
                else:
                    ele[k] = "<|assistant|>" + " \n " + text
            return ele

        dataset.map(transform)
        return dataset

    # __get_linear_module 获取线性模块
    def __get_linear_module(self):
        linear_class = bitsandbytes.nn.Linear8bitLt
        lora_module_names = set()
        for name, module in self.base_model.named_modules():
            if isinstance(module, linear_class):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        # needed for 16-bit
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    # __get_lora_model 构造部分参数微调的模型，基于 LoRA 方式
    def __get_lora_model(self):
        model = get_peft_model(self.base_model, DPOTuningConfig.lora_config)
        return model

    # merge_lora_tuned_model 合并经过dpo 微调之后的 lora 模型
    def merge_lora_tuned_model(self):
        lora_model = PeftModel.from_pretrained(self.base_model, DPOTuningConfig.peft_dpo_lora_file,
                                               torch_dtype=torch.float16)
        merged_model = lora_model.merge_and_unload()
        # 移动到GPU
        merged_model.cuda()
        # 切换到推理模式
        merged_model.eval()
        return merged_model

    # dpo_tune 对 LoRA 模型进行 DPO 算法微调
    def dpo_tune(self):
        # 带微调的peft_model
        peft_model = self.__get_lora_model()
        # 获取训练数据验证数据集
        train_data = self.__load_dataset(DPOTuningConfig.train_datafile)
        test_data = self.__load_dataset(DPOTuningConfig.test_datafile)

        # 定义dpo训练器
        dpo_trainer = DPOTrainer(
            model=peft_model,
            # ref_model=self.base_model,
            args=DPOTuningConfig.training_args,
            beta=0.1,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=self.tokenizer,
        )
        dpo_trainer.train()
        dpo_trainer.save_model()

    def chat(self, query: str, history: List[Dict] = None):
        model = self.merge_lora_tuned_model()
        responses, past = model.chat(self.tokenizer, query, history)
        return responses, past


if __name__ == "__main__":
    set_random_seed(29)
    dpoTuning = DPOTuning()
    dpoTuning.dpo_tune()
    query = "介绍一下你自己?"
    response, history = dpoTuning.chat(query)
    print(response, "\n", history)
