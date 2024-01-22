# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ChatGLM3PromptDataSet 构建训练数据集合
class ChatGLM3PromptDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, is_long_skip):
        self.all_data = []
        skip_data_number = 0
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                skip_flag = False

                src_tokens = [tokenizer.get_command("<|user|>")] + tokenizer.encode("\n", add_special_tokens=False) + tokenizer.encode(sample["instruction"] + sample["input"], add_special_tokens=False)
                if len(src_tokens) > max_src_len:
                    # 当输入内容超长时，随向后截断
                    src_tokens = src_tokens[:max_src_len]
                    skip_flag = True

                max_tgt_len = max_len - 6 - len(src_tokens)
                tgt_tokens = [tokenizer.get_command("<|assistant|>")] + tokenizer.encode("\n", add_special_tokens=False) + tokenizer.encode(
                    sample["output"], add_special_tokens=False)

                if len(tgt_tokens) > max_tgt_len:
                    # 当输出内容超长时，随向后截断
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                    skip_flag = True

                # ChatGLM3需要增加[gMASK]、sop两个标记，构造如下的训练数据格式
                # [gMASK][SOP] < | system | > '.'.join( ocr_texts) < | user | > question < | assistant | > answer < | eos | >
                input_ids = [tokenizer.get_command("[gMASK]"), tokenizer.get_command("sop")] + src_tokens + tgt_tokens + [tokenizer.eos_token_id]
                context_length = len(src_tokens) + 2
                # answer 下文设置损失
                labels = [-100] * context_length + input_ids[context_length:]

                assert len(input_ids) == len(labels)
                assert len(input_ids) <= max_len
                if is_long_skip and skip_flag:
                    skip_data_number += 1
                    continue
                self.all_data.append({"input_ids": input_ids, "labels": labels})
        print("the number of skipping data is {}".format(skip_data_number))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, id):
        item = self.all_data[id]
        return item


# ChatGLM3PrefixDataSet 构建训练数据集合
class ChatGLM3PrefixDataSet(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, is_long_skip):
        self.all_data = []
        skip_data_number = 0
        with open(data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                sample = json.loads(line.strip())
                skip_flag = False

                src_tokens = [tokenizer.get_command("<|user|>")] + tokenizer.encode("\n", add_special_tokens=False) + tokenizer.encode(sample["content"], add_special_tokens=False)
                if len(src_tokens) > max_src_len:
                    # 当输入内容超长时，随向后截断
                    src_tokens = src_tokens[:max_src_len]
                    skip_flag = True

                max_tgt_len = max_len - 6 - len(src_tokens)
                tgt_tokens = [tokenizer.get_command("<|assistant|>")] + tokenizer.encode("\n", add_special_tokens=False) + tokenizer.encode(sample["summary"], add_special_tokens=False)

                if len(tgt_tokens) > max_tgt_len:
                    # 当输出内容超长时，随向后截断
                    tgt_tokens = tgt_tokens[:max_tgt_len]
                    skip_flag = True

                # ChatGLM3需要增加[gMASK]、sop两个标记
                input_ids = [tokenizer.get_command("[gMASK]"), tokenizer.get_command("sop")] + src_tokens + tgt_tokens + [tokenizer.eos_token_id]
                context_length = len(src_tokens) + 2
                labels = [-100] * context_length + input_ids[context_length:]

                assert len(input_ids) == len(labels)
                assert len(input_ids) <= max_len
                if is_long_skip and skip_flag:
                    skip_data_number += 1
                    continue
                self.all_data.append({"input_ids": input_ids, "labels": labels})
        print("the number of skipping data is {}".format(skip_data_number))

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, id):
        item = self.all_data[id]
        return item


# DataCollate dataset 数据转换
class DataCollate:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch):
        lengths = [len(instance["input_ids"]) for instance in batch]
        batch_max_len = max(lengths)

        input_ids_batch, labels_batch = [], []
        for instance in batch:
            input_ids = instance["input_ids"]
            labels = instance["labels"]

            padding_len = batch_max_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * padding_len
            # -100 是计算损失函数时候，忽略掉这样的数据
            labels = labels + [-100] * padding_len

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)

        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long).to(device)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long).to(device)

        return {"input_ids": input_ids_batch, "labels": labels_batch}
