# -*- coding: utf-8 -*-
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda:0" if torch.cuda.is_available() else "cpu"
n_steps = 256
num_beams = 5


class GPT2:
    def __init__(self, model_name: str = "gpt2-xl"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def greedy_search_generate(self, input_text: str):
        input_ids = self.tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)
        output = self.model.generate(
            input_ids,
            pad_token_id=50256,
            max_length=n_steps,
            no_repeat_ngram_size=2,
            early_stopping=True)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output

    def beam_search(self, input_text: str):
        input_ids = self.tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)
        output = self.model.generate(
            input_ids,
            max_length=n_steps,
            pad_token_id=50256,
            no_repeat_ngram_size=2,
            num_beams=num_beams,
            early_stopping=True
        )
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output


if __name__ == "__main__":
    input_text = "I enjoy walking with my cute dog"
    gpt = GPT2()
    print("greed_search:\n", gpt.greedy_search_generate(input_text))
    print("beam search:\n", gpt.beam_search(input_text))
