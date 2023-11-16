# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModel

model_file = "../model/ChatGLM/chatGLM3"

tokenizer = AutoTokenizer.from_pretrained(model_file,  trust_remote_code=True)
model = AutoModel.from_pretrained(model_file, trust_remote_code=True).half().cuda()
model.eval()

input = "尝试比较一下1和2的大小"

response, history = model.chat(tokenizer, input, history=[])
print(response)