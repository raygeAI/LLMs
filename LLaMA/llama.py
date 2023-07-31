import torch
from transformers import LlamaTokenizer, AutoModelForCausalLM

tokenizer = LlamaTokenizer.from_pretrained("syzymon/long_llama_3b")
model = AutoModelForCausalLM.from_pretrained("syzymon/long_llama_3b", torch_dtype=torch.float32, trust_remote_code=True).to("cuda:0")

prompt = "A long time ago in a galaxy far, far away... there was this long, long way from here."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
outputs = model(input_ids=input_ids)
generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=256,
    num_beams=1,
    last_context_length=1792,
    do_sample=True,
    temperature=1.0,
)
print(tokenizer.decode(generation_output[0]))