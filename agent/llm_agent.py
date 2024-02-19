# -*- coding: utf-8 -*-
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, LLMMathChain
from langchain.agents import load_tools, AgentType, initialize_agent
from langchain.utilities import SerpAPIWrapper
from langchain.tools import BaseTool, Tool
from peft import PeftModel
from util import set_random_seed

# model_file = "../model/ChatGLM/chatGLM3"
# model_file = "../model/01_ai/yi-34b-4bit"
# model_file = "../model/microsoft/phi-2"
model_file = "../model/mistral/Mistral-7B-Instruct-v0.2"

set_random_seed(31)

os.environ["SERPAPI_API_KEY"] = "617ac3100516f1f058856f52ae2c0c31ab8067d659f065fd6ad668cb570c54f0"

tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_file, trust_remote_code=True, torch_dtype='auto', device_map="auto")
model = model.eval()

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

lora_file = "../finetune/ChatGLM/model/finetune/ChatGLM3/lora"


def merge_lora_tuned_model():
    lora_model = PeftModel.from_pretrained(model, lora_file,
                                           torch_dtype=torch.float16)
    merged_model = lora_model.merge_and_unload()
    # 移动到GPU
    merged_model.cuda()
    # 切换到推理模式
    merged_model.eval()
    return merged_model


# model = merge_lora_tuned_model()

pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
    max_length=3200,
    do_sample=True,
    # temperature=0.2,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=pipe)

# chain = LLMChain(llm=llm, prompt=prompt)
# llm_math_chain = LLMMathChain(llm=llm, verbose=True)
llm_math_chain = LLMMathChain.from_llm(llm, verbose=True)

search = SerpAPIWrapper()

# print(llm_math_chain.run("What is 13 raised to the .3432 power?"))
from pydantic import BaseModel, Field


class CalculatorInput(BaseModel):
    question: str = Field()


# tools = [
#     Tool.from_function(
#         func=search.run,
#         name="Search",
#         description="useful for when you need to answer questions about current events"
#     ),
#     Tool.from_function(
#         func=llm_math_chain.run,
#         name="Calculator",
#         description="useful for when you need to answer questions about math",
#         args_schema=CalculatorInput
#     )
# ]
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# 如果必须使用 sdp_kernel 上下文管理器，请使用 memory efficient 或 math 内核
with torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=True
):
    # Now let's test it out!
    agent.run(
        "What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the  0.4 power?")
