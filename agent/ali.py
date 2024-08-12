# -*- coding: utf-8 -*-
from http import HTTPStatus
import dashscope
from langchain_community.llms import Tongyi

api_key = "sk-013c0ff07bd84d14a7db70f77e090f23"


def langchain_call():
    llm = Tongyi(
        model_name="qwen-max",
        dashscope_api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
    )
    out = llm.invoke("介绍一下广州")
    print(out)


def call_with_stream():
    messages = [{'role': 'user', 'content': 'Introduce the capital of China'}]
    responses = dashscope.Generation.call("qwen-max",
                                          messages=messages,
                                          result_format='message',  # set the result to be "message"  format.
                                          stream=True,  # set streaming output
                                          incremental_output=True,  # get streaming output incrementally
                                          api_key=api_key,
                                          )
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            print(response.output.choices[0]['message']['content'], end='')
        else:
            print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
                response.request_id, response.status_code, response.code, response.message))


from openai import OpenAI


def get_response():
    client = OpenAI(
        api_key=api_key,
        # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
    )
    completion = client.chat.completions.create(
        model="qwen-max",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': '你是谁？'}]
    )
    print(completion.model_dump_json())


if __name__ == '__main__':
    pass
