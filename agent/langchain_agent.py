# -*- coding: utf-8 -*-
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Tongyi
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools.retriever import create_retriever_tool


documents = DirectoryLoader("./data", glob=["hotel_0010002.txt", "hotel_0010003.txt"]).load()
meta_info = ["0010002", "0010003"]
for id, doc in enumerate(documents):
    doc.metadata = {"hotel_id": meta_info[id]}

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
all_splits = text_splitter.split_documents(documents)
persist_directory = "./data/cache"
model = "yi:9b"

embedding_model = OllamaEmbeddings(
    model="quentinz/bge-large-zh-v1.5:latest",
    base_url="http://localhost:11434",
    # ollama_additional_kwargs={"mirostat": 0},
)

vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embedding_model,
    persist_directory=persist_directory,
)

# Try reloading index from disk and using for search:
# llm = Ollama(
#     base_url="http://localhost:11434",
#     model=model,
#     verbose=True,
# )
api_key = "sk-013c0ff07bd84d14a7db70f77e090f23"
llm = Tongyi(
    model_name="qwen-max",
    dashscope_api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
)

# 添加元数据过滤
retriever = vectorstore.as_retriever(search_kwargs={"filter": {"hotel_id":"0010003"}})

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="hotel",
    description="Query a retriever to get information about the hotel infrastructure"
)
tools = [retriever_tool]

react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)
executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True, max_iterations=5)
y = executor.invoke({"input": "酒店WiFi 秘密是多少?"})
print(y)
