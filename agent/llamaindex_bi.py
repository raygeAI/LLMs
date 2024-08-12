# -*- coding: utf-8 -*-
"""
ChatBI tongyi_max 实践;
"""
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.dashscope import DashScope, DashScopeGenerationModels
from llama_index.core.tools import QueryEngineTool, ToolMetadata

Settings.embed_model = OllamaEmbedding(
    model_name="quentinz/bge-large-zh-v1.5:latest",
    base_url="http://localhost:11434",
    # ollama_additional_kwargs={"mirostat": 0},
)

api_key = "sk-013c0ff07bd84d14a7db70f77e090f23"

llm = DashScope(
    model_name=DashScopeGenerationModels.QWEN_MAX, api_key=api_key
)
Settings.llm = llm



# 使用 create_engine函数建立连接
tables = ["hotel_transaction", "hotel_product_info", "hotel_status", "hotel_dim"]
engine = create_engine("duckdb:///./data/duck_db/dossen_test.duckdb", echo=True)
sql_database = SQLDatabase(engine, include_tables=tables)


def generate_prompt(prompt_file="prompt.md", metadata_file="metadata.sql"):
    with open(prompt_file, "r") as f:
        prompt = f.read()

    with open(metadata_file, "r", encoding="utf-8") as f:
        table_metadata_string = f.read()

    return prompt, table_metadata_string


prompt, table_metadata = generate_prompt("./data/prompt/prompt.md", "./data/prompt/metadata.sql")


llama_index_prompt = PromptTemplate(prompt, "text_to_sql")
p = llama_index_prompt.partial_format(table_metadata_string=table_metadata)

# 创建SQL 查询引擎
sql_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=tables,
    llm=llm,
    verbose=True,
    text_to_sql_prompt=p,
)

tools = [
    QueryEngineTool(
        query_engine=sql_engine,
        metadata=ToolMetadata(name="text2sql", description=("自然语言转换为 SQL 语句，执行SQL 查询")),
    ),
]

agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
y = agent.query("各个品牌酒店的销售综合销售间夜是多少?")
print("res: ", y)
