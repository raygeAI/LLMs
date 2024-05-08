# -*- coding: utf-8 -*-
import streamlit as st
from sql_agent import SQLAgent
import config

# 初始化 sql agent 对象
sql_agent = SQLAgent(
    config.model_file,
    config.db_file,
    config.prompt_file,
    config.metadata_file
)

st.title("东呈集团 ChatBI ")
question = st.text_input("请输入您的问题")
if question is not None:
    data = sql_agent.execute_sql(question)
    st.write(data)


if __name__ == "__main__":
    pass

