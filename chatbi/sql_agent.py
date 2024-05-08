# -*- coding: utf-8 -*-
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from db import DuckDB


# SQLAgent  完成文本到SQL 转换 并 执行SQL
class SQLAgent:
    def __init__(
            self,
            model_file: str,
            db_file: str = "dossen.duckdb",
            prompt_file: str = "./prompt/prompt.md",
            metadata_file: str = "./prompt/metadata.sql"
    ):
        self.duck_db = DuckDB(db_file)
        self.tokenizer, self.model = get_tokenizer_model(model_file)
        self.prompt, self.table_metadata = generate_prompt(prompt_file=prompt_file, metadata_file=metadata_file)

    # text2sql 将文本转换为SQL查询语句
    def text2sql(self, question: str) -> str:
        prompt = self.prompt.format(
            user_question=question, table_metadata_string=self.table_metadata
        )
        eos_token_id = self.tokenizer.eos_token_id
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=300,
            do_sample=False,
            return_full_text=False,  # added return_full_text parameter to prevent splitting issues with prompt
            num_beams=5,  # do beam search with 5 beams for high quality results
        )
        generated_query = (
                pipe(
                    prompt,
                    num_return_sequences=1,
                    eos_token_id=eos_token_id,
                    pad_token_id=eos_token_id,
                )[0]["generated_text"]
                .split(";")[0]
                .split("```")[0]
                .strip()
                + ";"
        )
        return generated_query

    # execute 将文本转化为sql, 然后在数据库中执行sql, 返回结果
    def execute_sql(self, question: str) -> pd.DataFrame:
        sql = self.text2sql(question)
        return self.duck_db.sql(sql).df()


def generate_prompt(prompt_file="prompt.md", metadata_file="metadata.sql"):
    with open(prompt_file, "r") as f:
        prompt = f.read()

    with open(metadata_file, "r") as f:
        table_metadata_string = f.read()

    return prompt, table_metadata_string


def get_tokenizer_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )
    return tokenizer, model


if __name__ == "__main__":
    pass
