# -*- coding: utf-8 -*-
"""
Created by guorui
duckdb 数据库封装
"""

import duckdb
import pandas as pd


# get_object_name  获取python 对象的字符串名称
def get_object_name(var):
    """
    Returns the name of a variable as string
    """
    for name in globals():
        if globals()[name] is var:
            return name
    return None


# DuckDB  数据库封装
class DuckDB(object):
    def __init__(self, db_name):
        # 断言一下db_name 必须以 .duckdb 或者 .db 结尾
        assert db_name.endswith('.duckdb') or db_name.endswith(".db"), "db_name must has suffix .duckdb or .db"
        self.db = duckdb.connect(database=db_name)

    # execute 执行写入sql, 如 create， insert， update, delete 等
    def execute(self, sql: str):
        return self.db.execute(sql).fetchall()

    # sql 执行select 查询 sql 返回结果
    def sql(self, sql: str):
        return self.db.sql(sql)

    # show_create_table 导出建表语句
    def show_create_table(self, table_name: str) -> str:
        table_def = self.execute("PRAGMA table_info({})".format(table_name)).fetchall()
        # Generate CREATE TABLE statement
        create_table_stmt = "CREATE TABLE {} (\n".format(table_name)
        for col in table_def:
            create_table_stmt += "  {} {} {},\n".format(col[1], col[2], "PRIMARY KEY" if col[5] else "")
        create_table_stmt = create_table_stmt[:-2]  # remove last comma
        create_table_stmt += "\n);"
        return create_table_stmt

    # df_import 导入 dataframe 数据到表格中
    def df_import(self, df: pd.DataFrame, table_name: str):
        df_name = get_object_name(df)
        sql = "create tabel {} as select * from {}".format(table_name, df_name)
        self.execute(sql)

    # close 关掉数据库连接
    def close(self):
        self.db.close()
