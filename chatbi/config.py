# sqlcoder-7b-2 模型文件路径
model_file = "../model/sqlcoder/sqlcode_7b_2"
# 数据库元数据文件路径，主要由建表语句，表 join 关系描述构成
metadata_file = "./prompt/metadata.sql"
# 数据库元数据文件路径，主要由建表语句，表 join 关系描述构成
simple_metadata_file = "./prompt/simple_metadata.sql"
# 任务提示词
prompts_file = "./prompt/prompt.md"
# DuckDB 数据库文件位置
db_file = "./data/duck_db/dossen.duckdb"

# 测试场景下 DuckDB 数据库文件位置
db_file_test = "./data/duck_db/dossen_test.duckdb"
