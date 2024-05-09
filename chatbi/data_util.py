# -*- coding: utf-8 -*-
import pandas as pd
from db import DuckDB
import config

duckdb = DuckDB(config.db_file_test)

table_file_dict = {
    "hotel_transaction": "./data/bi_dw.dwd_t_hoteltransactiontheme.xlsx",
    "hotel_product_info": "./data/bi_dw.dwd_t_hotelproducttheme.xlsx",
    "hotel_status": "./data/bi_dim.Dim_Hotel_MetLabel.xlsx",
    "hotel_dim": "./data/bi_dim.Dim_Hotel_curr.xlsx",
}

for table_name, file in table_file_dict.items():
    df = pd.read_excel(file)
    columns = df.columns.tolist()
    columns = [c.split(".")[1] for c in columns]
    df.columns = columns
    duckdb.df_import(df, table_name)
    table_desc = duckdb.show_create_table(table_name)
    print(table_desc)
    print("=" * 10, "\n")
