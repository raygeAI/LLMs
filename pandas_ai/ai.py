# -*- coding: utf-8 -*-
import pandas as pd
from pandasai import SmartDataframe, Agent
from pandasai import SmartDatalake
from langchain_community.llms import Ollama


llm = Ollama(model="llama3.1:latest", temperature=0)

sales_by_country = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "sales": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000]
})

employees_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Name': ['John', 'Emma', 'Liam', 'Olivia', 'William'],
    'Department': ['HR', 'Sales', 'IT', 'Marketing', 'Finance']
}

salaries_data = {
    'EmployeeID': [1, 2, 3, 4, 5],
    'Salary': [5000, 6000, 4500, 7000, 5500]
}

employees_df = pd.DataFrame(employees_data)
salaries_df = pd.DataFrame(salaries_data)

sdl = SmartDatalake([employees_df, salaries_df], config={"llm": llm})

res = sdl.chat("谁获得了最高的薪水？")
print(res)
sdf = SmartDataframe(sales_by_country, config={"llm": llm})
res = sdf.chat("销量排名前5的国家是哪些？")
print(res)


