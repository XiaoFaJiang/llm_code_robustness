import pandas as pd

df = pd.read_excel("mbpp_cpp.xlsx")
for i in range(len(df)):
    if df['code_str'][i].count('iostream') > 0:
        continue
    df['code_str'][i] = '''#include<iostream>
    ''' + "\n" + df['code_str'][i]
df.to_excel('mbpp_cpp.xlsx',index=False)