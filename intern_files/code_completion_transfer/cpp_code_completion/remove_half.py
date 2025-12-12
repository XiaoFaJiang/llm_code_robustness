import pandas as pd
import os
import random
import re
pattern = re.compile(r'[A-Za-z\d]')
random.seed(42)
df = pd.read_excel("mbpp_cpp.xlsx")

deleted_flag = []
deleted_code = []
codes_be_deleted = []
prompt = []
for i in range(len(df)):
    st = df['code_str'][i]

    prompt.append(df['prompt'][i].replace('python','cpp'))
    def remove(st):
        st_line = st.strip().split("\n")
        st_line_real = []
        import_line = []
        using_namespace_line = []
        for i,v in enumerate(st_line):
            if v.strip():
                if v.startswith("#include"):
                    import_line.append(v)
                elif v.startswith("using"):
                    using_namespace_line.append(v)
                else:
                    st_line_real.append(v)
        length = len(st_line_real)
        is_deleted = 0
        st_line_real = st_line_real[:length//2] + st_line_real[-1]
        st_line_real[length//2] = "//begin to write code\n"
        is_deleted = 1
        return is_deleted,'\n'.join(import_line + using_namespace_line + st_line_real)
    
    is_deleted,adeleted_code = remove(st)
    deleted_code.append(adeleted_code)
    deleted_flag.append(is_deleted)



df['is_deleted'] = deleted_flag
df['code_str_deleted'] = deleted_code
df = df[df['is_deleted'] == 1]
df['cpp_prompt'] = prompt
df.to_excel("mbpp_cpp_generation.xlsx",index=False)

    