import pandas as pd
import os
import random
import re
pattern = re.compile(r'[A-Za-z\d]')
random.seed(42)
os.chdir("/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/liujincheng06/code_completion_transfer/python_code_completion")
df = pd.read_excel("mbpp_python.xlsx")

deleted_flag = []
deleted_code = []
codes_be_deleted = []

for i in range(len(df)):
    st = df['code_str'][i]

    def remove(st):
        st_line = st.strip().split("\n")
        st_line_real = []
        import_line = []
        func_head_line = []
        for i,v in enumerate(st_line):
            if v.strip():
                if v.startswith("import"):
                    import_line.append(v)
                elif v.strip().startswith("def") and len(func_head_line) == 0:
                    func_head_line.append(v.strip())
                else:
                    if v.strip():
                        st_line_real.append(v)
        length = len(st_line_real)
        st_line_real = st_line_real[:length//2]
        indent = ""
        if st_line_real:
            for v in st_line_real[-1]:
                if v == ' ' or v == '\t':
                    indent += v
                else:
                    break
        st_line_real.append(indent + "#begin to write code\n")
        is_deleted = 1
        return is_deleted,'\n'.join(import_line + func_head_line + st_line_real)
    
    is_deleted,adeleted_code = remove(st)
    deleted_code.append(adeleted_code)
    deleted_flag.append(is_deleted)


df['is_deleted'] = deleted_flag
df['code_str_deleted'] = deleted_code
df = df[df['is_deleted'] == 1]
df.to_excel("mbpp_python_completion.xlsx",index=False)