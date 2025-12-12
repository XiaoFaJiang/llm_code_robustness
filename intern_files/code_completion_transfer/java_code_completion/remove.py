import pandas as pd
import os
import random
import re
pattern = re.compile(r'[A-Za-z\d]')
random.seed(42)
df = pd.read_excel("mbpp_java.xlsx")

deleted_flag = []
deleted_code = []
codes_be_deleted = []

for i in range(len(df)):
    st = df['code_str'][i]

    def remove(st):
        st_line = st.strip().split("\n")
        st_line_real = []
        import_line = []
        using_namespace_line = []
        for i,v in enumerate(st_line):
            if v.strip():
                if v.startswith("import"):
                    import_line.append(v)
                elif v.startswith("class"):
                    using_namespace_line.append(v)
                else:
                    st_line_real.append(v)
        length = len(st_line_real)
        is_deleted = 0
        codes = ""
        try:
            random.seed(42)
            random_number  = random.randint(1,length - 2)
            if not bool(pattern.search(st_line_real[random_number])):
                while not bool(pattern.search(st_line_real[random_number])):
                    random_number -= 1
                    if random_number == 1:
                        break
            if not bool(pattern.search(st_line_real[random_number])):
                while not bool(pattern.search(st_line_real[random_number])):
                    random_number += 1
                    if random_number == length - 2:
                        break
            if not bool(pattern.search(st_line_real[random_number])):
                raise Exception
            codes = st_line_real[random_number]
            st_line_real[random_number] = "// There is a line of code missing here.\n"
            is_deleted = 1
        except:
            pass
        return codes,is_deleted,'\n'.join(import_line + using_namespace_line + st_line_real)
    
    codes,is_deleted,adeleted_code = remove(st)
    codes_be_deleted.append(codes[:])
    deleted_code.append(adeleted_code)
    deleted_flag.append(is_deleted)



df['is_deleted'] = deleted_flag
df['code_str_deleted'] = deleted_code
df = df[df['is_deleted'] == 1]
df.to_excel("mbpp_java_completion.xlsx",index=False)

    