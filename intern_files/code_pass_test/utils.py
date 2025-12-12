import json
from .directly_call_openai import call
import time

def read_data(data_path):
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            data = json.loads(line)

            data_list.append(data)
    return data_list

def write_data(data_list, file_name):
    with open('./code/'+file_name, 'w', encoding='utf-8') as file:
        for one_data in data_list:
            json_str = json.dumps(one_data, ensure_ascii=False)
            # 写入JSON字符串并在末尾添加换行符
            file.write(json_str + '\n')
    print('数据存储完成!')
    
def generate_answer_useGpt4(prompt, max_temp=3):
    for i in range(max_temp):
        #采用贪婪策略
        answer = call(prompt, 0, 4000)
        if answer.data:
            return answer.data[0]
        time.sleep(0.5)
    return ''


import re
def extract_python_code(generation: str):
   generation = generation.replace("[PYTHON]", '```python').replace("[/PYTHON]", '```')

   pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
   matches = pattern.findall(generation)
   if len(matches) >= 1:
       code_block = matches[0]
       return code_block

   sep_index = generation.find("```")
   if sep_index != -1:
       pattern = re.compile(r"```\n(.*?)```", re.DOTALL)
       matches = pattern.findall(generation)
       if len(matches) >= 1:
           code_block = matches[0]
           return code_block
       elif generation[sep_index + len("```") : sep_index + len("```python")] == "python":
           generation = generation[sep_index + len("```python") :]
       else:
           generation = generation[sep_index + len("```") :]
   return generation


def construct_exe_string(code_str='', func_name='', test_str=''):
    final_test_str = code_str+ '\n' + test_str + f'\ncheck({func_name})\n'
    return final_test_str

def construct_exe_string_mbpp(code_str='', test_str=''):
    final_test_str = code_str+ '\n' + test_str
    return final_test_str

def extract_func_name_from_code(code_str):
    # 从代码字符串中提取出函数名
    pattern = r"def (.*?):"
    match = re.search(pattern, code_str, re.DOTALL)
    if match:
        func_name = match.group(1)
        return 1, func_name
    return 0, ''



def remove_special_string(code_str):
    code_str = code_str.replace('"""', '')
    return code_str

