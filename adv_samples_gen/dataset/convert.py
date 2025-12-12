import json
import os
#读取json文件

data = []
file_catagory = "valid_base"
with open(f"{file_catagory}.jsonl","r") as f:
    for oneline in f:
        data.append(json.loads(oneline))

func_name = {'cpp':[],'python':["def" for _ in range(len(data))],'java':[],'javascript':[]}

for lang in ['cpp','java','javascript']:
    with open(os.path.join("/data1/ljc/code/llm_robustness_eval_and_enhance/intern_files/dataset/generate",f"{lang}_func_name.jsonl"),"r") as f:
        for line in f:
            func_name[lang].append(json.loads(line)[0])


descriptions = {'cpp':[],'python':[],'java':[],'javascript':[]}

for lang in ['cpp','python','java','javascript']:
    with open(os.path.join("/data1/ljc/code/llm_robustness_eval_and_enhance/bigcode-evaluation-harness/dataset",f"mbpp_{lang}_tested.json")) as f:
        x = json.load(f)
        for oneline in x:
            descriptions[lang].append(oneline[f'{lang}_prompt'])


count = 0

f_data = []
for index,onecode in enumerate(data):
    code = onecode['Adversarial Code']
    codelines = code.split("\n")
    lang = onecode['lang']
    task_id = onecode['task_id']
    description = descriptions[lang][task_id]
    doc_first = "'''" if lang == "python" else "/*"
    doc_second = "'''" if lang == "python" else "*/"
    f = False
    for i,v in enumerate(codelines):
        if func_name[lang][task_id] in v:
            f = True
            count += 1
            indent = ""
            for x in codelines[i+1]:
                if x == ' ' or x == '\t':
                    indent += x
                else:
                    break
            if not indent:
                indent = "    "
            prompt = f"""
{indent}{doc_first}
{indent}progame language:{lang}
{indent}description:{description}
{indent}you must follow:
{indent}1. Provide the complete code without any textual explanations and do not generate test scripts.
{indent}2. Please strictly follow the specified format provided below for the code.
{indent}3. Do not change the function names.
{indent}4. The original code content must be fully included in the generated complete code, including all package import sections.
{indent}5. For C++ language, do not generate the main function; I have my own main function available.
{indent}6. Do not generate test cases.
{indent}{doc_second}
""" 
            codelines = codelines[:i+1] + [prompt] + codelines[i+1:]
            break
    if f:
        xx = {}
        xx['code_str_generate'] = "\n".join(codelines)
        xx['Adversarial truth'] = data[index]['Adversarial truth']
        f_data.append(xx)

print(count)
with open(f"{file_catagory}.jsonl","w") as f:
    for oneline in f_data:
        f.write(json.dumps(oneline) + "\n")
        