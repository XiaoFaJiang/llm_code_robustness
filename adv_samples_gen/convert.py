import json
#读取json文件

lang = "python"
with open(f"mbpp_{lang}_tested.json","r") as f:
    data = json.load(f)

func_name = []
if lang != "python":
    with open(f"{lang}_func_name.jsonl","r") as f:
        for line in f:
            func_name.append(json.loads(line)[0])
else:
    func_name = ["def" for _ in range(len(data))]

count = 0

doc_first = "'''" if lang == "python" else "/*"
doc_second = "'''" if lang == "python" else "*/"


for index,onecode in enumerate(data):
    code = onecode['code_str_deleted']
    codelines = code.split("\n")
    description = onecode[f"{lang}_prompt"]
    
    for i,v in enumerate(codelines):
        if func_name[index] in v:
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
            count += 1
            break
    data[index]["code_str_generate"] = "\n".join(codelines)

print(count)
with open(f"mbpp_{lang}_tested.json","w") as f:
    json.dump(data,f,indent = 4)   
        