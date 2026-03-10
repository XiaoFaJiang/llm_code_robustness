import json
import os
#读取json文件

data = []
file_catagory = "valid_dpo"
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
skipped_same_truth_rejected = 0
skipped_func_not_found = 0

f_data = []
for index,onecode in enumerate(data):
    code = onecode['adv_code']
    codelines = code.split("\n")
    lang = onecode['lang']
    task_id = onecode['task_id']
    description = descriptions[lang][task_id]

    # CRITICAL: Verify adv_truth != adv_prediction before processing (Bug 2 fix)
    adv_truth = onecode.get('adv_truth', '')
    adv_prediction = onecode.get('adv_prediction', '')
    if adv_truth == adv_prediction:
        skipped_same_truth_rejected += 1
        print(f"Skipping sample {index}: adv_truth == adv_prediction (length={len(adv_truth)})")
        continue

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
        xx['adv_truth'] = adv_truth  # Use pre-extracted value
        xx["adv_rejected"] = adv_prediction  # Use pre-extracted value
        f_data.append(xx)
    else:
        skipped_func_not_found += 1

print("\n" + "="*60)
print("CONVERT.PY STATISTICS")
print("="*60)
print(f"Input samples:                {len(data)}")
print(f"Successfully converted:       {count}")
print(f"Skipped - Same truth/rejected: {skipped_same_truth_rejected}")
print(f"Skipped - Function not found:  {skipped_func_not_found}")
print(f"Output samples:               {len(f_data)}")
print("="*60)

with open(f"{file_catagory}_base.jsonl","w") as f:
    for oneline in f_data:
        f.write(json.dumps(oneline) + "\n")

print(f"\nSuccessfully saved to {file_catagory}.jsonl")
        