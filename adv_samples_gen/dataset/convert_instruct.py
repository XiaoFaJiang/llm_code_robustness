


import json
import os
#读取json文件

data = []
file_catagory = "train"
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

f_data = []
for index,onecode in enumerate(data):
    code = onecode['Adversarial Code']
    lang = onecode['lang']
    task_id = onecode['task_id']
    description = descriptions[lang][task_id]

    # CRITICAL: Verify adv_truth != adv_prediction before processing (Bug 2 fix)
    adv_truth = onecode.get('Adversarial truth', '')
    adv_prediction = onecode.get('adv_prediction', '')
    if adv_truth == adv_prediction:
        skipped_same_truth_rejected += 1
        print(f"Skipping sample {index}: adv_truth == adv_prediction (length={len(adv_truth)})")
        continue

    x = ""
    count += 1
    if lang == "cpp":
        x = "5. Do not generate a main function, as I have my own main function available."
    elif lang == "java":
        x = "5. Do not modify class \"Solution\" as a public class."
    elif lang == "python":
        x = "5. Mind indent in python code."
    elif lang == "javascript":
        x = "5. Do not generate \"console.log\" statement, do not use \"require\" to import package."
    prompt = f"""
This is a code generation task. Please help me write the code. The programming language for the code is {lang}. In the code, I have already provided a portion of it, and the remaining part needs to be completed by you. The placeholder 'begin to write code' is where you begin to complete the code.
The prompt for the code is: {description}
The code content is:
-----------------------------
{code}
-----------------------------

Requirements:
1. I only need the function and related package import, don't generate any other imformations such as examples usage or test cases.
2. Follow the specified format strictly below.
3. Do not change the function name.
4. The original code content must be fully included in the complete code you generate.
{x} 

Format:
```{lang}
Complete code (including all the content of the code I provided and the code you generated)
```
"""
    xx = {}
    xx['prompt'] = prompt
    xx['response'] = adv_truth  # Use pre-extracted value
    f_data.append(xx)

print("\n" + "="*60)
print("CONVERT_INSTRUCT.PY STATISTICS")
print("="*60)
print(f"Input samples:                {len(data)}")
print(f"Successfully converted:       {count}")
print(f"Skipped - Same truth/rejected: {skipped_same_truth_rejected}")
print(f"Output samples:               {len(f_data)}")
print("="*60)

with open(f"{file_catagory}_instruct.jsonl","w") as f:
    for oneline in f_data:
        f.write(json.dumps(oneline) + "\n")

print(f"\nSuccessfully saved to {file_catagory}.jsonl")
        