import json
import os
from collections import Counter

BASE_MODELS = ['codellama-7b-base','deepseek-v2-lite-base',\
                'llama3.1-8b-base','llama3.2-1b-base','Qwen2.5-7b-base','Qwen2.5-coder-0.5b-base',\
                    'Qwen2.5-coder-1.5b-base','Qwen2.5-coder-3b-base','Qwen2.5-coder-7b-base',"codeQwen1.5-7B-base"]

INSTRUCT_MODELS = ['codellama-7b-instruct','deepseek-v2-lite-chat','llama3.1-8b-instruct',\
        'llama3.2-1b-instruct','qwen2.5-7b-instruct','qwen2.5-coder-0.5b-instruct',\
            'qwen2.5-coder-1.5b-instruct','qwen2.5-coder-3b-instruct','qwen2.5-coder-7b-instruct']

def read(model_name = "",languages = [""]):
    model_type = "instruct" if model_name.endswith("instruct") or model_name.endswith("chat") else "base"
    base_dir = "result"
    eva_results_dir = os.path.join(base_dir,model_name,"evaluation_results")
    perturbations = ["code_stmt_exchange","code_style","insert","rename","code_expression_exchange"]
    perturbation_success_map = {x:[] for x in perturbations}
    for lang in languages:
        for p in perturbations:
            perturbation_dir = f"mbpp_generate_{lang}_robust_{p}" + ("_instruct" if model_type == "instruct" else "")
            e_c_path = os.path.join(eva_results_dir,perturbation_dir,"evaluation_cases.json")
            if not os.path.exists(e_c_path):
                continue
            evaluation_cases = json.load(open(e_c_path))
            original_dir = f"mbpp_generate_{lang}_robust_no_change" + ("_instruct" if model_type == "instruct" else "")
            o_c_path = os.path.join(eva_results_dir,original_dir,"evaluation_cases.json")
            if not os.path.exists(o_c_path):
                continue
            original_cases = json.load(open(o_c_path))
            for i in range(len(evaluation_cases)):
                e_mini_cases = evaluation_cases[str(i)]
                task_id = e_mini_cases["task_id"]
                o_mini_cases = original_cases[str(task_id)]
                assert task_id == o_mini_cases['task_id']
                if e_mini_cases['passed'] == False and o_mini_cases['passed'] == True:
                    perturbation_success_map[p].append(e_mini_cases['perturbation_type'])
    return perturbation_success_map

if __name__ == '__main__':
    perturbations = ["code_stmt_exchange","code_style","insert","rename","code_expression_exchange"]
    languages = ["cpp","python","java","javascript"]
    BASE_PERTURBATION_SUCCESS_MAP = {lang : {x:[] for x in perturbations} for lang in languages}
    for lang in languages:
        for m in BASE_MODELS:
            x = read(m,[lang])
            for k,v in x.items():
                BASE_PERTURBATION_SUCCESS_MAP[lang][k].extend(v)

    
    for lang in languages:
        json.dump(BASE_PERTURBATION_SUCCESS_MAP[lang],open(f"perturbation_compare/base_{lang}_perturbation_success_map.json","w"),indent=4)

    INSTRUCT_PERTURBATION_SUCCESS_MAP = {lang : {x:[] for x in perturbations} for lang in languages}
    for lang in languages:
        for m in INSTRUCT_MODELS:
            x = read(m,[lang])
            for k,v in x.items():
                INSTRUCT_PERTURBATION_SUCCESS_MAP[lang][k].extend(v)

    for lang in languages:
        json.dump(INSTRUCT_PERTURBATION_SUCCESS_MAP[lang],open(f"perturbation_compare/instruct_{lang}_perturbation_success_map.json","w"),indent=4)

