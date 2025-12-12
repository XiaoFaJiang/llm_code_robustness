from evaluate import load
import os
import argparse
import json
import sys
from math import inf
import re
sys.path.append("../perturbation_pipeline")
from pipeline import PerturbationPipeline

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/usr/local/lib64'

'''
根据结果文件中的cases判断apply perturbation部分的pass@1是多少,然后计算pass-drop@1
'''
if __name__ == '__main__':


    MODELS = ['codellama-7b-instruct','deepseek-v2-lite-chat','llama3.1-8b-instruct',\
        'llama3.2-1b-instruct','qwen2.5-7b-instruct','qwen2.5-coder-0.5b-instruct',\
            'qwen2.5-coder-1.5b-instruct','qwen2.5-coder-3b-instruct','qwen2.5-coder-7b-instruct',\
            'codellama-7b-base','deepseek-v2-lite-base',\
                'llama3.1-8b-base','llama3.2-1b-base','Qwen2.5-7b-base','Qwen2.5-coder-0.5b-base',\
                    'Qwen2.5-coder-1.5b-base','Qwen2.5-coder-3b-base','Qwen2.5-coder-7b-base']
    
    PERTURBATION = ['code_style',"insert","rename","code_stmt_exchange","code_expression_exchange"]
    LANGUAGES = ['python','cpp','java','javascript']

    for model_name in ['qwen2.5-coder-0.5b-instruct-prompt']:
        for perturbation in ['code_stmt_exchange']:
            for language in LANGUAGES:
                for prompt_type in ["1random_prompt","3random_prompt","5random_prompt"]:
                    model_type = "causal_base" if model_name.endswith("base") else "causal_chat"
                    print(f"switch model:{model_name} perturbation:{perturbation} language:{language}")
                    prefix_path = "/data1/ljc/code/llm_robustness_eval_and_enhance/bigcode-evaluation-harness/result/"
                    perturbation_whole_name = ""
                    if model_type == "causal_base":
                        perturbation_whole_name = f"mbpp_generate_{language}_robust_{perturbation}"
                    else:
                        perturbation_whole_name = f"mbpp_generate_{language}_robust_{perturbation}_instruct"
                    task_ids = json.load(open(os.path.join("task_ids",perturbation_whole_name +".json")))
                    perturbed_cases = json.load(open(os.path.join(prefix_path,model_name,"evaluation_results",perturbation_whole_name + f"_{prompt_type}","evaluation_cases.json"),"r"))

                    for k,v in perturbed_cases.items():
                        perturbed_cases[k]['task_id'] = task_ids[int(k)]
                    
                    json.dump(perturbed_cases,open(os.path.join(prefix_path,model_name,"evaluation_results",perturbation_whole_name + f"_{prompt_type}", "evaluation_cases.json"),"w"),indent=4)






                

