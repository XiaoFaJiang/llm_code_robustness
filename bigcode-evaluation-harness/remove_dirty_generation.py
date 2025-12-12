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
python evaluate_pass_at.py --language=java\
  --model_name=qwen2.5-coder-3b-instruct\
  --perturbation=no_change\
  --model_type=causal_chat\
'''
ps = ["rename","code_style","insert","code_stmt_exchanging","code_expression_exchanging"]
pt = ["_1random_prompt","_3random_prompt","_5random_prompt","_1sorted_prompt","_3sorted_prompt","_5sorted_prompt"]
if __name__ == '__main__':
    base_dir = "result"
    for model in os.listdir(base_dir):
        if "instruct" in model or "chat" in model:
            '''
            next_dir = os.path.join(base_dir,model,"generations")
            for every in os.listdir(next_dir):
                if "java" in every and "script" not in every:
                    x = json.load(open(os.path.join(next_dir,every,"generations.json")))
                    store = []
                    for y in x:
                        y = y[0]
                        y = re.sub("```","",y)
                        store.append([y])
                    print(os.path.join(next_dir,every,"generations2.json"))
                    json.dump(store,open(os.path.join(next_dir,every,"generations2.json"),"w"))
            '''
            for p in ps:
                if "prompt" not in model:
                    print(f'python calculate_pass_drop.py --language=java --model_name={model} --perturbation={p} --model_type=causal_chat')
                else:
                    for lpt in pt:
                        print(f'python calculate_pass_drop.py --language=java --model_name={model} --perturbation={p} --model_type=causal_chat --prompt_type={lpt}')

                        


    




                

