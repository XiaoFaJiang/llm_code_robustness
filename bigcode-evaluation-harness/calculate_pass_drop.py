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
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--model_name", default="", type=str,
                        help="")
    parser.add_argument("--perturbation", default="", type=str,
                        help="")
    parser.add_argument("--model_type", default="", type=str,
                        help="")
    parser.add_argument("--language",default="", type=str)
    parser.add_argument("--prompt_type",default="", type=str)
    parser.add_argument("--dataset",type=str,default="mbpp")
    args = parser.parse_args()
    print(f"calculate model:{args.model_name} perturbation:{args.perturbation} language:{args.language} dataset:{args.dataset}")
    prefix_path = "./result/"
    perturbation_whole_name = ""
    if args.model_type == "causal_base":
        perturbation_whole_name = f"{args.dataset}_generate_{args.language}_robust_{args.perturbation}{args.prompt_type}"
        original_cases = json.load(open(os.path.join(prefix_path,args.model_name,"evaluation_results",f"{args.dataset}_generate_{args.language}_robust_no_change{args.prompt_type}","evaluation_cases.json"),"r"))
        
    else:
        perturbation_whole_name = f"{args.dataset}_generate_{args.language}_robust_{args.perturbation}_instruct{args.prompt_type}"
        original_cases = json.load(open(os.path.join(prefix_path,args.model_name,"evaluation_results",f"{args.dataset}_generate_{args.language}_robust_no_change_instruct{args.prompt_type}","evaluation_cases.json"),"r"))

    print(perturbation_whole_name)
    perturbed_results = json.load(open(os.path.join(prefix_path,args.model_name,"evaluation_results",perturbation_whole_name,"evaluation_results.json"),"r"))
    perturbed_cases = json.load(open(os.path.join(prefix_path,args.model_name,"evaluation_results",perturbation_whole_name,"evaluation_cases.json"),"r"))

    

    original_pass_at_1 = inf
    perturbed_pass_at_1 = perturbed_results[perturbation_whole_name]['pass@1']
    match_cases = 0
    match_passes = 0
    neg2pos = 0
    pos2neg = 0

    for k,v in perturbed_cases.items():
        index = str(v['task_id'])
        match_cases += 1
        if original_cases[index]['passed'] == True:
            assert original_cases[index]['task_id'] == int(index)
            match_passes += 1
            if v['passed'] == False:
                pos2neg += 1
        
        if v['passed'] == True:
            if original_cases[index]['passed'] == False:
                neg2pos += 1

    original_pass_at_1 = match_passes/match_cases
    pass_drop_at_1 = round((original_pass_at_1-perturbed_pass_at_1)/(original_pass_at_1+0.00001),3)

        
    results = {perturbation_whole_name:{"origin_pass@1":original_pass_at_1,"perturbated_pass@1":perturbed_pass_at_1,"pass-drop@1":pass_drop_at_1,"neg2pos":neg2pos,"pos2neg":pos2neg,"count":match_cases}}
    json.dump(results,open(os.path.join(prefix_path,args.model_name,"evaluation_results",perturbation_whole_name,"evaluation_results.json"),"w"),indent=4)



