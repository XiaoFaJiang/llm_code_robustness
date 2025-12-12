import json
import argparse
func_names = []

parser = argparse.ArgumentParser()
parser.add_argument("--lang",type=str)
parser.add_argument("--perturbation",type=str)
args = parser.parse_args()

lang = args.lang
p = args.perturbation
with open(f"{lang}_func_name.jsonl","r") as f:
    for line in f:
        func_names.append(json.loads(line)[0])

objec = None
with open(f'mbpp_python_{p}_robust.json',"r") as f:
    objec = json.loads(f.read())

for i,v in enumerate(func_names):
    objec[i][f'{lang}_func_name'] = func_names[i]

with open(f'mbpp_python_{p}_robust.json',"w") as f:
    json.dump(objec,f,indent=4)