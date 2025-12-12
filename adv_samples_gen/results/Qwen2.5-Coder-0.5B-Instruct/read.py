import os
import pandas as pd
import json
import random

def read_adv_samples():
    perturbations = ["code_expression_exchange","code_stmt_exchange","code_style","insert","rename"]
    langs = ["cpp","python","java","javascript"]
    res = []
    for lang in langs:
        for p in perturbations:
            print(lang,f"{p}.csv")
            df = pd.read_csv(os.path.join(lang,f"{p}.csv"))
            df = df.dropna()
            for index,row in df.iterrows():
                if row["Adversarial Code"] and row["Is Success"] == 1:
                    res.append({'task_id':index,'Adversarial Code':row['Adversarial Code'],'Adversarial truth':row['Adversarial truth'],'lang':lang})
    
    random.shuffle(res)
    n = len(res)
    train = res[:int(0.8*n)]
    with open("train.jsonl","w") as f:
        for oneline in train:
            f.write(json.dumps(oneline) + "\n")
    valid = res[int(0.8*n):]
    with open("valid.jsonl","w") as f:
        for oneline in valid:
            f.write(json.dumps(oneline) + "\n")
    

if __name__ == '__main__':
    read_adv_samples()