import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lang",type=str)
args = parser.parse_args()
df = pd.read_excel(f"mbpp_{args.lang}_tested.xlsx")
#pandas转dict

#excel转json文件

with open(f"mbpp_{args.lang}_completion_tested.json", "w") as f:
    
    f.write(df.to_json(orient="records",indent = 4) )