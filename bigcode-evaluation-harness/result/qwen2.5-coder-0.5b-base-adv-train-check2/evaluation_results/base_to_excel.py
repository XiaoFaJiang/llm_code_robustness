import pandas as pd
import os
import json
df = pd.DataFrame(columns = ['Test Case','neg2pos','pos2neg','count','origin_pass@1','perturbated_pass@1','pass-drop@1'])

for nowdir in os.listdir():
    if (nowdir.startswith('mbpp') or nowdir.startswith('humaneval'))  and not nowdir.endswith("instruct") and not("no_change") in nowdir:
        with open(os.path.join(nowdir,'evaluation_results.json')) as f:
            res = json.loads(f.read())
        newline = {}
        name = str(list(res.keys())[0])
        newline['Test Case'] = name
        newline.update(res[name])
        df.loc[len(df)] = newline

model_name = os.path.dirname(os.getcwd()).split('/')[-1]
df = df.sort_values(by='Test Case')
df.to_excel(f'{model_name}.xlsx',index=False)