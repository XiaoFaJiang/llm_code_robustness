import pandas as pd
import os
import json
local_dir = "./"
model_to_df = {}
for x_dir in os.listdir(local_dir):
    if (x_dir.endswith("chat") or x_dir.endswith("instruct")) and os.path.isdir(x_dir):
        df = pd.DataFrame(columns = ['Test Case','neg2pos','pos2neg','count','origin_pass@1','perturbated_pass@1','pass-drop@1'])
        for y_dir in os.listdir(x_dir):
            if y_dir == "evaluation_results":
                for nowdir in os.listdir(os.path.join(x_dir,y_dir)):
                    if (nowdir.startswith('mbpp') or nowdir.startswith('humaneval')) and nowdir.endswith("instruct") and not ("no_change" in nowdir):
                        with open(os.path.join(local_dir,x_dir,y_dir,nowdir,'evaluation_results.json')) as f:
                            res = json.loads(f.read())
                        #print(res)
                        try:
                            newline = {}
                            name = str(list(res.keys())[0])
                            newline['Test Case'] = name
                            newline.update(res[name])
                            df.loc[len(df)] = newline
                        except:
                            pass
        df = df.sort_values(by='Test Case')
        df.to_excel(f'{x_dir}.xlsx',index = False)
        model_to_df[x_dir] = df