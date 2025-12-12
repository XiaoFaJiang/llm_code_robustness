from evaluate import load
from datasets import load_from_disk
import os
import json
import re
os.environ['HF_ALLOW_CODE_EVAL']= '1'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/usr/local/lib64'

code_metric = load("./bigcode-evaluation-harness/bigcode_eval/tasks/custom_metrics/code_eval_octopack")
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    lang = 'js'
    df = pd.read_excel("mbpp_js_standard_rename_execute.xlsx")
    codes = df['%s_code_str_rename'%lang]
    tests = df['%s_test_rename'%lang]
    passed = [1 for _ in range(500)]
    unpassed_error_info = ['' for _ in range(500)]
    preds = []
    refs = []
    for i in tqdm(range(len(df))):
        preds.append([codes[i]])
        refs.append(tests[i])

    # preds = preds[:10]
    # refs = refs[:10]
    metrics, cases = code_metric.compute(
        references=refs,
        predictions=preds,
        language='javascript',
        timeout=10.0,
        num_workers=8,
        )
    print(metrics,cases)
    # if int(metrics['pass@1'] == 0):
    #     unpassed_error_info[i] = cases[0][0][1]['result']
    #     passed[i] = 0

    # df['unpassed_error_info'] = unpassed_error_info
    # df['is_passed'] = passed
    # df.to_excel("mbpp_%s.xlsx"%lang,index=False)