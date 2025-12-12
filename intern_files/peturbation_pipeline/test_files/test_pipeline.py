from evaluate import load
from datasets import load_from_disk
import os
import json
import re
import argparse
os.environ['HF_ALLOW_CODE_EVAL']= '1'
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/usr/local/lib64'

code_metric = load("../../code_pass_test/debugbench/bigcode-evaluation-harness/bigcode_eval/tasks/custom_metrics/code_eval_octopack")
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",type=str)
    args = parser.parse_args()
    lang = args.lang
    print(lang)
    import_helper = {
        'java':"""import java.util.*;
import java.util.OptionalInt;
import java.util.stream.IntStream;
import java.util.stream.Collectors;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.Arrays;
import java.util.ArrayList;
""",
        'cpp':"""
""",
        "python":"""
""",
        "javascript":"""
"""
    }
    df = pd.read_excel("mbpp_%s_tested.xlsx"%lang)
    codes = df['perturbated_codes']
    tests = df['perturbated_cases']
    passed = [1 for _ in range(500)]
    unpassed_error_info = ['' for _ in range(500)]
    preds = []
    refs = []
    for i in tqdm(range(len(df))):
        preds.append([import_helper[lang] + "\n" + codes[i]])
        refs.append(tests[i])

    #preds = preds[:5]
    #refs = refs[:5]

    metrics, cases = code_metric.compute(
        references=refs,
        predictions=preds,
        language=lang,
        timeout=100.0,
        num_workers=8,
        )
    
    print(metrics,cases)
    with open(f"cased_{lang}.json","w") as f:
        f.write(json.dumps(metrics))
        f.write(json.dumps(cases))
    #df['passed'] = [list(cases.keys())[0][i][1]['passed'] for i in range(len(metrics))]
    # if int(metrics['pass@1'] == 0):
    #     unpassed_error_info[i] = cases[0][0][1]['result']
    #     passed[i] = 0

    # df['unpassed_error_info'] = unpassed_error_info
    # df['is_passed'] = passed
    # df.to_excel("mbpp_%s.xlsx"%lang,index=False)