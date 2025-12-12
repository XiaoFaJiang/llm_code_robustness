import sys
sys.path.append("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujincheng06/peturbation_pipeline")
import pandas as pd
from pipeline import PerturbationPipeline
import re
import os
import argparse
from tqdm import tqdm



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",type=str)
    parser.add_argument("--file",type=str)
    parser.add_argument("--code_col_name",type=str)
    args = parser.parse_args()
    lang = args.lang
    df = pd.read_excel(args.file)
    codes = df[args.code_col_name]
    perturbated_codes = []
    perturbated_cases = []
    patterns = {'python':re.compile(r'assert.+',re.DOTALL),\
                'java':re.compile(r'public\s+class\s+Main\s*\{.*\}',re.DOTALL),\
                    'javascript':re.compile(r'const\s+\w+\s*\=\s*\(\s*\)\s*=>\s*.*',re.DOTALL),\
                        'cpp':re.compile(r'int\s+main.*',re.DOTALL)}
    changed = []
    p = PerturbationPipeline()
    p.init_pretrained_model()
    for i in tqdm(range(len(df))):
        tmp_code = codes[i].strip()
        tmp_test = df['test'][i].strip()
        p = PerturbationPipeline()
        p.set_seed(42)
        p.set_seed(42 + i)
        real_code = tmp_code + "\n" + tmp_test
        p.preprocess_code(real_code,lang)
        code_after = p.insert_dead_code(real_code).strip()
        changed.append(1 - int(real_code.strip() == code_after.strip()))
        test_after = re.search(patterns[lang],code_after).group(0)
        code_after = re.sub(patterns[lang],'',code_after)
        perturbated_codes.append(code_after) #应用某种变换
        perturbated_cases.append(test_after) #

    print(sum(changed))

    df['perturbated_codes'] = perturbated_codes
    df['perturbated_cases'] = perturbated_cases
    df['changed'] = changed
    df.to_excel("mbpp_%s_tested.xlsx"%lang,index = False)