import argparse
import os
import sys
import json
import tqdm

from evaluation.debugger.responser import HttpApiResponser
from leetcode_oj import LeetCodeTester
from debugger import GPT4Responser, TurboResponser, IODebugger

def load_bug_data():
    """ load data with different languages and bug types """
    res = {
        'cpp': {},
        'java': {},
        'python3': {},
    }
    files = os.listdir("benchmark")
    for file in files:
        file_name = os.path.splitext(file)[0]
        lang = file_name[:file_name.find('_')]
        bug_type = file_name[file_name.find('_') + 1:]
        res[lang][bug_type] = json.load(open(os.path.join("benchmark", file)))
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--customize_inference_ip', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default= None)
    parser.add_argument('--leetcode_session', type=str, default=None)
    parser.add_argument('--csrf_token', type=str, default=None)
    args = parser.parse_args()
    model = args.model.replace("/", "_")
    save_dir = args.save_dir if args.save_dir is not None else f"evaluation/res/{model}/debug"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    responser = HttpApiResponser(ip=args.customize_inference_ip)
    debugger = IODebugger(responser)
    tester = LeetCodeTester(leetcode_session=args.leetcode_session, csrf_token=args.csrf_token)

    bug_data = load_bug_data()
    for lang in bug_data.keys():
        for bug_type in bug_data[lang]:

            save_dir = os.path.join(save_dir, f"{lang}_{bug_type}.json")
            if not os.path.exists(save_dir):
                bug_data_split = bug_data[lang][bug_type]
                res = []

                for case in tqdm.tqdm(bug_data_split, desc=f"{lang}_{bug_type}"):
                    fixed_code, fixing_exp = debugger.debug(lang=lang, code=case['buggy_code'])
                    rw, res_dict = tester.test(code=fixed_code, language=lang, task_id=case['slug'])
                    case['fixed_code'] = fixed_code
                    case['fixing_exp'] = fixing_exp
                    case['test_result_bool'] = rw
                    case['test_result_dict'] = res_dict
                    res.append(case)

                with open(save_dir, 'w') as f:
                    json.dump(res, f, indent=4)


if __name__ == '__main__':
    main()
