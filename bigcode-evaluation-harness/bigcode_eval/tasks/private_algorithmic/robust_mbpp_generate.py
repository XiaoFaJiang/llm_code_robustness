"""Evaluating Large Language Models Trained on Code
https://arxiv.org/abs/2107.03374

The HumanEval dataset released by OpenAI includes 164 programming problems with a function signature,
docstring, body, and several unit tests. 
They were handwritten to ensure not to be included in the training set of code generation models.

Homepage: https://github.com/openai/human-eval
"""
import json
import re
import os
from collections import Counter, defaultdict
from datasets import load_from_disk
from evaluate import load

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
from bigcode_eval.utils import remove_after_return

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
from bigcode_eval.utils import extract_code
import random
import sys
import copy
sys.path.append("/data1/ljc/code/llm_robustness_eval_and_enhance/perturbation_pipeline")
from pipeline import PerturbationPipeline

_CITATION = """
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code},
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
"""
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/usr/local/lib64'

IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "cpp": [
        "using namespace std;",
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<cmath>",
        "#include<math.h>",
        "#include<numeric>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<set>",
        "#include<map>",
        "#include<queue>",
        "#include<stack>",
        "#include<list>",
        "#include<deque>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
        "#include<sstream>",
        "#include<fstream>",
    ],
    "java": [
        "import java.util.*;",
        "import java.util.OptionalInt;",
        "import java.util.stream.IntStream;",
        "import java.util.stream.Collectors;",
        "import java.util.regex.Matcher;",
        "import java.util.regex.Pattern;",
        "import java.util.Arrays;",
        "import java.util.ArrayList;"
    ],
    "javascript":[]
}

LANGUAGE_TO_TIMEOUT = {
    "python": 10,
    "cpp": 60,
    "javascript": 10,
    "java": 10,
    "go": 20,
    "rust": 300, # Necessary for first-time compilation of cargo
}

# Java sometimes fails with more workers; For javascript it's twice as fast with 4 workers
LANGUAGE_TO_NUM_WORKERS = {
    "python": 4,
    "cpp": 4,
    "javascript": 4,
    "java": 1,
    "go": 4,
    "rust": 1,
}


LANGUAGES = ["python", "cpp", "javascript", "java"]
PERTUBATION = ["no_change","rename","code_stmt_exchange","code_expression_exchange","insert","code_style"]

ALL_TASKS = "mbpp_generate_python_robust_no_change,mbpp_generate_python_robust_rename,mbpp_generate_python_robust_code_stmt_exchange,\
mbpp_generate_python_robust_code_expression_exchange,mbpp_generate_python_robust_insert,\
mbpp_generate_python_robust_code_style,\
mbpp_generate_cpp_robust_no_change,mbpp_generate_cpp_robust_rename,mbpp_generate_cpp_robust_code_stmt_exchange,\
mbpp_generate_cpp_robust_code_expression_exchange,mbpp_generate_cpp_robust_insert,mbpp_generate_cpp_robust_code_style,\
mbpp_generate_javascript_robust_no_change,mbpp_generate_javascript_robust_rename,mbpp_generate_javascript_robust_code_stmt_exchange,\
mbpp_generate_javascript_robust_code_expression_exchange,mbpp_generate_javascript_robust_insert,\
mbpp_generate_javascript_robust_code_style,\
mbpp_generate_java_robust_no_change,mbpp_generate_java_robust_rename,\
mbpp_generate_java_robust_code_stmt_exchange,mbpp_generate_java_robust_code_expression_exchange,\
mbpp_generate_java_robust_insert,mbpp_generate_java_robust_code_style"

def create_all_tasks():
    ret = {}
    for type in ['robust']:
        for language in LANGUAGES:
            for pertubation in PERTUBATION:
                ret[f"mbpp_generate_{language}_{type}_{pertubation}"] = create_task(type, language,pertubation)
    return ret


def create_task(type,language,pertubation):
    class MBPPgenerate(GeneralMbppGenerate):
        def __init__(self):
            super().__init__(type,language,pertubation)

    return MBPPgenerate


class GeneralMbppGenerate(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    def __init__(self, type, language,pertubation,k=[1, 5, 10, 100], model_series="", model_type="causal_base"):
        super().__init__(
            stop_words=[],
            requires_execution=True,
        )
        self.type = type
        self.language = language
        self.perturbation = pertubation
        self.k = k
        self.model_series = model_series
        self.model_type = model_type
        self.code_col_name = "code_str_generate"
        self.DATASET_PATH = f"../../../dataset/mbpp_{language}_tested.json"
        self.p = PerturbationPipeline()
        self.p.set_seed(42)
        self.p.init_pretrained_model()
        self.p.preprocess_code('',self.language)
        self.perturbate = {'rename':self.p.rename_perturbation,'code_stmt_exchange':self.p.code_stmt_perturbtion,\
                           'code_expression_exchange':self.p.code_expression_perturbtion,'insert':self.p.insert_perturbation,\
                            'code_style':self.p.code_style_perturbtion,'no_change':self.p.no_change_perturbation}
        
        self.patterns = {'python':re.compile(r'assert.+',re.DOTALL),\
                'java':re.compile(r'public\s+class\s+Main\s*\{.*\}',re.DOTALL),\
                    'javascript':re.compile(r'const\s+\w+\s*\=\s*\(\s*\)\s*=>\s*.*',re.DOTALL),\
                        'cpp':re.compile(r'int\s+main.*',re.DOTALL)}
        self.instrcution = ""

    def get_dataset(self):
        self.dataset = json.load(open(self.DATASET_PATH, "r"))
        self.count = 0
        self.real_dataset = []
        indexs = []
        perturbated_code = []
        perturbated_test = []
        for i in range(len(self.dataset)):
            perturbations_one_time = self.perturbate[self.perturbation]() #针对于某个样本的扰动复制
            while perturbations_one_time:
                real_pertubertion = random.choice(perturbations_one_time) #从中随机选择一个扰动
                real_pertubertion_copy = real_pertubertion
                perturbation_type = real_pertubertion[1]
                real_pertubertion = real_pertubertion[0]
                code_before = self.dataset[i][self.code_col_name]
                is_perturbated = False
                if 'func' in perturbation_type: #只有重命名函数名会将code和test接在一起进行扰动，因为test中函数名也会跟着变
                    code = self.dataset[i][self.code_col_name] + "\n" + self.dataset[i]['test']
                    code = real_pertubertion(code).strip() #应用扰动
                    test = re.search(self.patterns[self.language],code).group(0) #测试用例
                    code = re.sub(self.patterns[self.language],'',code) #没有测试用例的代码
                else:
                    code = self.dataset[i][self.code_col_name]
                    code = real_pertubertion(code).strip() #应用扰动
                    test = self.dataset[i]['test']
                is_perturbated = not(code.strip() == code_before.strip()) #如果不相等，说明扰动成功
                if is_perturbated: #如果扰动成功，将扰动结果加入，并结束扰动
                    indexs.append(i)
                    perturbated_code.append(code)
                    perturbated_test.append(test)
                    break
                else: #如果扰动失败，删除此扰动方式，继续随机选择一个扰动，直到所有扰动都被选择
                    perturbations_one_time.remove(real_pertubertion_copy)
        
        assert len(perturbated_code) == len(perturbated_test) == len(indexs)
        for i,v in enumerate(perturbated_code):
            data = {}
            data['code_str'] = perturbated_code[i]
            data['test'] = perturbated_test[i]
            data[f'{self.language}_prompt'] = self.dataset[indexs[i]][f'{self.language}_prompt']
            self.real_dataset.append(copy.deepcopy(data))
        
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        self.dataset = copy.deepcopy(self.real_dataset)
        return self.dataset

    def get_prompt(self, doc):
        return doc['code_str']

    def get_reference(self, doc):
        return doc['test']
            
    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """

        prompt = self.get_prompt(self.dataset[idx])
        
        #generation = generation[len(prompt):]
        print("generation:")
        print(generation)
        generation = extract_code(generation, self.language)
        print("extracted code:")
        print(generation)
        return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        code_metric = load("./bigcode_eval/tasks/custom_metrics/code_eval_octopack")
        timeout = LANGUAGE_TO_TIMEOUT[self.language]
        num_workers = LANGUAGE_TO_NUM_WORKERS[self.language]
        import_helper = "\n".join(IMPORT_HELPER[self.language])
        length = len(generations)
        generations = [
           [(import_helper + "\n" + g).strip() for g in gen] for gen in generations
           ]

        metrics, cases = code_metric.compute(
            references=references,
            predictions=generations,
            language= self.language,
            timeout=timeout,
            num_workers=num_workers,
            k = self.k
        )

        stat = {}
        stat["name"] = {"name": "full_level_robustness", "split": "test"}
        stat["count"] = length//2
        stat['generations'] = generations
        stat["variance"] = 0.0
        stat["stddev"] = 0.0

        stats = [[stat, f"robust_mbpp_perturbation:mbpp_generate_{self.language}_{self.type}_{self.perturbation}-generation-generation,"]]

        return metrics, cases, stats

