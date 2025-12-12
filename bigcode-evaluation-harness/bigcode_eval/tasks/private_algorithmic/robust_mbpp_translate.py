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
        "#include<boost/any.hpp>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
        "#include<sstream>",
        "#include<fstream>",
        "#include<unordered_set>",
        "#include<cassert>",
        "using namespace std;",
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

# Java sometimes fails with more workers; For JS it's twice as fast with 4 workers
LANGUAGE_TO_NUM_WORKERS = {
    "python": 4,
    "cpp": 4,
    "javascript": 4,
    "java": 1,
    "go": 4,
    "rust": 1,
}


LANGUAGES = ["python", "cpp", "javascript", "java"]
TYPES = ["normal", "robust"]
PERTUBATION = ["codebert","normalize","insert_deadcode"]

def create_all_tasks():
    ret = {}
    for type in TYPES:
        for language in LANGUAGES:
            for pertubation in PERTUBATION:
                ret[f"mbpp_translate_{language}_{type}_{pertubation}"] = create_task(type, language,pertubation)
    return ret


def create_task(type,language,pertubation):
    class MBPPtrans(GeneralMBPPtrans):
        def __init__(self):
            super().__init__(type,language,pertubation)

    return MBPPtrans


class GeneralMBPPtrans(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # DATASET_PATH = "openai_humaneval"

    def __init__(self, type, language,pertubation,k=[1, 10, 100], model_series="", model_type="causal_base"):
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
        self.DATASET_PATH = f"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujincheng06/dataset/translate/mbpp_python_{pertubation}_robust.json"
        self.instruction = ""

    def get_dataset(self):
        self.dataset = json.load(open(self.DATASET_PATH, "r"))
        # dataset = []
        # for item in data:
        #     dataset.append(item["code_str_rename_del_oneRow"])
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from."""
        
        original_code = doc["code_str"]
        func_name = doc[f'{self.language}_func_name']
        if self.type == "robust":
            original_code = doc["perturbated_codes"]
        original_prompt = doc["prompt"]
        # declaration = doc["declaration"]
        # function_head = declaration.strip().split('\n')[-1]
        # prompt = f"请你将已有的python程序\n```python\n{original_code}\n```\n改写成{self.language}的代码，```{self.language}\n{function_head}"
        
        prompt = f"""
请你将已有的python程序转译为{self.language}的代码。
-----------------------------
{original_code}
-----------------------------
这是这段代码的原始信息:{original_prompt}。
注意：忽略原始信息中的编程语言要求,例如write a python function中，可以忽略python，你只关注你现在正在完成的编程语言。
要求："""
        if self.language == "java":
            prompt = prompt + f"""
1. 改写后是java代码，用class Solution开头，不要添加public;
2. Solution中方法声明为静态方法，即static public;
3. 函数名:{func_name}，请严格使用此函数名;
4、只需要给出转译后的代码，不需要做任何文字解释，也不需要生成测试脚本；
5、请按照下面的指定格式给出代码；
"""
        elif self.language == "cpp":
            prompt = prompt + f"""
1. 改写后是cpp代码，不要生成int main函数，我有自己的main函数可用;
2. 直接生成函数,不要生成class;
3. 函数名:{func_name}，请严格使用此函数名;
4、只需要给出转译后的代码，不需要做任何文字解释，也不需要生成测试脚本；
5、请按照下面的指定格式给出代码；
"""     
        elif self.language == "javascript":
            prompt = prompt + f"""
1. 函数名:{func_name}，请严格使用此函数名;
2、只需要给出转译后的代码，不需要做任何文字解释，也不需要生成测试脚本；
3. 如果源代码中存在print语句，可以忽略不进行翻译;
3、请按照下面的指定格式给出代码。
"""     
        else:
            prompt = prompt + f"""
1. 函数名:{func_name}，请严格使用此函数名;
2、只需要给出转译后的代码，不需要做任何文字解释，也不需要生成测试脚本；
3、请按照下面的指定格式给出代码；
"""     
        prompt = prompt + f"""
格式：
```{self.language}
转译后对应语言的代码
```
"""
        return prompt   

    def get_reference(self, doc):
        test_func = doc[f"{self.language}_test"]
        return test_func

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """

        prompt = self.get_prompt(self.dataset[idx])
        generation = generation[len(prompt):]
        generation = extract_code(generation, self.language)
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
        generations = [
           [(import_helper + "\n" + g).strip() for g in gen] for gen in generations
           ]

        metrics, cases = code_metric.compute(
            references=references,
            predictions=generations,
            language= self.language,
            timeout=timeout,
            num_workers=num_workers,
        )

        stat = {}
        stat["name"] = {"name": "quasi_prefix_exact_match", "split": "test"}
        stat["count"] = len(references)
        sum_ = metrics["pass@1"] * stat["count"]
        stat["sum"] = sum_
        stat["sum_squared"] = sum_ * sum_
        stat["min"] = metrics["pass@1"]
        stat["max"] = metrics["pass@1"]
        stat["mean"] = metrics["pass@1"]
        stat["variance"] = 0.0
        stat["stddev"] = 0.0
        subdata = ""
        stats = [[stat, f"robust_mbpp_perturbation:mbpp_translate_{self.language}_{self.type}_{self.perturbation}-generation-generation,"]]

        return metrics, cases, stats
