import json
import re

from evaluate import load
from bigcode_eval.base import Task
from bigcode_eval.utils import extract_code

LANGUAGES = ["python", "cpp", "js", "java"]

LANGUAGE_TO_EXTENSION = {
    "python": "python",
    "cpp": "cpp",
    "java": "java",
    "js": "javascript",
}

# Taken from https://huggingface.co/datasets/nuprl/MultiPL-E/ & https://github.com/THUDM/CodeGeeX
LANGUAGE_TO_STOP_WORDS = {
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L164
    "python": ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\nassert", '\n"""'],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L185
    "cpp": [],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L188
    "js": [],
    # https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L169
    "java": [],
}

LANGUAGE_TO_TIMEOUT = {
    "python": 10,
    "cpp": 60,
    "js": 10,
    "java": 10,
}

# Java sometimes fails with more workers; For JS it's twice as fast with 4 workers
LANGUAGE_TO_NUM_WORKERS = {
    "python": 4,
    "cpp": 4,
    "js": 4,
    "java": 1,
}

# https://github.com/THUDM/CodeGeeX/blob/23ee51505a2bcd34d59d2e271b22e5bd91475462/codegeex/benchmark/utils.py#L6
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

        "from collections import *",
        "from functools import *",
        "from itertools import *",
        "from math import *",
        "from typing import *",
        "inf = float('inf')",
    ],
    "go": [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
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
        "#include<boost/any.hpp>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
        "#include<sstream>",
        "#include<fstream>",
    ],
    "java": [
        "import java.util.*;",
        "import java.lang.*;"
    ],
}


def create_all_tasks():
    return {f"private_code_prompt_code-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class PrivatePromptCode(GeneralPrivatePromptCode):
        def __init__(self, model_series, model_type, stop_words):
            super().__init__(language, model_series, model_type, stop_words)

    return PrivatePromptCode


class GeneralPrivatePromptCode(Task):
    """Parent class for all PrivatePromptCode tasks"""
    def __init__(self, language, model_series, model_type, stop_words):
        self.language = language
        self.DATASET_PATH = f"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/shenyibin/code_eval/leetcode/prompt_code_{language}.json"
        self.DATASET_NAME = language
        self.model_series = model_series
        self.model_type = model_type
        self.use_instruction = False
        stop_tokens = LANGUAGE_TO_STOP_WORDS[language]
        self.use_instruction = False
        self.instruction = f"阅读以下函数签名和文档字符串，并完整实现所描述的函数。你的回答应该只包含这个函数的{LANGUAGE_TO_EXTENSION[language]}代码。\n"
        if self.model_type == "causal_chat":
            self.use_instruction = True
            if self.language == "python":
                stop_tokens = ["if __name__", "\nprint", "\nassert"]
        super().__init__(stop_words=stop_tokens, requires_execution=True)
        self.stop_words += stop_words

    def get_dataset(self):
        self.dataset = json.load(open(self.DATASET_PATH))
        return self.dataset

    def get_prompt(self, doc):
        if self.use_instruction:
            prompt = self.instruction + doc["prompt_chat"]
        else:
            prompt = doc["prompt_base"]
        return prompt

    def get_reference(self, doc):
        return "\n" + doc["reference"]

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
            (not used for Humaneval-Task)
        """
        prompt = self.get_prompt(self.dataset[idx])
        context = self.dataset[idx]["context"]
        generation = generation[len(prompt) :]

        if (not self.use_instruction) or generation.startswith("\n        "):
            generation = self._stop_at_stop_token(generation, self.stop_words)
            generation = context + generation
        else:
            generation = extract_code(generation, self.language)
            generation = self._stop_at_stop_token(generation, self.stop_words)
            if self.language == "python":
                if generation.startswith("\n        ") or generation.startswith("        "):
                    if not generation.startswith("\n"):
                        generation = "\n" + generation
                    generation = context + generation
                else:
                    sep_index = generation.find("class Solution")
                    if sep_index != -1:
                        generation = generation[sep_index + len("class Solution") :]
                    sep_index = generation.find(":\n        ") + 1
                    if generation[sep_index :].startswith("\n            "):
                        generation = generation.lstrip()
                    else:
                        generation = generation[sep_index :]
                        if not generation.startswith("\n"):
                            generation = "\n" + generation
                        generation = context + generation

        return generation

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references.

        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        # code_metric = load("Muennighoff/code_eval_octopack")
        code_metric = load("./bigcode_eval/tasks/custom_metrics/code_eval_octopack")
        timeout = LANGUAGE_TO_TIMEOUT[self.DATASET_NAME]
        num_workers = LANGUAGE_TO_NUM_WORKERS[self.DATASET_NAME]
        language = self.DATASET_NAME if self.DATASET_NAME != "js" else "javascript"

        ### CUSTOM PROG LANGUAGE CHANGES ###
        # Inspiration: https://github.com/THUDM/CodeGeeX/blob/ebeb850f227a90c79de39f7e26b1302f374f3240/codegeex/benchmark/evaluate_humaneval_x.py
        if language == "python":
            python_imports = "\n".join(IMPORT_HELPER["python"])
            generations = [
                [(python_imports + "\n" + g).strip() for g in gen] for gen in generations
            ]
        elif language == "cpp":
            cpp_imports = "\n".join(IMPORT_HELPER["cpp"])
            # Remove main in case present
            generations = [
                [(cpp_imports + "\n" + g.split("int main")[0]).strip() for g in gen] for gen in generations
            ]
        elif language == "java":
            generations = [
                [g.replace("public class Main {\n    }", "").strip() for g in gen] for gen in generations
            ]

        ### EVALUATION ###
        metrics, cases = code_metric.compute(
            references=references,
            predictions=generations,
            language=language,
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
        stats = [[stat, f"private_code_prompt_code_zero_shot:{self.DATASET_NAME}-generation-generation,"]]

        return metrics, cases, stats