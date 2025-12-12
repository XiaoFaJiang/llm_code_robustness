import json
import re

from collections import Counter, defaultdict
from evaluate import load

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval
from bigcode_eval.utils import remove_after_return

# os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/usr/local/lib64'

IO_BASE_PROMPT = """Observe the following {LANG} faulty code which is complete with no extra context. Your task is to fix up the code. Your reply should be like this:
<code>
your fixed code
</code>
"""

IO_INTENTION_PROMPT = """Observe the function intention and its corresponding {LANG} implementation which is complete with no extra context. The implementation is faulty. Your task is to fix up the code and explain on the modification in less than 20 words.
You have to write the fixed code again. You should put <code></code> and <exp></exp> on the boundary of the code and the explanation. Do not write anything else in your response. Your reply should be like this:
<code>
fixed code
</code>
<exp>
short explanation about the bug
</exp>"""

IO_EX_PROMPT = """"""

IO_TRACEBACK_PROMPT = """Observe the following {LANG} faulty code which is complete with no extra context and its traceback messages. Your task is to fix up the code and explain on the modification in less than 20 words.
You have to write the fixed code again. You should put <code></code> and <exp></exp> on the boundary of the code and the explanation. Do not write anything else in your response. Your reply should be like this:
<code>
fixed code
</code>
<exp>
short explanation about the bug
</exp>"""

IO_COMPLETE_SYSTEM_INFO = """Observe the program question and {LANG} implementation schema. Your task is to complete the implementation. 
You should write the completed code again. You should put <code></code> on the boundary of the code. Do not write anything else in your response. Your reply should be like this:
<code>
completed code
</code>"""

IO_COMPLETE_USER_PROMPT = """- Program Question
{INTENTION}
- Implementation Schema
{SIGNATURE}
"""

EXTRACT_SIGNATURE_PROMPT = """Observe the following code. Your task extract ONLY the class & function SIGNATURE. Wrap the cleaned code with <code>, </code>. Do not write anything else in the response. Do not forget the class. Do not miss any functions.
Here are some examples.

- User Input
class Solution {
public:
    string gcdOfStrings(string str1, string str2) {

        if(str1+str2==str2+str1)
        {
            return str1.substr(0,gcd(str1.length(),str2.length()));
        }
        else{
            return "";
        }
        
    }
};
- Response
<code>class Solution {
public:
    string gcdOfStrings(string str1, string str2);
};</code>

- User Input
class Solution {
    public Node connect(Node node) {
        Map<Integer, List<Node>> map = new HashMap<>();
        goDFS(0, node, map);
        for (int key : map.keySet()) {
            List<Node> list = map.get(key);
            for (int i = 1; i < list.size(); i++) {
                list.get(i - 1).next = list.get(i);
            }
        }
        return node;
    }

    private void goDFS(int lvl, Node node, Map<Integer, List<Node>> map) {
        if (node == null) return;

        List<Node> list = map.computeIfAbsent(lvl, k -> new ArrayList<>());
        list.add(node);
        lvl++;
        goDFS(lvl, node.left, map);
        goDFS(lvl, node.right, map);
    }
}
- Response
<code>class Solution {
    public Node connect(Node node);
    private void goDFS(int lvl, Node node, Map<Integer, List<Node>> map);
}</code>

- User Input
def tsum(root):
    if(root==None):
        return 0
    x= root.val+tsum(root.left)+tsum(root.right)
    return x
def fun(root,sm,mx):
    if(root==None):
        return 0
    a=fun(root.left,sm,mx)
    b=fun(root.right,sm,mx)
    mx[0]=max(mx[0],a*(sm-a),b*(sm-b))
    return a+b+root.val
    
class Solution:
    def maxProduct(self, root: Optional[TreeNode]) -> int:
        mx=[0]
        sm=tsum(root)
        memo={}
        fun(root,sm,mx)
        return mx[0]%(10**9+7)
- Response
<code>def tsum(root):
def fun(root,sm,mx):
class Solution:
    def maxProduct(self, root: Optional[TreeNode]) -> int:</code>
"""

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

LANGUAGE_TO_TIMEOUT = {
    "python": 10,
    "cpp": 60,
    "js": 10,
    "java": 10,
    "go": 20,
    "rust": 300, # Necessary for first-time compilation of cargo
}

# Java sometimes fails with more workers; For JS it's twice as fast with 4 workers
LANGUAGE_TO_NUM_WORKERS = {
    "python": 4,
    "cpp": 4,
    "js": 4,
    "java": 1,
    "go": 4,
    "rust": 1,
}

HEAD = {
    "cpp": """
#undef NDEBUG
#include<assert.h>
bool issame(int a,vector<int>b){
    if (b.size() > 1) return false;
    if (a!=b[0]) return false;
    return true;
};
bool issame(string a,vector<string>b){
    if (b.size() > 1) return false;
    if (a!=b[0]) return false;
    return true;
};
bool issame(vector<int> a,vector<int>b){
    if (a.size()!=b.size()) return false;
    for (int i=0;i<a.size();i++)
    {
    if (a[i]!=b[i]) return false;
    }
    return true;
};
int main(){
""",

    "java": """
public class Main {
    public static void main(String[] args) {
        Solution s = new Solution();
        List<Boolean> correct = Arrays.asList(
"""
}
TAIL = {
    "cpp" : '}',
    "java" : """
            );
            if (correct.contains(false)) {
                throw new AssertionError();
            }
        }
    }
""" 
}

LANGUAGES = ["java","cpp","python"]

def create_all_tasks():
    return {f"debugbench-{language}": create_task(language) for language in LANGUAGES}


def create_task(language):
    class Debugbench(GeneralDebugbench):
        def __init__(self):
            super().__init__(language)

    return Debugbench


class GeneralDebugbench(Task):
    def __init__(self, language, k=[1, 10, 100], model_series="", model_type="causal_base"):
        super().__init__(
            stop_words=[],
            requires_execution=True,
        )
        self.language = language
        self.DATASET_PATH = f"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhaoyicong03/dataset/flitered_{language}.json"
        self.k = k
        self.model_series = model_series
        self.model_type = model_type

    def convert_class_method_to_function_python(self, class_method_code):
        lines = class_method_code.split('\n')
        function_lines = []
        for line in lines:
            if line.strip().startswith('class'):
                continue
            if line.strip().startswith('def'):
                function_lines.append(re.sub(r'\(self, ', '(', line[4:]))
            else:
                function_lines.append(line[4:])
        return '\n'.join(function_lines)

    def convert_class_method_to_function_cpp(self,code):
        lines = code.split('\n')[3:-2]  # 去除第一行、第二行以及最后一行
        for i in range(len(lines)):
            if re.match(r'\s*\w+\s+\w+\s*\(', lines[i]):  # match function declaration
                lines[i] = lines[i].replace('&', '')
        return '\n'.join(lines)

    def get_dataset(self):
        dataset = json.load(open(self.DATASET_PATH, "r"))
        #random.seed(0)
        #random.shuffle(dataset)
        return dataset
    
    def get_prompt(self, doc):
        code = doc["buggy_code"]
        question = doc["description"]
        lines = code.strip().split('\n')
        name = lines[1]
        name = name.lstrip()

        if self.language == 'cpp':
            code = self.convert_class_method_to_function_cpp(code)
        elif self.language == 'java':
            code = code
        elif self.language == 'python':
            code = self.convert_class_method_to_function_python(code)

        # if self.intention:
        #     system_prompt = IO_INTENTION_PROMPT.replace("{LANG}", self.language)
        # elif self.case:
        #     system_prompt = IO_EX_PROMPT.replace("{LANG}", self.language)
        # elif self.traceback:
        #     system_prompt = IO_TRACEBACK_PROMPT.replace("{LANG}", self.language)
        # else:
        system_prompt = IO_BASE_PROMPT.replace("{LANG}", self.language)

        # if self.traceback:
        #     user_prompt = code + f'\nTraceback\n{self.traceback}'
        # elif self.intention:
        #     user_prompt = f'\nIntention\n{self.intention}' + f'\nImplementation\n{code}'
        # else:
        user_prompt = code
        
        system_prompt = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer." + system_prompt +"### Instruction:\n"
        user_prompt = "question:\n" +  question + "\n" +"Here is faulty implementation." + user_prompt + "\n" +"please only response with the fixed code between <code> and </code>" + "###Response: <code>"
        return system_prompt + user_prompt
        
    def get_reference(self, doc):
        parts = doc["examples"]
        gt = doc["oracle_code"]
        if self.language == 'python':
            pattern = r'def (?!__init__\()(\w+)\('
            matches = re.search(pattern, gt)
            name = matches.group(1)
            output_pattern = re.compile(r"Output: ([^\n]+)")
            fun = ""
            for item in parts:
                outputs = output_pattern.findall(item)
                if len(outputs) == 1:
                    output = outputs[0]
                else:
                    output = "(" + ", ".join(outputs) + ")"
                s = item.split("Output:")[0]
                parts = s.split('=')
                l = [part.rsplit(',', 1)[0].strip() for part in parts[1:-1]]
                l.append(parts[-1].strip().rstrip('\n'))
                l = ', '.join(str(result) for result in l)
                fun += f"assert {name}({l}) == {output}\n"
            fun = fun.replace("false", "False")
            fun = fun.replace("true", "True")
            fun = fun.replace("null", "None")
            return fun
        elif self.language == 'cpp':
            name = gt.split('(')[0].strip().split(' ')[-1]
            output_pattern = re.compile(r"Output: ([^\n]+)")
            fun = ""
            for item in parts:    
                outputs = output_pattern.findall(item)
                if len(outputs) == 1:
                    output = outputs[0]
                else:
                    output = "(" + ", ".join(outputs) + ")"
                output = "{" + output + "}"
                s = item.split("Output:")[0]
                parts = s.split('=')
                l = [part.rsplit(',', 1)[0].strip() for part in parts[1:-1]]
                l.append(parts[-1].strip().rstrip('\n'))
                l = ', '.join(str(result) for result in l)
                ans = l.replace("[","{").replace("]","}")
                fun += f"assert(issame({name}({ans}),{output}));\n"
            fun = fun.replace("false", "False")
            fun = fun.replace("true", "True")
            fun = fun.replace("null", "None")
            fun = HEAD[self.language] + fun + TAIL[self.language]
            return fun
        elif self.language == 'java':
            name = gt.split('(')[0].strip().split(' ')[-1]
            output_pattern = re.compile(r"Output: ([^\n]+)")
            fun = ""
            for item in parts:
                outputs = output_pattern.findall(item)
                if len(outputs) == 1:
                    output = outputs[0]
                else:
                    output = "(" + ", ".join(outputs) + ")"
                output = output.replace("[","{").replace("]","}")
                s = item.split("Output:")[0]
                parts = s.split('=')
                l = [part.rsplit(',', 1)[0].strip() for part in parts[1:-1]]
                l.append(parts[-1].strip().rstrip('\n'))
                l = ', '.join(str(result) for result in l)
                ans = l.replace("[","(").replace("]",")")
                fun += f"s.{name}({ans}) == {output},\n"
            fun = fun.replace("false", "False")
            fun = fun.replace("true", "True")
            fun = fun.replace("null", "None")
            fun = fun.rstrip(",\n")
            fun = HEAD[self.language] + fun + TAIL[self.language]
            return fun

    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        pattern = re.compile(r"```{}(.*?)```".format(self.language), re.DOTALL)
        match = pattern.search(generation)
        if match:
            return match.group(1).strip()
        else:  
            pattern = re.compile(r'<code>(.*?)</code>', re.DOTALL)
            matches = pattern.findall(generation)
            if len(matches) >= 3:
                # 对java的项做额外后处理头文件
                generation = matches[2]
                return generation
            else:
                # 如果没有至少两对标签，返回空字符串或None
                return ""

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
        stats = [[stat, "debugbench_zero_shot-generation-generation:"]]

        return metrics, cases, stats