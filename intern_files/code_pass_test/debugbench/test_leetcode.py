import json
import os
import sys
from evaluate import load


## 头文件辅助
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

        "import sortedcontainers",
        "from collections import *",
        "from functools import *",
        "from itertools import *",
        "from typing import *",
        "from bisect import *",
        "from heapq import *",
        "from string import *",
        "inf = float('inf')",
    ],
    "cpp": [
        "using namespace std;",
        "#include<bits/stdc++.h>",
    ],
    "java": [
        "import java.util.*;",
        "import java.lang.*;",
        "import java.math.BigInteger;"
    ],
    "js" :[
    ]
}

## 函数辅助
DEF_HELPER = {
    "python": [
        """def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)""",

        """def bit_count(x: int) -> int:
    return bin(x).count("1")""",

        """def bisect_left(a, x, lo=0, hi=None, *, key=None):
    if lo < 0:
        raise ValueError("lo must be non-negative")
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if key(a[mid]) < x:
                lo = mid + 1
            else:
                hi = mid
    return lo""",

        """def comb(n, k):
    return math.comb(n, k)""",
        """def isqrt(n):
    return math.isqrt(n)""",
        """def lcm(a, b):
    return math.lcm(a, b)""",
    ],
    "cpp": [
    ],
    "java" : [
    """
class Pair<K, V> implements Comparable<Pair<K, V>> {
    private K key;
    private V value;

    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }

    public K getKey() {
        return key;
    }

    public V getValue() {
        return value;
    }

    @Override
    public int compareTo(Pair<K, V> other) {
        if (this.key instanceof Comparable && other.key instanceof Comparable) {
            return ((Comparable) this.key).compareTo(other.key);
        }
        throw new UnsupportedOperationException("Keys are not comparable.");
    }

    @Override
    public String toString() {
        return "(" + key + ", " + value + ")";
    }
}
"""
    ],
    "js": [
    """class PriorityQueue {
  constructor({ compare }) {
    this.items = [];
    this.compare = compare;
  }

  enqueue(element) {
    this.items.push(element);
    this.items.sort(this.compare);
  }

  dequeue() {
    return this.items.shift();
  }

  front() {
    return this.items[0];
  }

  isEmpty() {
    return this.items.length === 0;
  }

  print() {
    console.log(this.items);
  }
}

class MaxPriorityQueue {
  constructor() {
    this.items = [];
  }

  enqueue(element) {
    if (this.isEmpty()) {
      this.items.push({ element });
    } else {
      let added = false;
      for (let i = 0; i < this.items.length; i++) {
        if (element.count > this.items[i].element.count) {
          this.items.splice(i, 0, { element });
          added = true;
          break;
        }
      }
      if (!added) {
        this.items.push({ element });
      }
    }
  }

  dequeue() {
    return this.items.shift();
  }

  front() {
    return { element: this.items[0].element };
  }

  isEmpty() {
    return this.items.length === 0;
  }

  print() {
    console.log(this.items);
  }
}

class MinPriorityQueue {
  constructor(options = {}) {
    this.compare = options.compare || ((a, b) => a - b);
    this.items = [];
  }

  enqueue(element) {
    this.items.push(element);
    this.items.sort(this.compare);
  }

  dequeue() {
    return this.items.shift();
  }

  front() {
    return { element: this.items[0] };
  }

  isEmpty() {
    return this.items.length === 0;
  }
}

"""
    ]
}

lang = "java"
time = "2406"
os.environ["HF_ALLOW_CODE_EVAL"]= "1"
os.environ["LD_LIBRARY_PATH"] = "$LD_LIBRARY_PATH:/usr/local/lib64"
df = json.load(open("/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/liujincheng06/code_pass_test/debugbench/prompt_code_2406_zh_%s copy.json"%lang))

## 模型预测部分
preds = []
for i,v in enumerate(df):
    #if v["id"] == "3003":
      if v["solution"]:
          preds.append(["\n".join(IMPORT_HELPER[lang]) + "\n" + "\n".join(DEF_HELPER[lang]) + "\n" + v["class_helper"] + "\n" + v["solution"]])
## 测试用例
refs =[]
for i,v in enumerate(df):
    #if v["id"] == "3003":
      if v["solution"]:
          refs.append(v["reference"])

assert len(preds) == len(refs)
#preds = preds[0:10]
#refs = refs[0:10]
## 测试脚本
code_metric = load("./bigcode-evaluation-harness/bigcode_eval/tasks/custom_metrics/code_eval_octopack")

metrics, cases = code_metric.compute(
    references=refs,
    predictions=preds,
    language="java",
    timeout=60,
    num_workers=8,
)
print(metrics)
print(cases)