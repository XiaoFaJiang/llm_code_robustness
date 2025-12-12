#coding=utf-8
import pandas as pd
from tree_sitter import Language, Parser, Tree, Node
from pipeline import PerturbationPipeline
import random

def dfs(node):
    if not node:
        return
    else:
        print(node.text.decode("utf-8"))
        print(node,node.parent)
        print(node.children)
        itertion = node.children if node.children else []
        for child in itertion:
            dfs(child)
lang = 'python'
df = pd.read_excel("test_files/mbpp_%s.xlsx"%lang)
LANGUAGE = Language('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liujincheng06/peturbation_pipeline/tree_sitter/my-languages.so', lang)
parser = Parser()
parser.set_language(LANGUAGE)
a = PerturbationPipeline()
a.init_pretrained_model()
random.choice(a.no_change_perturbation())
for i in range(len(df)):
    index = 3
    a.set_seed(42 + index)
    code = df['code_str'][index]
    test = df['test'][index]
    temp = code + "\n" + test
    tree = parser.parse(bytes(temp,"utf-8"))
    dfs(tree.root_node)
    temp = a.preprocess_code(temp,lang)
    print(temp)
    code = a.insert_dead_code(temp)
    print(code,code == temp,sep="\n")
    break