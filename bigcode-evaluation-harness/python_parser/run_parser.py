# coding=utf-8
import argparse
from os import replace
import sys
sys.path.append("../perturbation_pipeline")
sys.path.append("../../perturbation_pipeline")
from pipeline import PerturbationPipeline
import os
from queue import Queue
import copy
import pandas as pd
import re
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoConfig
import openai
import re

# 获取当前文件的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录的路径
parent_dir = os.path.dirname(current_dir)

# 将上一级目录添加到 sys.path 中
sys.path.insert(0, parent_dir)
sys.path.insert(0,current_dir)

from parser_folder.DFG_python import DFG_python
from parser_folder.DFG_c import DFG_c
from parser_folder.DFG_java import DFG_java
from parser_folder.DFG_javascript import DFG_js
from parser_folder import (remove_comments_and_docstrings,
                           tree_to_token_index,
                           index_to_code_token,)
from tree_sitter import Language, Parser
import os
import random
from typing import Generator
from tree_sitter import Language, Parser, Tree, Node
from utils_parser import java_keywords , java_special_ids, c_keywords, c_macros, c_special_ids, js_all_names,stmt_separators,all_keywords,\
        literal_comparsion,condition_stmt_name_convert_dict,python_keywords


from keyword import iskeyword


path = 'parser_folder/my-languages.so'
path = os.path.join(os.path.dirname(__file__),path)

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'cpp': DFG_c,
    'javascript':DFG_js,
}


LANG_LIB_MAP = {
    'python': 'tree_sitter_assets/python.so',
    'cpp': 'tree_sitter_assets/cpp.so',
    'java': 'tree_sitter_assets/java.so',
    'javascript': 'tree_sitter_assets/javascript.so',
    
}

LANG_REPO_MAP = {
    'python': 'tree-sitter-python',
    'cpp': 'tree-sitter-cpp',
    'java': 'tree-sitter-java',
    'javascript': 'tree-sitter-javascript',
    
}



if not os.path.exists(path):
    for lang in LANG_LIB_MAP:
        print(f'Installing {lang} language library...')
        if not os.path.exists(LANG_REPO_MAP[lang]):
            os.popen(
                f'git clone https://github.com/tree-sitter/{LANG_REPO_MAP[lang]}.git'
            ).read()
    Language.build_library(path, list(LANG_REPO_MAP.values()))

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language(path, lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


def is_valid_variable_name(name: str, lang: str) -> bool:
    # check if matches language keywords and special ids
    def is_valid_variable_python(name: str) -> bool:
        return name.isidentifier() and not iskeyword(name) and (name not  in python_keywords)

    def is_valid_variable_java(name: str) -> bool:
        if not name.isidentifier():
            return False
        elif name in java_keywords:
            return False
        elif name in java_special_ids:
            return False
        return True

    def is_valid_variable_c(name: str) -> bool:

        if not name.isidentifier():
            return False
        elif name in c_keywords:
            return False
        elif name in c_macros:
            return False
        elif name in c_special_ids:
            return False
        return True

    def is_valid_variable_cpp(name: str) -> bool:
        return is_valid_variable_c(name)

    def is_valid_variable_js(name: str) -> bool:
        if not name.isidentifier():
            return False
        elif name in js_all_names:
            return False
        return True

    functions = {"cpp":is_valid_variable_cpp,"java":is_valid_variable_java,"python":is_valid_variable_python,"javascript":is_valid_variable_js}
    return functions[lang](name)

def get_code_tokens(code, lang):
    code = code.split('\n')
    code_tokens = [x + '\\n' for x in code if x ]
    return code_tokens

def extract_dataflow(code, lang):
    parser = parsers[lang]
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf-8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    index_to_code = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code)

    index_table = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_table[idx] = index

    DFG, _ = parser[1](root_node, index_to_code, {})

    DFG = sorted(DFG, key=lambda x: x[1])
    return DFG, index_table, code_tokens

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def get_identifiers(code, lang):
    dfg, index_table, code_tokens = extract_dataflow(code, lang)
    ret = []
    for d in dfg:
        if is_valid_variable_name(d[0], lang):
            ret.append(d[0])
    ret = unique(ret)
    ret = [ [i] for i in ret]
    
    return ret, code_tokens
        
def get_program_stmt_nums(code,lang):
    #得到源代码中statement的位置和数量
    ret = code.split(stmt_separators[lang])
    return ret

def replace_node_of_code(code:str,node:Node,replaced:str,diff:int,lang:str):
        code = code[:node.start_byte + diff] + replaced + code[node.end_byte + diff:]
        diff += -(node.end_byte - node.start_byte) + len(replaced)
        return code[:],diff

def insert_code_of_a_node(code,node,inserted,diff,lang):
        # 在node之前insert一段代码
        real_inserted = inserted + stmt_separators[lang] 
        acc = len(real_inserted)
        code = code[:node.start_byte + diff] +  real_inserted + code[node.start_byte + diff:]
        diff += acc
        return code[:],diff   
    
def build_cfg(code,lang):
    #构建一段代码的控制流图，以便于后续操作
    parser = parsers[lang]
    tree = parser[0].parse(bytes(code, 'utf-8'))
    #1. 构建基本块；2. 根据跳转关系连接基本块；
    #把所有的语言都统一转换为一种中间（三地址码）表示，然后再进行构建
    #一个if或while语句的node就是一个基本块，它的所有node就是要跳转到的语句
    #对于死代码检测，很简单，我们只需要考察if或while node的第二个child的值是不是恒定为false，如果是，那就是死代码
    #对于常量False，很好判断；但是对于变量或者是变量之间的比较，就需要进行常量传播后分析了
    #那么对于我们这里判断死代码，只判断常量False。我们理想的情况是在判断死代码前先进行常量分析，将所有的变量转换为常量，然后再进行判断


        



class mytree_node:
    def __init__(self,node,is_source=True):
        self.node = node
        self.is_source = is_source
        self.children = []
        self.parent = None
    
    def set_parent(self,attack_code):
        flag = False if self.node.parent and self.node.parent.text.decode("utf-8") in attack_code else True
        self.parent = mytree_node(self.node.parent if self.node.parent else None,flag)
    
    def set_children(self,attack_code):
        for child in self.node.children:
            flag = False if child and child.text.decode("utf-8") in attack_code else True
            self.children.append(mytree_node(child,flag))
    
class mytree:
    def __init__(self,code,lang,attack_code = []):
        self.__parser = parsers[lang]
        self.tree = self.__parser[0].parse(bytes(code,"utf-8"))
        self.my_root_node = mytree_node(self.tree.root_node,True)
        def dfg(node:mytree_node):
            node.set_parent(attack_code)
            node.set_children(attack_code)
            for child in node.children:
                dfg(child)
        dfg(self.my_root_node)

           
        
        
def traverse_tree(tree: Tree) -> Generator[Node, None, None]:
    cursor = tree.walk()
    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent():
            break


def trasverse_my_tree(node:mytree_node):
    myqueue = [node]
    while myqueue:
        onenode = myqueue.pop()
        yield onenode
        myqueue.extend(onenode.children)

class Defense_technique:
    def __init__(self,attack_code = []) -> None:
        self.attack_code = attack_code[:]
    
    def return_root_node(self,code,lang):
        tree = mytree(code,lang,self.attack_code)
        return tree.my_root_node
    
    def valid_code(self,code,lang):
        parser = parsers[lang]
        tree = parser[0].parse(bytes(code, 'utf-8'))
        for node in traverse_tree(tree):
            if node.is_error:
                return False
        return True
    
    def judge_print_statement(self,node,lang):
            def c(node):
                if node.type == 'expression_statement':
                    expression = node.children[0]
                    if expression.type == 'call_expression':
                        function = expression.child_by_field_name('function')
                        if function.type == 'identifier' and function.text.decode("utf-8") == 'printf':
                            return True
                return False
            #解析java语言，判断node是否为print语句
            def java(node: Node) -> bool:
                if node.type == 'expression_statement':
                    expression = node.children[0]
                    if expression.type == 'method_invocation':
                        method = expression.child_by_field_name('name')
                        if method.type == 'identifier' and method.text.decode("utf-8") == 'println':
                            return True
                return False
            
            def javascript(node):
                if node.type == 'expression_statement':
                    expression = node.children[0]
                    if expression.type == 'call_expression':
                        callee = expression.child_by_field_name('function')
                        if callee.type == 'member_expression' and callee.text.decode("utf-8") == 'console.log':
                            return True
                return False
            def python(node: Node) -> bool:
                if node.type == 'expression_statement':
                    expression = node.children[0]
                    if expression.type == 'call':
                        callee = expression.child_by_field_name('function')
                        if callee.type == 'identifier' and callee.text.decode("utf-8") == 'print':
                            return True
                return False
        
            functions = {"cpp":c,"java":java,"python":python,"javascript":javascript}
        
            return functions[lang](node)
    

    def judge_false_condition(self,node,lang):
        #检查条件是否为永假
        if node.type == condition_stmt_name_convert_dict[lang]["if_statement"] or node.type == condition_stmt_name_convert_dict[lang]["while_statement"]:
            condition_node = node.child_by_field_name('condition')
            if condition_node:
                condition_code = re.findall("[0-9a-z]+",condition_node.text.decode("utf-8").lower())[0]
                if condition_code == '0' or condition_code == 'false':
                    return True
            else:
                condition_node = node.children[1]
                if condition_node:
                    condition_code = re.findall("[0-9a-z]+",condition_node.text.decode("utf-8").lower())[0]
                    if condition_code == '0' or condition_code == 'false':
                        return True


    def judge_difinition_of_a_var(self,node,var,lang):
        def cpp(node:Node,var):
            var_name = ""
            #print(node,node.text.decode("utf-8"))
            if node.type in ["declaration"]:
                var_name = node.child_by_field_name("declarator")
                if var_name:
                    var_name = var_name.child_by_field_name("declarator")
                    if var_name:
                        var_name = var_name.text.decode("utf-8")
                        if var_name in var:
                            return True
                #print(var_name)
            return False
        
        #解析java语言，检查node是否为var的定义或赋值语句
        def java(node: Node, var: str) -> bool:
            #print(node,node.text.decode("utf-8"))
            var_name = ""
            if node.type == 'local_variable_declaration':
                child = node.child_by_field_name('declarator')
                var_name = child.child_by_field_name("name").text.decode("utf-8")
                if var_name in var:
                    return True
            
            return False
        def javascript(node: Node, var: str) -> bool:
            var_name = ""
            if node.type == 'variable_declarator':
                var_name = node.child_by_field_name('name').text.decode("utf-8")
                if var_name in var:
                    return True
            return False
        
        #解析python语言，检查node是否为var的定义或赋值语句
        def python(node: Node, var: str) -> bool:
            return False
        
        functions = {"cpp":cpp,"java":java,"python":python,"javascript":javascript}
        
        return functions[lang](node,var)

    def judge_difinition_or_assign_for_a_var(self,node,var,lang):
        #检查此node是否为var的定义或赋值语句
        def cpp(node:Node,var):
            #print(node,node.text.decode("utf-8"))
            var_name = ""
            f = False
            right = None
            if node.type in ["parameter_declaration","init_declarator","declaration","reference_declarator"]:
                var_name = node.child_by_field_name("declarator")
                if var_name:
                    var_name = var_name.text.decode("utf-8")
                right = node.child_by_field_name("value")
                f = True
            elif node.type == "assignment_expression":
                child = node.child_by_field_name("left")
                var_name = child.text.decode("utf-8")
                right = node.child_by_field_name("right")
                f = True
            elif node.type == "update_expression":
                child = node.child_by_field_name("argument")
                var_name = child.text.decode("utf-8")
                right = node.child_by_field_name("operator")
                f = True
            if var_name == var:
                    return True,var_name,f,right
            return False,var_name,f,right
        
        #解析java语言，检查node是否为var的定义或赋值语句
        def java(node: Node, var: str) -> bool:
            #print(node,node.text.decode("utf-8"))
            var_name = ""
            f = False
            right = None
            if node.type == 'local_variable_declaration':
                child = node.child_by_field_name('declarator')
                var_name = child.child_by_field_name("name").text.decode("utf-8")
                right = child.child_by_field_name('value')
                f = True
            elif node.type == 'assignment_expression':
                var_name = node.child_by_field_name('left').text.decode("utf-8")
                right = node.child_by_field_name('right')
                f = True
            elif "update_expression" in node.type:
                var_name = node.children[0].text.decode("utf-8")
                right = node.children[1]
                f = True

            if var_name == var:
                return True,var_name,f,right
            
            return False,var_name,f,right
        def javascript(node: Node, var: str) -> bool:
            var_name = ""
            f = False
            right = None
            if node.type == 'variable_declarator':
                
                var_name = node.child_by_field_name('name').text.decode("utf-8")
                right = node.child_by_field_name('value')
                f = True
            elif 'assignment_expression' in node.type:
                var_name = node.child_by_field_name('left').text.decode("utf-8")
                right = node.child_by_field_name('right')
                f = True
            elif "update_expression" in node.type:
                var_name = node.child_by_field_name('argument').text.decode("utf-8")
                right = node.child_by_field_name('operator')
                f = True
            if var_name == var:
                return True,var_name,f,right
            return False ,var_name,f,right
        
        #解析python语言，检查node是否为var的定义或赋值语句
        def python(node: Node, var: str) -> bool:
            var_name = ""
            f = False
            right = None
            if node.type == 'assignment':
                left_child = node.child_by_field_name('left')
                right = node.child_by_field_name('right')
                var_name = left_child.text.decode("utf-8")
                f = True
                if left_child and  var_name== var:
                    return True,var_name,f,right
            return False,var_name,f,right
        
        functions = {"cpp":cpp,"java":java,"python":python,"javascript":javascript}
        
        #返回值 1.代表是不是var的定义或赋值语句;2.代表从这个node中得到的var的名字（如果是定义或赋值语句);3.代表这个node是否是定义或赋值语句;4.代表从这个node中得到的right的名字(如果是定义或赋值语句)
        return functions[lang](node,var)
      

    def judge_no_use_difinition_or_assign(self,node,lang,vars_use_pos):
        #检查左侧变量永远没有使用到的定义或赋值语句
        #首先要判断这个node是否为一个定义或赋值语句
        _,left,f,_ = self.judge_difinition_or_assign_for_a_var(node,"",lang)
        if f:
            if len(vars_use_pos.get(left,set())) == 0:
                return True
        return False
        
    def find_use_var_pos(self,code,lang):
        #找到一段代码中所有将var作为右值使用的位置
        vars, _ = get_identifiers(code,lang)
        ret = {}
        for var in vars:
            ret[var[0]] = self.find_use_stmts_for_a_var(code,var[0],lang)   

        return ret

    def find_use_stmts_for_a_var(self,code,var,lang):
        #找到code中对于一个变量var的所有作为右值使用的语句
        parser = parsers[lang]
        tree = parser[0].parse(bytes(code, 'utf-8'))
        ret = set()
        #问题在这里，这里应该dfg
        root_node = tree.root_node
        
        def dfg(node):
            f,_,_,_ = self.judge_difinition_or_assign_for_a_var(node,var,lang)
            if f:
                return 
            elif node.text.decode("utf-8") == var:
                ret.add(node)
                return
            for child in node.children:
                dfg(child) 
            
        dfg(root_node)
        ret = list(ret)
        ret.sort(key = lambda x:x.end_byte)
        return ret


    def dead_code_judge(self,code,lang):
        #判断一段代码中所有存在的死代码，并将死代码的位置返回
        ret = set()
        vars_use_pos = self.find_use_var_pos(code,lang)
        root_node = self.return_root_node(code,lang)
        def dfg(node:mytree_node):
            
            if self.judge_false_condition(node.node,lang) and not node.is_source: #永远不会执行的死代码
                ret.add(node.node)
                return
            elif self.judge_no_use_difinition_or_assign(node.node,lang,vars_use_pos) and not node.is_source: #永远不会使用的定义或赋值语句
                if node.node.prev_sibling and node.node.prev_sibling.text.decode("utf-8") in all_keywords[lang] :
                    ret.add(node.node.prev_sibling)
                if node.node.next_sibling and node.node.next_sibling.text.decode("utf-8") == stmt_separators[lang]:
                    ret.add(node.node.next_sibling)
                ret.add(node.node)
                return
            elif self.judge_print_statement(node.node,lang) and not node.is_source: #print语句也是死代码
                ret.add(node.node)
                return
            for child in node.children:
                dfg(child)
        
        dfg(root_node)
        ret = list(ret)
        ret.sort(key = lambda x:x.end_byte)
        return ret
                    
    def delete_dead_code(self,code,lang):
        #删除一段代码中所有的死代码
        ret = self.dead_code_judge(code,lang)
        acc = 0
        for v in ret:
            code,acc = replace_node_of_code(code[:],v,"",acc,lang)
        return code


    #下面是简化版常量传播
    def find_var_assignments(self,code,var,lang):
        #根据code找到某个变量的所有定义和赋值语句
        parser = parsers[lang]
        tree = parser[0].parse(bytes(code, 'utf-8'))
        ret = set()
        for node in traverse_tree(tree):
            f,left,_,right = self.judge_difinition_or_assign_for_a_var(node,var,lang)
            if f and right:
                ret.add(right)
        
        ret = list(ret)
        ret.sort(key = lambda x:x.end_byte)
        return ret

    def find_var_relations(self,code,var,lang):
        #如果一个var是右侧变量，那么它对应的所有左侧变量进行关联
        parser = parsers[lang]
        tree = parser[0].parse(bytes(code, 'utf-8'))
        ret = set()
        
           
        for node in traverse_tree(tree):   
            _,left,f,right = self.judge_difinition_or_assign_for_a_var(node,"",lang)
            if f:#是一个定义或赋值语句
                if right :
                    if right.text.decode("utf-8") == var:#右侧是var
                        ret.add(left) #左侧就是关联的变量
                    children = right.children
                    for onechild in children:
                        if onechild.text.decode("utf-8") == var:
                            ret.add(left)
        
        return ret

    #对一段代码中所有变量创建一个dict，表示这个变量的取值



    def is_constant_or_identifier(self,node,num,lang):
            # 判断node是否是一个常量或者是一个identifier，如果是，返回node.text（忽略node中的符号)
            # 返回值1：是否为常量；返回值2：是否为变量；返回值3：node.text

            if node.type in literal_comparsion[lang]["normal"] + literal_comparsion[lang]["container"]:
                num += 1
                return num == 1,False,node.text.decode("utf-8")
            elif is_valid_variable_name(node.text.decode("utf-8"),lang):
                num += 1
                return False,num == 1,node.text.decode("utf-8")
            
            return False,False,node.text.decode("utf-8")

    def is_function_call(self,node,lang):
        #判断是否为function call
        if node.type.count("call") > 0:
            return True
        return False

    def constant_propogation(self,code,lang):
        #常量传播
        vars,_ = get_identifiers(code,lang)
        var_values = {var[0]:"UDF" for var in vars} #值统一用python的str来表示
        # "UDF"表示undefined,"NAC"表示not a constant
        work_list = [var[0] for var in vars]
        var_values_copy = copy.deepcopy(var_values)
        var_assigns = {var[0]:self.find_var_assignments(code,var[0],lang) for var in vars}
        var_relations = {var[0]:self.find_var_relations(code,var[0],lang) for var in vars}
        #print(var_assigns)
        while work_list:
            var_values_copy = copy.deepcopy(var_values)
            work_list_temp = []
            for var in work_list:
                value = copy.deepcopy(var_values.get(var,"UDF"))
                all_assigns = var_assigns.get(var,[])
                for right_var in all_assigns:
                    if self.is_function_call(right_var,lang):
                        var_values[var] = "NAC"
                        continue
                    is_constant,is_identifier,ret_str = self.is_constant_or_identifier(right_var,0,lang)
                    if is_constant:
                        if var_values[var] == "UDF":
                            var_values[var] = ret_str
                        elif var_values[var] != ret_str:
                            var_values[var] = "NAC"
                        #如果右侧是常量，那么就是一个常量，如果右侧是变量，还需要进一步判断
                    else:
                        #如果右侧是变量,首先考虑右侧只有一个变量的情况
                        #这个判断有问题，应该判断right_var中是否只有一个identifier，如果只有一个identifier，返回它即可
                        if is_identifier:
                            #右侧只有一个变量
                            if var_values[var] == "UDF":
                                var_values[var] = var_values.get(ret_str,"UDF")
                            elif var_values[var] != var_values.get(ret_str,"UDF") and var_values.get(ret_str,"UDF") != "UDF":
                                var_values[var] = "NAC"
                        else:
                            #右侧超过一个变量,说明右侧是一个表达式，这时候就需要计算右侧表达式的值到底是不是一个常量
                            #这下面一定是要多于一个identifier或者常量的表达式才行，否则就有问题
                            #右侧为表达式统一为"NAC",不考虑过多，会引起混乱
                            var_values[var] = "NAC"
                            

                if value != var_values.get(var,"UDF"):
                    work_list_temp.extend(var_relations[var])
            work_list = copy.deepcopy(work_list_temp)
            if var_values_copy == var_values:
                break
        
        return var_values


    def edit_code_by_constant_information(self,code,lang):
        #根据常量传播的结果将code的所有constant转换
        constant_res = self.constant_propogation(code,lang)
        #找到constrant_res中是常量的部分
        code_res = copy.deepcopy(code)
        print(constant_res)
        vars_use_pos = self.find_use_var_pos(code,lang)
        replace_order = []
        constant_vars = set()
        for k,v in constant_res.items():
            if v!="UDF" and v!="NAC":
                #这里说明k变量是一个常量，接下来将k变量的所有位置替换掉
                for node in vars_use_pos[k]:
                    #变量k的所有位置
                    replace_order.append([k,v,node])
                constant_vars.add(k)
        replace_order.sort(key=lambda x:x[2].end_byte)
        acc = 0
        for value in replace_order:
            k = value[0]
            v = value[1]
            node = value[2]
            code_res,acc = replace_node_of_code(code_res,node,v,acc,lang)


        parser = parsers[lang]
        tree = parser[0].parse(bytes(code_res, 'utf-8'))
        acc = 0
        for node in traverse_tree(tree):
            if self.judge_difinition_of_a_var(node,constant_vars,lang):
                code_res,acc = replace_node_of_code(code_res,node,"",acc,lang)

        return code_res
    
    def get_example(self,code, tgt_word, substitute, lang):
        #为什么不直接用code.replace?因为我们只replace identifier,不会将code中所有与其相符的字符串都replace掉，这样有可能会有问题
        #如：一个identifier为f，如果直接用code.replace("f","g")，那么所有的f都会被替换掉，这样就会有问题。if语句里面的f就错了
        parser = parsers[lang]
        tree = parser[0].parse(bytes(code, 'utf-8'))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        replace_pos = {}
        for index, code_token in enumerate(code_tokens):
            if code_token == tgt_word:
                try:
                    replace_pos[tokens_index[index][0][0]].append((tokens_index[index][0][1], tokens_index[index][1][1]))
                except:
                    replace_pos[tokens_index[index][0][0]] = [(tokens_index[index][0][1], tokens_index[index][1][1])]
        diff = len(substitute) - len(tgt_word)
        for line in replace_pos.keys():
            for index, pos in enumerate(replace_pos[line]):
                code[line] = code[line][:pos[0]+index*diff] + substitute + code[line][pos[1]+index*diff:]

        return "\n".join(code)
    
    def outlier_detection(self,code:str,lang,word_embeddings,tokenizer,threshold):
        #复现DAMP论文中的outlier_detection
        identifiers,_ = get_identifiers(code,lang)
        identifiers = [x[0] for x in identifiers]
        identifier_embedding_dict = {x:None for x in identifiers}
        for x in identifiers:
            tensors = tokenizer(x,return_tensors="pt",add_special_tokens = False)
            id_embedding = word_embeddings(tensors.input_ids.cuda())
            identifier_embedding_dict[x] = torch.sum(id_embedding,dim=1)[0]/id_embedding.shape[1]
            assert id_embedding.shape[-1] == len(identifier_embedding_dict[x])
        #计算所有的identifier之间的相似度
        all_similarities = {x:0 for x in identifiers}
        for i in range(len(identifiers)):
            embeddings = torch.zeros(size=(len(identifier_embedding_dict[identifiers[0]]),)).cuda()
            for j in range(len(identifiers)):
                if i == j:
                    continue
                embeddings += identifier_embedding_dict[identifiers[j]]
            sub_res = embeddings/(len(identifiers)-1) - identifier_embedding_dict[identifiers[i]]
            all_similarities[identifiers[i]] = torch.dot(sub_res.T,sub_res)
        simi_sort = []
        for k,v in all_similarities.items():
            simi_sort.append((k,v))
        simi_sort.sort(key=lambda x:x[1],reverse=True)
        #print(simi_sort)
        for id,simi in simi_sort:
            if simi > threshold:
                code = self.get_example(code[:],id,str(tokenizer.unk_token),lang)
            else:
                break
        return code
                
        
    
    def open_source_llm_recoginze_unrelated_identifier(self,code:str,lang,model,tokenizer):
        #询问大模型
        """
        我将给你一段代码，它是使用某种编程语言编写的。在这段代码中，有一些变量名被恶意修改，打破了原代码的可读性。修改后的变量名通常有如下特点：1. 和整段代码的执行功能不统一，如一段求和的代码，变量名却为div；2. 和代码中其他变量名格格不入，不适配。
请你帮我找出这些被修改后的变量名，并尝试猜测修改前的变量名是什么。
在代码中：带有<|identifier|>后缀的是一个变量，不需要你去识别哪些是一个变量
输出格式：原变量名 -> 修改后的变量名
        """
        prompt = """I will give you a piece of code written in the """ + lang + """ language. In this code, some variable names have been maliciously modified, breaking the original code's readability. The modified variable names usually have the following characteristics: 1. They are inconsistent with the overall functionality of the code, such as having a code snippet for summation but with a variable name like "div"; 2. They do not match other variable names in the code, appearing out of place.

Please help me identify these modified variable names and try to guess what the original variable names were.

In the code: Variables with suffix "<|identifier|>" are variables. You don't need to identify which ones are variables.

Output format: Original variable name -> Modified variable name """

        def preprocess_code(code:str):
            identifiers,_ = get_identifiers(code,lang)
            identifiers = [x[0] for x in identifiers]
            for x in identifiers:
                code = self.get_example(code[:],x,x + "<|identifier|>",lang)
            return code
            
        codepr = preprocess_code(code)      
        prompt = prompt + codepr
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs,max_new_tokens = 100,do_sample = True)
        output_str = tokenizer.decode(outputs[0])
        res = output_str.replace(prompt,"")
        print(res)
        re_res = re.findall("([\"a-zA-Z_0-9<|>]+) -> ([\"a-zA-Z_0-9 <|>]+)",res)
        print(re_res)
        for s,t in re_res:
            s = s.replace("<|identifier|>","")
            s = s.replace("\"","")
            t = re.findall("([a-zA-Z_0-9]+)",t)
            t = t[0]
            #print(s,t)
            code = self.get_example(code[:],s,t,lang)
        return code

    
    def gpt_3_llm_recoginze_unrelated_identifier(self,code:str,lang,model="gpt-4o-mini",seed = 42):
        openai.api_key = "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"
        
        prompt = """I will give you a piece of code written in the """ + lang + """ language. In this code, some of the variable names have been maliciously modified, breaking the original code's readability. The modified variable names usually have the following characteristics: 1. They are inconsistent with the overall functionality of the code, such as having a code snippet for summation but with a variable name like "div"; 2. They do not match other variable names in the code, appearing out of place.

Please help me identify these modified variable names and try to guess what the original variable names were. 

In the code: Variables with suffix "<|identifier|>" are variables. You don't need to identify which ones are variables. In the result, you need to remove the "<|identifier|>" suffix. And only one modified variable is good enough.

Output format: Original variable name -> Modified variable name"""

        def preprocess_code(code:str):
            identifiers,_ = get_identifiers(code,lang)
            identifiers = [x[0] for x in identifiers]
            for x in identifiers:
                code = self.get_example(code[:],x,x + "<|identifier|>",lang)
            return code
            
        codepr = preprocess_code(code)      
        
        chat_completion = openai.ChatCompletion.create(
            model = model,
            messages=[
                {
                "role": "system",
                "content": prompt
                },
                {
                "role": "user",
                "content": codepr
                }
            ],
            temperature = 0.3,
            top_p = 0.1,
            max_tokens=100,
            seed = seed
        )
        res = chat_completion.choices[0].message.content
        #print(res)
        re_res = re.findall("([\"a-zA-Z_0-9<|>]+) -> ([\"a-zA-Z_0-9 <|>]+)",res)
        #print(re_res)
        for s,t in re_res:
            s = s.replace("<|identifier|>","")
            s = s.replace("\"","")
            t = re.findall("([a-zA-Z_0-9]+)",t)
            t = t[0]
            #print(s,t)
            code = self.get_example(code[:],s,t,lang)
        return code
        
    
    def normalize_identifier_names(self,code,lang):
        #统一变量命名方式
        #这是一种re-training的防御方法，也可以不重训练（但可能影响performance）
        #缺点就是会影响代码的可读性，以及可能影响模型的performance，以及重训练代价比较大
        p = PerturbationPipeline()
        p.preprocess_code('',lang)
        return p.normalize_identifer(code)


    def google_code_style_reformat(self,code):
        '''
        根据https://google.github.io/styleguide/中描述的缩进和空行方式重新组织代码,使代码呈现良好的可读性
        1. 缩进组织:(1)制表符全部替换为空格
        2. 空行组织:(1)删除掉代码头部和尾部多余的空行;(2)两个连续空行,删除掉其中一个;
        '''
        
        #所有缩进使用四个空格
        pattern = re.compile(r"\t")#将所有制表符换为四个空格
        code = pattern.sub(" "*4,code)
        code = code.strip("\n")
        pattern = re.compile(r"\n\n")
        code = pattern.sub("\n",code)
        return code

    def google_renaming(self,code,lang):
        

        def python(name):
            '''
            采用snake_case命名法
            即单词都用小写字母,单词与单词之间使用下划线区分
            '''
            parts = re.sub(r'([A-Z_])', r' \1', name).split()
            real_parts = []
            for v in parts:
                if re.search(r'[0-9a-zA-Z]*',v):
                    x = re.findall(r'[0-9a-zA-Z]*',v)
                    for mini_v in x:
                        if mini_v:
                            real_parts.append(mini_v.lower())
            ret = name
            if len(real_parts) > 0:
                ret = '_'.join(real_parts)
            #print(ret)
            return ret


        def cpp(name):
            '''
            snake_case (all lowercase, with underscores between words).和python一样
            '''
            return python(name)

        def java(name):
            '''
            采用lowerCamelCase小驼峰命名法,即标识符首字母小写,单词之间使用大写字母进行区分
            '''
            parts = re.sub(r'([A-Z_])', r' \1', name).split()
            real_parts = []
            for v in parts:
                if re.search(r'[0-9a-zA-Z]',v):
                    x = re.findall(r'[0-9a-zA-Z]*',v)
                    for mini_v in x:
                        if mini_v:
                            real_parts.append(mini_v[0].upper() + mini_v[1:].lower())
            
            try:
                return real_parts[0].lower() + ''.join(real_parts[1:])
            except:
                return name

        def javascript(name):
            '''
            采用lowerCamelCase小驼峰命名法,即标识符首字母小写,单词之间使用大写字母进行区分
            '''
            return java(name)
        
        identifiers,_ = get_identifiers(code,lang)
        identifiers = [v[0] for v in identifiers]
        functions = {'python':python,'java':java,'cpp':cpp,'javascript':javascript}
        for v in identifiers:
            code = self.get_example(code[:],v,functions[lang](v),lang)
        
        return code

        

        


    
    
if __name__ == '__main__':
    # 1. 对注释不要动
    # 2. 原来的代码不要动，只修改我自己攻击的代码
    
    code = '''
#include <queue>                                                                                                                                                           
#include<iostream>                                                                                                                                                         
#include <vector>                                                                                                                                                          
#include <cassert>                                                                                                                                                         
using namespace std;                                                                                                                                                       
int no_of_subsequences(vector<int> arr, int k) {                                                                                                                           
    int n = arr.size();                                                                                                                                                    
    vector<vector<int>> dp(k + 1, vector<int>(n + 1, 0));
    for (int i = 1; i <= k; i++) {
        for (int j = 1; j <= n; j++) {
            dp[i][j] = dp[i][j - 1];
//begin to write code

    '''
    print(code)
    lang = "cpp"
    dt = Defense_technique()
    '''
    code = dt.delete_dead_code(code,lang)
    
    code = dt.google_code_style_reformat(code)
    code = [dt.normalize_identifier_names(code,lang),dt.outlier_detection(code,lang),dt.gpt_3_llm_recoginze_unrelated_identifier(code,lang)]
    '''
    code = dt.edit_code_by_constant_information(code,lang)
    #code = dt.google_renaming(code,lang)
    
    print(code)
    
