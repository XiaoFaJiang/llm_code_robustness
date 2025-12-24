import json
import re
import math

# 原始JSON数据
data = {
    "task_id": "Python/0",
    "prompt": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n",
    "declaration": "from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n",
    "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n",
    "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(has_close_elements):\n    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert has_close_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert has_close_elements([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\ncheck(has_close_elements)",
    "example_test": "def check(has_close_elements):\n    assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n    assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\ncheck(has_close_elements)\n"
}

def process_solution(solution):
    """删除solution的后2/3部分"""
    lines = solution.split('\n')
    total_lines = len(lines)
    # 保留前1/3的行数
    keep_lines = math.ceil(total_lines / 3)
    return '\n'.join(lines[:keep_lines])

def extract_prompt_info(prompt,lang):
    """从prompt中提取提示字段"""
    # 使用正则表达式匹配docstring中的内容
    patterns = {"python":r'[\"\'][\"\'][\"\']\s*(.*?)\s*[\"\'][\"\'][\"\']','cpp':r'\/\*\s*(.*?)\s*\*\/',\
                'javascript':r'\/\*\s*(.*?)\s*\*\/','java':r'\/\*\*\s*(.*?)\s*\*\/'}
    match = re.search(patterns[lang], prompt, re.DOTALL)
    if match:
        return match.group(1)
    return ""

def create_docstring(language, description):
    """根据编程语言创建相应的多行注释docstring"""
    docstring_templates = {
        "python": f'"""\n    program language: python\n    description: {description}\n    you must follow:\n    1. Provide the complete code without any textual explanations and do not generate test scripts.\n    2. Please strictly follow the specified format provided below for the code.\n    3. Do not change the function names.\n    4. The original code content must be fully included in the generated complete code, including all package import sections.\n    5. For Python language, do not generate test cases or main function.\n    6. Do not generate test cases.\n    """',
        "cpp": f'/*\n    program language: cpp\n    description: {description}\n    you must follow:\n    1. Provide the complete code without any textual explanations and do not generate test scripts.\n    2. Please strictly follow the specified format provided below for the code.\n    3. Do not change the function names.\n    4. The original code content must be fully included in the generated complete code, including all package import sections.\n    5. For C++ language, do not generate the main function; I have my own main function available.\n    6. Do not generate test cases.\n    */',
        "java": f'/**\n    program language: java\n    description: {description}\n    you must follow:\n    1. Provide the complete code without any textual explanations and do not generate test scripts.\n    2. Please strictly follow the specified format provided below for the code.\n    3. Do not change the function names.\n    4. The original code content must be fully included in the generated complete code, including all package import sections.\n    5. For Java language, do not generate the main method.\n    6. Do not generate test cases.\n    */',
        "javascript": f'/*\n    program language: javascript\n    description: {description}\n    you must follow:\n    1. Provide the complete code without any textual explanations and do not generate test scripts.\n    2. Please strictly follow the specified format provided below for the code.\n    3. Do not change the function names.\n    4. The original code content must be fully included in the generated complete code, including all package import sections.\n    5. For JavaScript language, do not generate test cases.\n    6. Do not generate test cases.\n    */'
    }
    
    return docstring_templates.get(language.lower(), docstring_templates["python"])

def create_code_str_generate(prompt, deleted_solution, language, description):
    """创建code_str_generate字段，包含特定语言的docstring"""
    docstring = create_docstring(language, description)
    code_str = docstring + '\n\n' + prompt + deleted_solution
    
    return code_str

languages = ['javascript']
for lang in languages:
    local_path = f"multi_languages_humaneval/humaneval_{lang}_tested.jsonl"
    data = []
    with open(local_path) as f:
        for line in f:
            data.append(json.loads(line))

    writed_data = []
    for d in data:
        # 1. 提取prompt信息
        prompt_info = extract_prompt_info(d["prompt"],lang)

        # 2. 删除solution的后2/3部分
        deleted_solution = process_solution(d["canonical_solution"])

        code_str_generate = create_code_str_generate(
                d["prompt"], 
                deleted_solution, 
                lang, 
                prompt_info
            )
            

        # 创建处理后的数据结构（以Python为例，因为原始代码是Python）
        processed_data = {
            "task_id": d["task_id"],
            "code_str": d["prompt"] + d["canonical_solution"],
            "test": d["test"],
            "prompt": prompt_info,
            f"{lang}_prompt": prompt_info,
            "code_str_deleted": d["prompt"] + deleted_solution, #给instruct模型的，没有doc_string
            "code_str_generate":code_str_generate, #给base模型的，有doc_string
            "example_test": d["example_test"],
            
        }

        writed_data.append(processed_data)
    # 保存到文件
    with open(f"humaneval_{lang}_tested.json", "w", encoding="utf-8") as f:
        json.dump(writed_data, f, indent=2, ensure_ascii=False)
