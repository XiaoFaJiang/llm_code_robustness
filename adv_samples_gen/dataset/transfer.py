import re
import json

def remove_comments_and_docstrings(code_str):
    """
    删除Python/Java/JavaScript/C++代码中的注释、文档字符串和空行
    """
    lines = code_str.split('\n')
    result_lines = []
    
    # 状态变量
    in_multiline_comment = False  # 用于C++/Java/JS的多行注释 /* ... */
    in_triple_quote = False  # 用于Python的三引号字符串/文档字符串
    triple_quote_char = None  # 记录是三单引号还是三双引号
    
    # 正则表达式模式
    single_line_comment_pattern = re.compile(r'^\s*(#|//)')  # Python单行注释或C++/Java/JS单行注释
    multiline_comment_start_pattern = re.compile(r'/\*')  # C++/Java/JS多行注释开始
    multiline_comment_end_pattern = re.compile(r'\*/')  # C++/Java/JS多行注释结束
    triple_quote_pattern = re.compile(r"(\"\"\"|\'\'\')")  # Python三引号
    
    i = 0
    while i < len(lines):
        line = lines[i]
        original_line = line
        
        # 如果不是在多行注释或三引号字符串中，检查是否开始新的多行注释或三引号
        if not in_multiline_comment and not in_triple_quote:
            # 检查三引号
            triple_quote_match = triple_quote_pattern.search(line)
            if triple_quote_match:
                # 找到三引号
                triple_quote_char = triple_quote_match.group(1)
                in_triple_quote = True
                # 移除从三引号开始到行尾的内容
                line = line[:triple_quote_match.start()]
                # 如果三引号后面还有内容，检查是否在同一行结束
                remaining_line = original_line[triple_quote_match.end():]
                if triple_quote_char in remaining_line:
                    # 同一行内结束了三引号
                    in_triple_quote = False
                    # 移除三引号之后的内容
                    end_quote_match = triple_quote_pattern.search(remaining_line)
                    if end_quote_match:
                        line = line + remaining_line[end_quote_match.end():]
            
            # 检查多行注释开始（如果不是在三引号中）
            elif not in_triple_quote:
                multiline_start_match = multiline_comment_start_pattern.search(line)
                if multiline_start_match:
                    in_multiline_comment = True
                    # 移除从多行注释开始到行尾的内容
                    line = line[:multiline_start_match.start()]
                    # 检查是否在同一行内结束了多行注释
                    remaining_line = original_line[multiline_start_match.end():]
                    multiline_end_match = multiline_comment_end_pattern.search(remaining_line)
                    if multiline_end_match:
                        # 同一行内结束了多行注释
                        in_multiline_comment = False
                        line = line + remaining_line[multiline_end_match.end():]
        
        # 如果在三引号中，检查是否结束
        elif in_triple_quote:
            if triple_quote_char in line:
                # 找到了结束的三引号
                end_quote_match = triple_quote_pattern.search(line)
                if end_quote_match:
                    # 移除从行开始到三引号结束的部分
                    line = line[end_quote_match.end():]
                    in_triple_quote = False
                    # 检查这一行三引号后面是否还有内容
                    remaining_line = line
                    # 检查是否还有另一个三引号开始（罕见情况）
                    triple_quote_match = triple_quote_pattern.search(remaining_line)
                    if triple_quote_match:
                        triple_quote_char = triple_quote_match.group(1)
                        in_triple_quote = True
                        line = remaining_line[:triple_quote_match.start()]
            else:
                # 仍然在三引号中，跳过这一行
                line = ""
        
        # 如果在多行注释中，检查是否结束
        elif in_multiline_comment:
            multiline_end_match = multiline_comment_end_pattern.search(line)
            if multiline_end_match:
                # 多行注释结束
                in_multiline_comment = False
                # 移除从行开始到多行注释结束的部分
                line = line[multiline_end_match.end():]
                # 检查这一行注释后面是否还有内容
                remaining_line = line
                # 检查是否立即开始另一个多行注释（罕见情况）
                multiline_start_match = multiline_comment_start_pattern.search(remaining_line)
                if multiline_start_match:
                    in_multiline_comment = True
                    line = remaining_line[:multiline_start_match.start()]
            else:
                # 仍然在多行注释中，跳过这一行
                line = ""
        
        # 如果既不在三引号中也不在多行注释中，处理单行注释
        if not in_multiline_comment and not in_triple_quote:
            # 移除单行注释
            if single_line_comment_pattern.match(line.strip()):
                line = ""
        
        # 去除行尾空格
        line = line.rstrip()
        
        # 如果不是空行，添加到结果中
        if line.strip():
            result_lines.append(line)
        
        i += 1
    
    return '\n'.join(result_lines)

split = "valid"
data = []
with open(f"{split}_base.jsonl","r") as f:
    for line in f:
        data.append(json.loads(line))

rewrite_data = []
cnt = 0
for k,v in enumerate(data):
    st1 = v["code_str_generate"]
    st2 = v["Adversarial truth"]
    removed_st1 = remove_comments_and_docstrings(st1)
    removed_st2 = remove_comments_and_docstrings(st2)
    if removed_st2.find(removed_st1) != -1:
        removed_st2 = removed_st2[len(removed_st1):]
        if not removed_st2:
            print('empty')
        item = {
            'prompt': st1,
            "response": removed_st2,
            'truth':st2
        }
        rewrite_data.append(item)

with open(f"{split}_base_test.jsonl","w") as f:
    for line in rewrite_data:
        f.write(json.dumps(line) + "\n")

    
