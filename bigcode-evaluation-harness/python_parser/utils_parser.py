import torch
import torch.nn as nn
import copy
import random
import sys
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import csv
python_keywords = ['import', '', '[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">",
                   '+', '-', '*', '/', 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
                   'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global',
                   'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
                   'while', 'with', 'yield']
java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "do", "double", "else", "enum",
                 "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
                 "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
                 "short", "static", "strictfp", "super", "switch", "throws", "transient", "try", "void", "volatile",
                 "while"]
java_special_ids = ["main", "args", "Math", "System", "Random", "Byte", "Short", "Integer", "Long", "Float", "Double", "Character",
                    "Boolean", "Data", "ParseException", "SimpleDateFormat", "Calendar", "Object", "String", "StringBuffer",
                    "StringBuilder", "DateFormat", "Collection", "List", "Map", "Set", "Queue", "ArrayList", "HashSet", "HashMap"]

java_all_names = java_keywords + java_special_ids

c_keywords = ["auto", "break", "case", "char", "const", "continue",
                 "default", "do", "double", "else", "enum", "extern",
                 "float", "for", "goto", "if", "inline", "int", "long",
                 "register", "restrict", "return", "short", "signed",
                 "sizeof", "static", "struct", "switch", "typedef",
                 "union", "unsigned", "void", "volatile", "while",
                 "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                 "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                 "_Thread_local", "__func__"]

c_macros = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
              "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
              "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]     # <stdlib.h> macro
c_special_ids = ["main",  # main function
                   "stdio", "cstdio", "stdio.h",                                # <stdio.h> & <cstdio>
                   "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",     # <stdio.h> types & streams
                   "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", # <stdio.h> functions
                   "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                   "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                   "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                   "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                   "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                   "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                   "stdlib", "cstdlib", "stdlib.h",                             # <stdlib.h> & <cstdlib>
                   "size_t", "div_t", "ldiv_t", "lldiv_t",                      # <stdlib.h> types
                   "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                   "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                   "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                   "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                   "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                   "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                   "mbstowcs", "wcstombs",
                   "string", "cstring", "string.h",                                 # <string.h> & <cstring>
                   "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",     # <string.h> functions
                   "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                   "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                   "strpbrk" ,"strstr", "strtok", "strxfrm",
                   "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",      # <string.h> extension functions
                   "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                   "iostream", "istream", "ostream", "fstream", "sstream",      # <iostream> family
                   "iomanip", "iosfwd",
                   "ios", "wios", "streamoff", "streampos", "wstreampos",       # <iostream> types
                   "streamsize", "cout", "cerr", "clog", "cin",
                   "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",    # <iostream> manipulators
                   "noshowbase", "showpoint", "noshowpoint", "showpos",
                   "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                   "left", "right", "internal", "dec", "oct", "hex", "fixed",
                   "scientific", "hexfloat", "defaultfloat", "width", "fill",
                   "precision", "endl", "ends", "flush", "ws", "showpoint",
                   "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",    # <math.h> functions
                   "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                   "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                   "modf", "fmod", "hypot", "ldexp", "poly", "matherr","std"]

special_char = ['[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/',
                '|']

c_all_names = c_keywords + c_macros + c_special_ids + special_char

go_keywords = ['break','default','func','interface','select',
'case', 'defer','go','map','struct',
'chan','else','goto','package','switch',
'const','fallthrough','if','range','type',
'continue','for','import','return','var',"main","fmt","Println"]

go_predefined_identifiers = [
    "bool", "byte", "complex64", "complex128", "error",
    "float32", "float64", "int", "int8", "int16",
    "int32", "int64", "rune", "string", "uint",
    "uint8", "uint16", "uint32", "uint64", "uintptr",
    "true", "false", "iota",
    "nil",
    "append", "cap", "close", "complex", "copy",
    "delete", "imag", "len", "make", "new",
    "panic", "print", "println", "real", "recover"
]

go_all_names = go_keywords + go_predefined_identifiers

js_keywords = [
    "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete", "do",
    "else", "export", "extends", "finally", "for", "function", "if", "import", "in", "instanceof",
    "new", "return", "super", "switch", "this", "throw", "try", "typeof", "var", "void",
    "while", "with", "yield","let","const","console","log"
]

# JavaScript 未来保留字
js_future_reserved_words = [
    "enum", "await", 
    # 严格模式下的保留字
    "implements", "package", "protected", "interface", "private", "public"
]

# JavaScript 全局对象、函数和类
js_global_objects = [
    "Array", "Boolean", "Date", "Error", "Function", "JSON", "Math", "Number", "Object", "RegExp",
    "String", "Map", "Set", "WeakMap", "WeakSet", "Symbol", "Promise", "Proxy", "Reflect",
    "GlobalThis", "Infinity", "NaN", "undefined", "null", "eval", "isFinite", "isNaN",
    "parseFloat", "parseInt", "decodeURI", "decodeURIComponent", "encodeURI", "encodeURIComponent"
]

# 合并所有列表
js_all_names = js_keywords + js_future_reserved_words + js_global_objects

php_keywords = [
    "__halt_compiler", "abstract", "and", "array", "as", "break", "callable", "case", "catch", "class",
    "clone", "const", "continue", "declare", "default", "die", "do", "echo", "else", "elseif",
    "empty", "enddeclare", "endfor", "endforeach", "endif", "endswitch", "endwhile", "eval", "exit", "extends",
    "final", "finally", "fn", "for", "foreach", "function", "global", "goto", "if", "implements",
    "include", "include_once", "instanceof", "insteadof", "interface", "isset", "list", "match", "namespace", "new",
    "or", "print", "private", "protected", "public", "readonly", "require", "require_once", "return", "static",
    "switch", "throw", "trait", "try", "unset", "use", "var", "while", "xor", "yield",
    "yield from"
]

# PHP 魔术常量
php_magic_constants = [
    "__LINE__", "__FILE__", "__DIR__", "__FUNCTION__", "__CLASS__", "__TRAIT__", "__METHOD__", "__NAMESPACE__"
]

# 常用的预定义常量和类名（示例）
php_common_predefined = [
    "TRUE", "FALSE", "NULL", "PHP_VERSION", "PHP_INT_MAX", "PHP_INT_MIN", "PHP_FLOAT_MAX", "PHP_FLOAT_MIN",
    "DIRECTORY_SEPARATOR", "PATH_SEPARATOR", "STDIN", "STDOUT", "STDERR"
]

# 合并所有列表
php_all_names = php_keywords + php_magic_constants + php_common_predefined

ruby_keywords = [
    "BEGIN", "END", "__ENCODING__", "__END__", "__FILE__", "__LINE__",
    "alias", "and", "begin", "break", "case", "class", "def", "defined?",
    "do", "else", "elsif", "end", "ensure", "false", "for", "if", "in",
    "module", "next", "nil", "not", "or", "redo", "rescue", "retry", "return",
    "self", "super", "then", "true", "undef", "unless", "until", "when",
    "while", "yield"
]


rust_reserved_keywords = [
    "as", "break", "const", "continue", "crate", "else", "enum", "extern",
    "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod",
    "move", "mut", "pub", "ref", "return", "self", "Self", "static", "struct",
    "super", "trait", "true", "type", "unsafe", "use", "where", "while",
    "async", "await", "dyn"
]

rust_weak_keywords = [
    "abstract", "become", "box", "do", "final", "macro", "override", "priv",
    "typeof", "unsized", "virtual", "yield", "try"
]

rust_keywords = rust_reserved_keywords + rust_weak_keywords

csharp_keywords = [
    "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked",
    "class", "const", "continue", "decimal", "default", "delegate", "do", "double", "else",
    "enum", "event", "explicit", "extern", "false", "finally", "fixed", "float", "for",
    "foreach", "goto", "if", "implicit", "in", "int", "interface", "internal", "is", "lock",
    "long", "namespace", "new", "null", "object", "operator", "out", "override", "params",
    "private", "protected", "public", "readonly", "ref", "return", "sbyte", "sealed", "short",
    "sizeof", "stackalloc", "static", "string", "struct", "switch", "this", "throw", "true",
    "try", "typeof", "uint", "ulong", "unchecked", "unsafe", "ushort", "using", "virtual",
    "void", "volatile", "while","System","Program","Main","Console","WriteLine"
]

# C# 上下文关键字
csharp_contextual_keywords = [
    "add", "alias", "ascending", "async", "await", "by", "descending", "dynamic", "equals",
    "from", "get", "global", "group", "into", "join", "let", "nameof", "on", "orderby",
    "partial", "remove", "select", "set", "value", "var", "when", "where", "yield"
]

# 常见的预定义类型名称
csharp_predefined_types = [
    "Object", "String", "Boolean", "Byte", "SByte", "Char", "Decimal", "Double", "Single",
    "Int32", "UInt32", "Int64", "UInt64", "Int16", "UInt16", "Void"
]

# 合并列表
csharp_avoid = csharp_keywords + csharp_contextual_keywords + csharp_predefined_types


all_keywords = {'c':c_all_names, 'c_sharp':csharp_avoid, 'go':go_all_names, 'java':java_all_names, 'javascript':js_all_names, 'php':php_all_names, 'python':python_keywords, 'ruby':ruby_keywords, 'rust':rust_keywords}

stmt_separators = {
    "c": ";",
    "c_sharp": ";",
    "go": "\n", # 在正式代码中通常是可选的，由编译器自动插入
    "java": ";",
    "javascript": ";", # 虽然在某些情况下可以省略，但通常用于分隔语句
    "php": ";",
    "python": "\n", # 或使用分号 (;) 来在同一行中分隔多个语句
    "ruby": "\n", # 或使用分号 (;)
    "rust": ";", # 表达式语句以分号结尾，但定义和其他一些语句不需要
}


block_separators = {
    "c": ["{", "}"],  # C语言使用大括号来分隔代码块
    "c_sharp": ["{", "}"],  # C#也使用大括号
    "go": ["{", "}"],  # Go语言使用大括号
    "java": ["{", "}"],  # Java使用大括号
    "javascript": ["{", "}"],  # JavaScript使用大括号
    "php": ["{", "}"],  # PHP使用大括号
    "python": ["\n"],  # Python使用缩进来分隔代码块，因此只有一个“符号”
    "rust": ["{", "}"],  # Rust使用大括号
    "ruby":["\t"]
}



literal_comparsion = {
  "c": {
    "normal": ['char_literal', 'compound_literal_expression', 'number_literal', 'string_literal'],
    "container": []
  },
  "c_sharp": {
    "normal": ['integer_literal', 'preproc_integer_literal', \
        'string_literal_encoding', 'verbatim_string_literal', 'real_literal', 'character_literal_unescaped', \
            'character_literal', 'string_literal', 'preproc_string_literal', 'raw_string_literal', 'null_literal', 'boolean_literal',"false","true"],
    "container": ["array_creation_expression"]
  },
  "cpp": {
    "normal": ['char_literal', 'raw_string_literal', 'compound_literal_expression', 'string_literal', 'number_literal', 'literal_suffix', 'user_defined_literal'],
    "container": ["initializer_list"]
  },
  "go": {
    "normal": ['interpreted_string_literal', 'rune_literal', 'literal_value', 'int_literal', 'raw_string_literal', 'imaginary_literal', 'func_literal', 'literal_element', 'composite_literal', 'float_literal',"true", "false", "nil"],
    "container": ["composite_literal"]
  },
  "java": {
    "normal": ['decimal_integer_literal', 'hex_integer_literal', 'character_literal', 'octal_integer_literal', 'decimal_floating_point_literal',\
        'binary_integer_literal', 'null_literal', 'string_literal', 'hex_floating_point_literal', 'class_literal', '_literal',"false","true"],
    "container": ["array_initializer"]
  },
  "javascript": {
    "normal": ["numeric_literal", "string_literal", "boolean_literal", "null","false","true"],
    "container": ["array", "object"]
  },
  "php": {
    "normal": ["integer", "float", "string", "boolean", "null"],
    "container": ["array"]
  },
  "python": {
    "normal": ["integer", "float", "imaginary", "string", "True", "False", "None"],
    "container": ["list", "tuple", "dict", "set"]
  },
  "ruby": {
    "normal": ["int", "float", "string", "symbol", "true", "false", "nil"],
    "container": ["array", "hash"]
  },
  "rust": {
    "normal": ['float_literal', 'boolean_literal', 'negative_literal', 'literal', '_literal_pattern', 'raw_string_literal', '_literal', 'string_literal', 'char_literal', 'integer_literal'],
    "container": ["array_expression", "struct_expression", "tuple_expression"]
  }
}
#这些node_type在tree_sitter中为常量类型,其中normal一定为常量,container还需判断其中的所有元素是否都为常量


all_languages_constant_variable_types = {
    "c": {
        "char_literal": ["char"],
        "compound_literal_expression": [""],  # C语言中，复合字面量的类型依赖于具体复合类型的定义，不直接书写类型
        "number_literal": ["double", "int"],  # 数字字面量默认为int，但double为更通用类型
        "string_literal": ["char[]"],
    },
    "c_sharp": {
        "integer_literal": ["int"],
        "string_literal_fragment": [""],  # 动态类型，不直接书写类型
        "preproc_integer_literal": [""],  # 预处理器中使用，不直接书写类型
        "string_literal_encoding": ["string"],
        "verbatim_string_literal": ["string"],
        "real_literal": ["double", "float"],
        "character_literal_unescaped": ["char"],
        "character_literal": ["char"],
        "string_literal": ["string"],
        "preproc_string_literal": [""],  # 预处理器中使用，不直接书写类型
        "raw_string_literal": ["string"],
        "null_literal": ["null"],
        "boolean_literal": ["bool"],
        "false": ["bool"],
        "true": ["bool"],
    },
    "go": {
        "interpreted_string_literal": ["string"],
        "rune_literal": ["rune"],
        "literal_value": [""],  # 根据上下文推断，不直接书写类型
        "int_literal": ["int"],
        "raw_string_literal": ["string"],
        "imaginary_literal": ["complex128", "complex64"],
        "func_literal": ["func"],  # 函数字面量类型由其签名决定
        "literal_element": [""],  # 根据上下文推断，不直接书写类型
        "composite_literal": [""],  # 根据上下文推断，不直接书写类型
        "float_literal": ["float64", "float32"],
        "true": ["bool"],
        "false": ["bool"],
        "nil": ["nil"],
    },
    "java": {
        "decimal_integer_literal": ["int", "long"],
        "hex_integer_literal": ["int", "long"],
        "character_literal": ["char"],
        "octal_integer_literal": ["int", "long"],
        "decimal_floating_point_literal": ["double", "float"],
        "binary_integer_literal": ["int", "long"],
        "null_literal": ["null"],
        "string_literal": ["String"],
        "hex_floating_point_literal": ["double", "float"],
        "class_literal": [""],  # 类字面量，用于反射，类型为Class
        "_literal": [""],  # 特殊情况，不直接书写类型
        "false": ["boolean"],
        "true": ["boolean"],
    },
    "javascript": {
        "numeric_literal": [""],  # JavaScript为动态类型语言，不直接书写类型
        "string_literal": [""],
        "boolean_literal": [""],
        "null": [""],
        "false": [""],
        "true": [""],
    },
    "php": {
        "integer": [""],  # PHP为动态类型语言，不直接书写类型
        "float": [""],
        "string": [""],
        "boolean": [""],
        "null": [""],
    },
    "python": {
        "integer": [""],  # Python为动态类型语言，不直接书写类型
        "float": [""],
        "imaginary": [""],
        "string": [""],
        "True": [""],
        "False": [""],
        "None": [""],
    },
    "ruby": {
        "int": [""],  # Ruby为动态类型语言，不直接书写类型
        "float": [""],
        "string": [""],
        "symbol": [""],
        "true": [""],
        "false": [""],
        "nil": [""],
    },
    "rust": {
        "float_literal": ["f64", "f32"],  # 浮点字面量，默认为f64，可通过后缀指定为f32
        "boolean_literal": ["bool"],
        "negative_literal": ["i32", "i64", "f32", "f64"],  # 负数字面量，默认为i32，可为其他整数或浮点类型
        "literal": [""],  # 根据上下文推断，不直接书写类型
        "_literal_pattern": [""],  # 用于模式匹配，不直接书写类型
        "raw_string_literal": ["&str"],  # 原生字符串字面量
        "_literal": [""],  # 特殊占位符，可能表示任何字面量，根据上下文推断
        "string_literal": ["&str"],
        "char_literal": ["char"],
        "integer_literal": ["i32", "u32", "i64", "u64"],  # 整数字面量，默认为i32，可通过后缀指定其他整数类型
    },
}

condition_stmt_name_convert_dict = {"cpp":{"if_statement":"if_statement","while_statement":"while_statement"}, "java":{"if_statement":"if_statement","while_statement":"while_statement"}, \
    "python":{"if_statement":"if_statement","while_statement":"while_statement"},\
            "javascript":{"if_statement":"if_statement","while_statement":"while_statement"},\
                "java":{"if_statement":"if_statement","while_statement":"while_statement"}}


def select_parents(population):
    length = range(len(population))
    index_1 = random.choice(length)
    index_2 = random.choice(length)
    chromesome_1 = population[index_1]
    chromesome_2 = population[index_2]
    return chromesome_1, index_1, chromesome_2, index_2

def mutate(chromesome, variable_substitue_dict):
    tgt_index = random.choice(range(len(chromesome)))
    tgt_word = list(chromesome.keys())[tgt_index]
    chromesome[tgt_word] = random.choice(variable_substitue_dict[tgt_word])

    return chromesome

def crossover(csome_1, csome_2, r=None):
    if r is None:
        r = random.choice(range(len(csome_1))) # 随机选择一个位置.
        # 但是不能选到0

    child_1 = {}
    child_2 = {}
    for index, variable_name in enumerate(csome_1.keys()):
        if index < r: #前半段
            child_2[variable_name] = csome_1[variable_name]
            child_1[variable_name] = csome_2[variable_name]
        else:
            child_1[variable_name] = csome_1[variable_name]
            child_2[variable_name] = csome_2[variable_name]
    return child_1, child_2





from keyword import iskeyword
def is_valid_variable_python(name: str) -> bool:
    return name.isidentifier() and not iskeyword(name)

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

def is_valid_variable_name(name: str, lang: str) -> bool:
    # check if matches language keywords
    if lang == 'python':
        return is_valid_variable_python(name)
    elif lang == 'c':
        return is_valid_variable_c(name)
    elif lang == 'java':
        return is_valid_variable_java(name)
    else:
        return False


def is_valid_substitue(substitute: str, tgt_word: str, lang: str) -> bool:
    '''
    判断生成的substitues是否valid,如是否满足命名规范
    '''
    is_valid = True

    if not is_valid_variable_name(substitute, lang):
        is_valid = False

    return is_valid


def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '')
    words = seq.split(' ')

    sub_words = []
    keys = []
    index = 0
    for word in words:
        # 并非直接tokenize这句话，而是tokenize了每个splited words.
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        # 将subwords对齐
        index += len(sub)

    return words, sub_words, keys


def get_identifier_posistions_from_code(words_list: list, variable_names: list) -> dict:
    '''
    给定一串代码，以及variable的变量名，如: a
    返回这串代码中这些变量名对应的位置.
    '''
    positions = {}
    for name in variable_names:
        for index, token in enumerate(words_list):
            if name == token:
                try:
                    positions[name].append(index)
                except:
                    positions[name] = [index]

    return positions


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    '''
    得到substitues
    '''
    # substitutes L, k

    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates 

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes[:24]:  # 去掉不用的计算.
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i
    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to('cuda')
    # 不是，这个总共不会超过24... 那之前生成那么多也没用....
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    '''
    将生成的substitued subwords转化为words
    '''
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        # 比如空格对应的subwords就是[a,a]，长度为0
        return words

    elif sub_len == 1:
        # subwords就是本身
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._decode([int(i)]))
            # 将id转为token.
    else:
        # word被分解成了多个subwords
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    return words


def get_masked_code_by_position2(tokens: list, positions: list):
    
    res = tokens[:]
    for i,v in enumerate(positions):
        res[v] = '<mask>'
    return res



def get_masked_code_by_position(tokens: list, positions: dict):
    
    masked_token_list = {}
    replace_token_positions = {}
    for variable_name in positions.keys():
        for pos in positions[variable_name]:
            #这里需要变一下
            try :
                masked_token_list[variable_name] = masked_token_list[variable_name][0:pos] + ['<unk>'] + masked_token_list[variable_name][pos+1 : ]
                replace_token_positions[variable_name].append(pos)
            except:
                masked_token_list[variable_name] = tokens[0:pos] + ['<unk>'] + tokens[pos + 1:]
                replace_token_positions[variable_name] = [pos]

    
    return masked_token_list, replace_token_positions



def build_vocab(codes, limit=5000):
    
    vocab_cnt = {"<str>": 0, "<char>": 0, "<int>": 0, "<fp>": 0}
    for c in tqdm(codes):
        for t in c:
            if len(t)>0:
                if t[0] == '"' and t[-1] == '"':
                    vocab_cnt["<str>"] += 1
                elif t[0] == "'" and t[-1] == "'":
                    vocab_cnt["<char>"] += 1
                elif t[0] in "0123456789.":
                    if 'e' in t.lower():
                        vocab_cnt["<fp>"] += 1
                    elif '.' in t:
                        if t == '.':
                            if t not in vocab_cnt.keys():
                                vocab_cnt[t] = 0
                            vocab_cnt[t] += 1
                        else:
                            vocab_cnt["<fp>"] += 1
                    else:
                        vocab_cnt["<int>"] += 1
                elif t in vocab_cnt.keys():
                    vocab_cnt[t] += 1
                else:
                    vocab_cnt[t] = 1
    vocab_cnt = sorted(vocab_cnt.items(), key=lambda x:x[1], reverse=True)
    
    idx2txt = ["<unk>"] + ["<pad>"] + [it[0] for index, it in enumerate(vocab_cnt) if index < limit-1]
    txt2idx = {}
    for idx in range(len(idx2txt)):
        txt2idx[idx2txt[idx]] = idx
    return idx2txt, txt2idx



# From MHM codebases


# import pycparser
# import torch

def getTensor(batch, batchfirst=False):
    
    inputs, labels = batch['x'], batch['y']
    inputs, labels = torch.tensor(inputs, dtype=torch.long).cuda(), \
                     torch.tensor(labels, dtype=torch.long).cuda()
    if batchfirst:
        # inputs_pos = [[pos_i + 1 if w_i != 0 else 0 for pos_i, w_i in enumerate(inst)] for inst in inputs]
        # inputs_pos = torch.tensor(inputs_pos, dtype=torch.long).cuda()
        return inputs, labels
    inputs = inputs.permute([1, 0])
    return inputs, labels

__key_words__ = ["auto", "break", "case", "char", "const", "continue",
                 "default", "do", "double", "else", "enum", "extern",
                 "float", "for", "goto", "if", "inline", "int", "long",
                 "register", "restrict", "return", "short", "signed",
                 "sizeof", "static", "struct", "switch", "typedef",
                 "union", "unsigned", "void", "volatile", "while",
                 "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex",
                 "_Generic", "_Imaginary", "_Noreturn", "_Static_assert",
                 "_Thread_local", "__func__"]
__ops__ = ["...", ">>=", "<<=", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=",
           ">>", "<<", "++", "--", "->", "&&", "||", "<=", ">=", "==", "!=", ";",
           "{", "<%", "}", "%>", ",", ":", "=", "(", ")", "[", "<:", "]", ":>",
           ".", "&", "!", "~", "-", "+", "*", "/", "%", "<", ">", "^", "|", "?"]
__macros__ = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX",  # <stdio.h> macro
              "FILENAME_MAX", "L_tmpnam", "SEEK_CUR", "SEEK_END", "SEEK_SET",
              "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]     # <stdlib.h> macro
__special_ids__ = ["main",  # main function
                   "stdio", "cstdio", "stdio.h",                                # <stdio.h> & <cstdio>
                   "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",     # <stdio.h> types & streams
                   "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", # <stdio.h> functions
                   "fopen", "freopen", "setbuf", "setvbuf", "fprintf", "fscanf",
                   "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf",
                   "vscanf", "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets",
                   "fputc", "getc", "getchar", "putc", "putchar", "puts", "ungetc",
                   "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell",
                   "rewind", "clearerr", "feof", "ferror", "perror", "getline"
                   "stdlib", "cstdlib", "stdlib.h",                             # <stdlib.h> & <cstdlib>
                   "size_t", "div_t", "ldiv_t", "lldiv_t",                      # <stdlib.h> types
                   "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold",  # <stdlib.h> functions
                   "strtol", "strtoll", "strtoul", "strtoull", "rand", "srand",
                   "aligned_alloc", "calloc", "malloc", "realloc", "free", "abort",
                   "atexit", "exit", "at_quick_exit", "_Exit", "getenv",
                   "quick_exit", "system", "bsearch", "qsort", "abs", "labs",
                   "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb",
                   "mbstowcs", "wcstombs",
                   "string", "cstring", "string.h",                                 # <string.h> & <cstring>
                   "memcpy", "memmove", "memchr", "memcmp", "memset", "strcat",     # <string.h> functions
                   "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll",
                   "strcpy", "strncpy", "strerror", "strlen", "strspn", "strcspn",
                   "strpbrk" ,"strstr", "strtok", "strxfrm",
                   "memccpy", "mempcpy", "strcat_s", "strcpy_s", "strdup",      # <string.h> extension functions
                   "strerror_r", "strlcat", "strlcpy", "strsignal", "strtok_r",
                   "iostream", "istream", "ostream", "fstream", "sstream",      # <iostream> family
                   "iomanip", "iosfwd",
                   "ios", "wios", "streamoff", "streampos", "wstreampos",       # <iostream> types
                   "streamsize", "cout", "cerr", "clog", "cin",
                   "boolalpha", "noboolalpha", "skipws", "noskipws", "showbase",    # <iostream> manipulators
                   "noshowbase", "showpoint", "noshowpoint", "showpos",
                   "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase",
                   "left", "right", "internal", "dec", "oct", "hex", "fixed",
                   "scientific", "hexfloat", "defaultfloat", "width", "fill",
                   "precision", "endl", "ends", "flush", "ws", "showpoint",
                   "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",    # <math.h> functions
                   "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf",
                   "ceil", "floor", "abs", "fabs", "cabs", "frexp", "ldexp",
                   "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]
                   
__parser__ = None

def tokens2seq(_tokens):
    
    '''
    Return the source code, given the token sequence.
    '''
    
    seq = ""
    for t in _tokens:
        if t == "<INT>":
            seq += "0 "
        elif t == "<FP>":
            seq += "0. "
        elif t == "<STR>":
            seq += "\"\" "
        elif t == "<CHAR>":
            seq += "'\0' "
        else:
            while "<__SPACE__>" in t:
                t.replace("<__SPACE__>", " ")
            while "<__BSLASH_N__>" in t:
                t.replace("<__BSLASH_N__>", "\n")
            while "<__BSLASH_R__>" in t:
                t.replace("<__BSLASH_R__>", "\r")
            seq += t + " "
    return seq

def getAST(_seq=""):
    
    '''
    Return the AST of a c/c++ file.
    '''
    
    global __parser__
    if __parser__ is None:
        __parser__ = pycparser.CParser()
    _ast = __parser__.parse(_seq)
    return _ast
    
def getDecl(_seq="", _syms={}):
    
    '''
    Return all declaration names in an AST.
    '''
    
    _node = getAST(_seq)
    if isinstance(_node, pycparser.c_ast.Decl):
        if isinstance(_node.children()[0][1], pycparser.c_ast.TypeDecl):
            _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.PtrDecl):
            _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.ArrayDecl):
            _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.FuncDecl):
            _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.Struct):
            _syms.add(_node.children()[0][1].name)
            if not _node.name is None:
                _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.Union):
            _syms.add(_node.children()[0][1].name)
            if not _node.name is None:
                _syms.add(_node.name)
    try:
        for _child in _node.children():
            _syms = getDecl(_child[1], _syms)
    except:
        _node.show()
    return _syms
    
def isUID(_text=""):
    
    '''
    Return if a token is a UID.
    '''
    
    _text = _text.strip()
    if _text == '':
        return False

    if " " in _text or "\n" in _text or "\r" in _text:
        return False
    elif _text in __key_words__:
        return False
    elif _text in __ops__:
        return False
    elif _text in __macros__:
        return False
    elif _text in __special_ids__:
        return False
    elif _text[0].lower() in "0123456789":
        return False
    elif "'" in _text or '"' in _text:
        return False
    elif _text[0].lower() in "abcdefghijklmnopqrstuvwxyz_":
        for _c in _text[1:-1]:
            if _c.lower() not in "0123456789abcdefghijklmnopqrstuvwxyz_":
                return False
    else:
        return False
    return True
    
def getUID(_tokens=[], uids=[]):
    
    '''
    Return all UIDs and their indeces, given a token sequence.
    '''
    
    ids = {}
    for i, t in enumerate(_tokens):
        if isUID(t) and t in uids[0].keys():
            if t in ids.keys():
                ids[t].append(i)
            else:
                ids[t] = [i]
    return ids
    


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



class Recorder():
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.f = open(file_path, 'w')
        self.writer = csv.writer(self.f)
        self.writer.writerow(["Index",
                        "Original Code", 
                        "Program Length", 
                        "Adversarial Code", 
                        "True Label", 
                        "Original Prediction", 
                        "Adv Prediction", 
                        "Is Success", 
                        "Extracted Names",
                        "Importance Score",
                        "Beam search No. Changed Names",
                        "Beam search No. Changed Tokens",
                        "Beam search No. Insert stmts",
                        "Replaced Names",
                        "Query Times",
                        "Time Cost",
                        "Suc Type",
                        "Insert Words",
                        "const transfer record",
                        "orig prob",
                        "current prob",
                        "Attack paths",
                        "Attack code"])
    
    def write(self, index, code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, score_info, nb_changed_var, nb_changed_pos,nb_insert_stmts,replace_info, query_times, time_cost,suctype,insertwords,const_transfer,orig_prob,curr_prob,attack_paths,attack_code):
        self.writer.writerow([index,
                        code, 
                        prog_length, 
                        adv_code, 
                        true_label, 
                        orig_label, 
                        temp_label, 
                        is_success, 
                        ",".join(variable_names),
                        score_info,
                        nb_changed_var,
                        nb_changed_pos,
                        nb_insert_stmts,
                        replace_info,
                        query_times,
                        time_cost,
                        suctype,
                        insertwords,
                        const_transfer,
                        orig_prob,
                        curr_prob,
                        attack_paths,
                        attack_code])


class Defend_recoder:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.f = open(file_path, 'w')
        self.writer = csv.writer(self.f)
        self.writer.writerow(["source code","orig prob","orig label","adversarial code","adversarial prob","adversarial label","constant propogation code","constant propogation prob","constant propogation label","delete dead code","delete dead prob","delete dead label"\
            ,"outlier code","outlier prob","outlier label","normalized code","normalized prob","normalized label","llm helpd code","llm helped prob","llm helpd label","open llm helped code","open llm helped prob","open llm helped label"])
    
    def write(self, source_code,orig_prob,orig_label,adversarial_code,adversarial_prob,adversarial_label,\
        constant_propogation_code,constant_propogation_prob,constant_propogation_label,delete_dead_code,\
            delete_dead_prob,delete_dead_label,outlier_code,outlier_prob,outlier_label,normalized_code,normalized_prob,normalized_label,llm_helpd_code,llm_helped_prob,llm_helpd_label,open_llm_helped_code,open_llm_helped_prob,open_llm_helped_label):
        self.writer.writerow([source_code,orig_prob,orig_label,adversarial_code,adversarial_prob,adversarial_label,\
            constant_propogation_code,constant_propogation_prob,constant_propogation_label,delete_dead_code,\
                delete_dead_prob,delete_dead_label,outlier_code,outlier_prob,outlier_label,normalized_code,normalized_prob,normalized_label,llm_helpd_code,llm_helped_prob,llm_helpd_label,\
                    open_llm_helped_code,open_llm_helped_prob,open_llm_helped_label])

#