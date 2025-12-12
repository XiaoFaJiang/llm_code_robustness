#coding=utf-8

common_built_in_functions_of_python = [
    # 构造和初始化
    "__init__",
    "__new__",
    "__del__",

    # 操作符重载
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__pow__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",

    # 容器类型模拟
    "__len__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__iter__",
    "__reversed__",
    "__contains__",

    # 描述符
    "__get__",
    "__set__",
    "__delete__",

    # 其他
    "__call__",
    "__str__",
    "__repr__",
    "__hash__",
    "__bool__",
    'print',
    'len',
    'type',
    'int',
    'str',
    'float',
    'list',
    'tuple',
    'set',
    'dict',
    'range',
    'input',
    'sorted',
    'max',
    'min',
    'sum',
    'abs',
    'round',
    'zip',
    'enumerate',
    'open',
    "findall",
    "append",
    "key",
    "re.sub",
    "re.findall",
    "re.match",
    'str.capitalize', 'str.casefold', 'str.center', 'str.count', 'str.encode',
    'str.endswith', 'str.expandtabs', 'str.find', 'str.format', 'str.format_map',
    'str.index', 'str.isalnum', 'str.isalpha', 'str.isascii', 'str.isdecimal',
    'str.isdigit', 'str.isidentifier', 'str.islower', 'str.isnumeric', 'str.isprintable',
    'str.isspace', 'str.istitle', 'str.isupper', 'str.join', 'str.ljust',
    'str.lower', 'str.lstrip', 'str.maketrans', 'str.partition', 'str.replace',
    'str.rfind', 'str.rindex', 'str.rjust', 'str.rpartition', 'str.rsplit',
    'str.rstrip', 'str.split', 'str.splitlines', 'str.startswith', 'str.strip',
    'str.swapcase', 'str.title', 'str.translate', 'str.upper', 'str.zfill',
    're.compile', 're.search', 're.match', 're.fullmatch', 're.split',
    're.findall', 're.finditer', 're.sub', 're.subn', 're.escape',
    're.error',
    'list.append', 'list.clear', 'list.copy', 'list.count', 'list.extend',
    'list.index', 'list.insert', 'list.pop', 'list.remove', 'list.reverse',
    'list.sort',
    'set.add', 'set.clear', 'set.copy', 'set.difference', 'set.difference_update',
    'set.discard', 'set.intersection', 'set.intersection_update', 'set.isdisjoint',
    'set.issubset', 'set.issuperset', 'set.pop', 'set.remove', 'set.symmetric_difference',
    'set.symmetric_difference_update', 'set.union', 'set.update',
    'dict.clear', 'dict.copy', 'dict.fromkeys', 'dict.get', 'dict.items',
    'dict.keys', 'dict.pop', 'dict.popitem', 'dict.setdefault', 'dict.update',
    'dict.values'

]

java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "do", "double", "else", "enum",
                 "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
                 "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
                 "short", "static", "strictfp", "super", "switch", "throws", "transient", "try", "void", "volatile",
                 "while","compare"]

java_special_ids = ["main", "args", "Math", "System", "Random", "Byte", "Short", "Integer", "Long", "Float", "Double", "Character",
                    "Boolean", "Data", "ParseException", "SimpleDateFormat", "Calendar", "Object", "String", "StringBuffer",
                    "StringBuilder", "DateFormat", "Collection", "List", "Map", "Set", "Queue", "ArrayList", "HashSet", "HashMap","equals","hashCode",
                    "deepEquals","compareTo","true","false"]

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
                   "modf", "fmod", "hypot", "ldexp", "poly", "matherr","sort","length","size","std","using","vector","begin","to_string","to_lower",
                   "first", "empty", "M_PI","push_back","substr","end","second","front","stoi","rbegin","rend","type","set"
                   ,"INT_MAX","bitset","tie","accumulate","istringstream","static_assert","top","pair","set_intersection",
                   "toupper","insert","map","reverse","make_pair","stack","get","tuple","gcvt","isspace","greater",
                   "min","max","erase","INT_MIN","pop","getline","min_element","partial_sort_copy","unordered_map",
                   "push","isalnum","max_element","emplace_back","Holder","template","this","hasher","true","false","regex","count","operator","make_tuple"]

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
    "parseFloat", "parseInt", "decodeURI", "decodeURIComponent", "encodeURI", "encodeURIComponent","sin","cos","acos","tan","PI"
]

commutative_operators = [
    '+',  # 加法
    '*',  # 乘法
    '==', # 等于（在比较相同类型的值时）
    '!=', # 不等于（在比较相同类型的值时）
    '&&', # 逻辑与（在布尔值之间）
    '||', # 逻辑或（在布尔值之间）
    '&',  # 按位与（在整数之间）
    '|',  # 按位或（在整数之间）
    '^'   # 按位异或（在整数之间）
]



base_module = {
    'python': [
        'os',
        'sys',
        'math',
        'datetime',
        'json',
        're',  # Regular expressions
        'collections',
        'itertools',
        'random',
        'subprocess',
        'io',  # Input/Output
        'csv',
        'unittest',  # Testing
        'threading',  # Multithreading
        'multiprocessing',  # Multiprocessing
        'socket',  # Networking
        'http',  # HTTP client/server
        'urllib',  # URL handling
        'argparse',  # Command-line parsing
        'logging',  # Logging
        'pickle',  # Object serialization
    ],
    'cpp': [
        # C++标准库模块通常通过头文件引入，以下是一些常见的头文件
        '<iostream>',
        '<vector>',
        '<map>',
        '<set>',
        '<queue>',
        '<stack>',
        '<algorithm>',
        '<string>',
        '<fstream>',
        '<sstream>',
        '<iomanip>',
        '<cmath>',
        '<cstdlib>',
        '<ctime>',
        '<cstdio>',
        '<memory>',
        '<thread>',
        '<mutex>',
        '<condition_variable>',
        '<atomic>',
    ],
    'java': [
        'java.lang.*',
        'java.util.*',
        'java.io.*',
        'java.nio.*',
        'java.math.*',
        'java.time.*',
        'java.sql.*',
        'java.net.*',
        'javax.swing.*',
        'java.security.*',
        'java.text.*',
        'java.rmi.*',
        'java.beans.*',
        'java.applet.*',
    ],
    'javascript': [
    'fs',  # Node.js文件系统模块
    'path',  # Node.js路径模块
    'http',  # Node.js HTTP模块
    'https',  # Node.js HTTPS模块
    'os',  # Node.js操作系统模块
    'url',  # Node.js URL模块
    'querystring',  # Node.js查询字符串模块
    'events',  # Node.js事件模块
    'stream',  # Node.js流模块
    'zlib',  # Node.js压缩模块
    'crypto',  # Node.js加密模块
    'util',  # Node.js实用工具模块
    'assert',  # Node.js断言模块
    'buffer',  # Node.js缓冲区模块
    'child_process',  # Node.js子进程模块
    'cluster',  # Node.js集群模块
    'dgram',  # Node.js数据报模块
    'dns',  # Node.jsDNS模块
    'net',  # Node.js网络模块
    'readline',  # Node.js读取行模块
]
}


all_languages_constant_variable_types = {
    "cpp": {
        "char_literal": ["const char"],
        "number_literal": ["const auto"],  # 数字字面量默认为int，但double为更通用类型
        "string_literal": ["const std::string"],
    },
    "java": {
        "decimal_integer_literal": ["final int", "long"],
        "hex_integer_literal": ["final int", "long"],
        "character_literal": ["final char"],
        "octal_integer_literal": ["final int", "long"],
        "decimal_floating_point_literal": ["final double", "float"],
        "binary_integer_literal": ["final int", "long"],
        "string_literal": ["final String"],
        "hex_floating_point_literal": ["final double", "float"],
        "false": ["final boolean"],
        "true": ["final boolean"],
    }
}

python_literal = ['false','true','float','integer','none','string']
cpp_literal = ['string_literal','char_literal','number_literal']
java_literal = [ "decimal_integer_literal", "hex_integer_literal", "character_literal", "octal_integer_literal",\
                 "decimal_floating_point_literal", "binary_integer_literal","string_literal","hex_floating_point_literal",\
                     "false","true"]
javascript_literal = ["string","number"]

literals = {'python':python_literal,'cpp':cpp_literal,'java':java_literal,'javascript':javascript_literal}

python_tree_sitter_simple_statement = [
      {
        "type": "assert_statement",
        "named": True
      },
      {
        "type": "break_statement",
        "named": True
      },
      {
        "type": "continue_statement",
        "named": True
      },
      {
        "type": "delete_statement",
        "named": True
      },
      {
        "type": "exec_statement",
        "named": True
      },
      {
        "type": "expression_statement",
        "named": True
      },
      {
        "type": "future_import_statement",
        "named": True
      },
      {
        "type": "global_statement",
        "named": True
      },
      {
        "type": "import_from_statement",
        "named": True
      },
      {
        "type": "import_statement",
        "named": True
      },
      {
        "type": "nonlocal_statement",
        "named": True
      },
      {
        "type": "pass_statement",
        "named": True
      },
      {
        "type": "print_statement",
        "named": True
      },
      {
        "type": "raise_statement",
        "named": True
      },
      {
        "type": "return_statement",
        "named": True
      },
      {
        "type": "type_alias_statement",
        "named": True
      }
    ]

python_complex_treesitter_statement = [
      {
        "type": "class_definition",
        "named": True
      },
      {
        "type": "decorated_definition",
        "named": True
      },
      {
        "type": "for_statement",
        "named": True
      },
      {
        "type": "function_definition",
        "named": True
      },
      {
        "type": "if_statement",
        "named": True
      },
      {
        "type": "match_statement",
        "named": True
      },
      {
        "type": "try_statement",
        "named": True
      },
      {
        "type": "while_statement",
        "named": True
      },
      {
        "type": "with_statement",
        "named": True
      }
    ]