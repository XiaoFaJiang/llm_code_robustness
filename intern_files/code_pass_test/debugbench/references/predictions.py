import re
import os
import json


list = []
s="""
import java.util.*;
import java.lang.*;
"""
with open("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhaoyicong03/dataset/flitered_java.json",'r') as file:
    data = json.load(file)
    for item in data:
        l = item["oracle_code"]
        l = s + l
        list.append(l)
    
    
filename = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhaoyicong03/references/java_pre.json'

# 使用 'with' 语句来打开文件，确保它会被正确地关闭
with open(filename, 'w') as file:
    # 使用 json.dump 将数据写入文件
    json.dump(list, file)

print(len(list))