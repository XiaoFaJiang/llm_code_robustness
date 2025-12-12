import json
with open("/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/zhaoyicong03/datatranslate/test/postsample.json", 'r') as f2:
    references = json.load(f2)


result_list = [list(item.values())[0] for item in references]
result_list = [item.split("class Solution ")[0] for item in result_list]


res = []
for idx, item in enumerate(result_list):
    l = {f"{idx}":f"{item}"}
    res.append(l)

with open("/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/zhaoyicong03/1.json", 'w') as f:
    json.dump(res,f,ensure_ascii=False)