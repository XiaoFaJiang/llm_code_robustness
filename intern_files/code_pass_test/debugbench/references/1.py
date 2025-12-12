import json
with open("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhaoyicong03/bigcode-evaluation-harness/zhaoyicong1/debugbench-cpp/generations.json", 'r') as indices:
    ind = json.load(indices)

print(ind[8][0])
