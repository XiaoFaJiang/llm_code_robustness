# from datasets import load_dataset, load_from_disk

# dataset = load_dataset("codeparrot/instructhumaneval")
# dataset.save_to_disk("instructhumaneval")

import json

from datasets import load_from_disk

data = load_from_disk("instructhumaneval")
res = {}
for k,_ in data.items():
    v = []
    for d in data[k]:
        v.append(d)
    res[k] = v
with open("instructhumaneval/task.json", "w") as f:
    json.dump(res, f, indent=4, ensure_ascii=False)

data = data.shuffle(seed=0)

difficulty = json.load(open("divide_diff_baseon_num.json"))
id_dict = {}
for i,d in enumerate(data["test"]):
    id_dict[i] = d["task_id"]

difficulty_num = {}
for k,v in difficulty.items():
    difficulty_num[k] = []
    for idx in v:
        difficulty_num[k].append(id_dict[idx])
    difficulty_num[k] = sorted(difficulty_num[k], key=lambda x: int(x.split("/")[1]))

with open("difficulty_num.json", "w") as f:
    json.dump(difficulty_num, f, indent=4, ensure_ascii=False)