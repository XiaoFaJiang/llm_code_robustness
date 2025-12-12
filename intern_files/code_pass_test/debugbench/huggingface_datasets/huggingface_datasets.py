import json
from datasets import load_from_disk


paratype_df = json.load(open("../humaneval_paratype.json"))
no_paratype_df = json.load(open("../humaneval_no_paratype.json"))
no_example_df = json.load(open("../humaneval_no_example.json"))
instruction_no_paratype_df = json.load(open("../humaneval_instruction_no_paratype.json"))
instruction_paratype_df = json.load(open("../humaneval_instruction_paratype.json"))

paratype_dict, no_paratype_dict, no_example_dict, instruction_no_paratype_dict, instruction_paratype_dict = {}, {}, {}, {}, {}
for d in paratype_df:
    paratype_dict[d["task_id"]] = d["prompt"]
for d in no_paratype_df:
    no_paratype_dict[d["task_id"]] = d["prompt"]
for d in no_example_df:
    no_example_dict[d["task_id"]] = d["prompt"]

for d in instruction_no_paratype_df:
    instruction_no_paratype_dict[d["task_id"]] = d["instruction"]
for d in instruction_paratype_df:
    instruction_paratype_dict[d["task_id"]] = d["instruction"]


instruct_humaneval = load_from_disk("humaneval/instructhumaneval")
def fix_paratype(d):
    d["prompt_paratype"] = paratype_dict[d["task_id"]]
    d["prompt_noparatype"] = no_paratype_dict[d["task_id"]]
    d["prompt_noexample"] = no_example_dict[d["task_id"]]
    d["instruction_noparatype"] = instruction_no_paratype_dict[d["task_id"]]
    d["instruction_paratype"] = instruction_paratype_dict[d["task_id"]]
    return d

instruct_humaneval = instruct_humaneval.map(fix_paratype)
instruct_humaneval.save_to_disk("humaneval/instructhumaneval_paratype")

