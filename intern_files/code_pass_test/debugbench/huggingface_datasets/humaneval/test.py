import os
from datasets import load_from_disk
from code_eval import compute_code_eval

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
data = load_from_disk("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/llm-eval/benchmark/bigcode/instructhumaneval_paratype")["test"]


test_func = data["test"]
references = ["\n" + test_func[i] + "\n" + f"check({data['entry_point'][i]})" for i in range(len(test_func))]
predictions = [[data["prompt_noparatype"][i] + data["canonical_solution"][i]] for i in range(len(test_func))]
print(predictions[0])
print(references[0])

# print("------")
# pass_at_k_n, cases = compute_code_eval(predictions=predictions, references=references)
# print(pass_at_k_n)

# for _,v in cases.items():
#     for k,stats in v[-1][-1].items():
#         if k == "passed":
#             if not stats:
#                 print(v)