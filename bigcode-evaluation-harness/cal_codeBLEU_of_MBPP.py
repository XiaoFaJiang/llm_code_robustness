import sys
sys.path.append('../adv_samples_gen')
from CodeBLEU.calc_code_bleu import compute_metrics
from transformers import AutoTokenizer
import json
import heapq

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("/data1/model/qwen/Qwen/Qwen2.5-Coder-0.5B-Instruct")
    LANGAUGES = ['cpp','java','javascript']
    for lang in LANGAUGES:
        dataset = json.load(open(f'dataset/mbpp_{lang}_tested.json'))
        n = len(dataset)
        all_code_str_deleted = []
        for i,v in enumerate(dataset):
            all_code_str_deleted.append(v['code_str_deleted'])
        codebleu = [[0 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i,n):
                cb = compute_metrics((tokenizer(all_code_str_deleted[i],return_tensors="pt").input_ids,tokenizer(all_code_str_deleted[j], return_tensors="pt").input_ids),tokenizer,lang)
                codebleu[i][j] = cb['CodeBLEU']
            codebleu[i][i] = -1
        #codebleu[i][j] (i<j)
        best_indexs = []
        for i in range(n):
            arr = codebleu[i]
            top5_indices = heapq.nlargest(5, range(len(arr)), key=lambda i: arr[i])
            best_indexs.append(top5_indices)
        json.dump(best_indexs,open(f"dataset/codebleu_similarity/{lang}.json","w"),indent=4)
    