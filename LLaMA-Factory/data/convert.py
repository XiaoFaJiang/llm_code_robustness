import json
import re

def convert_dpo_data(input_file, output_file):
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item['code_str_generate']
            lang = prompt.split("progame language:")[1].split("\n")[0]
            # 构建标准 ShareGPT 格式的 DPO 数据
            chosen = f"""
```{lang}
{item["adv_truth"]}
```
"""

            rejected = f"""
```{lang}
{item["adv_rejected"]}
```
"""

            new_item = {
                "conversations": [
                    {
                        "from": "human",
                        "value": prompt
                    }
                ],
                "chosen": {
                    "from": "gpt",
                    "value": chosen
                },
                "rejected": {
                    "from": "gpt",
                    "value": rejected
                }
            }
            converted_data.append(new_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 请替换为你的实际文件名
    convert_dpo_data("train_dpo_base.jsonl", "train_dpo_standard_base.json")
    print("转换完成！生成了 train_dpo_standard.json")
    convert_dpo_data("valid_dpo_base.jsonl", "valid_dpo_standard_base.json")
    print("转换完成！生成了 valid_dpo_standard.json")