#coding=utf-8
from transformers import AutoModel, AutoTokenizer

# 模型名称（Hugging Face 模型库中的名称）
model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"

# 下载模型和分词器到本地
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 保存模型和分词器到本地目录
save_directory = "./Qwen2.5-Coder-3B-Instruct"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)

print(f"Model and tokenizer saved to {save_directory}")