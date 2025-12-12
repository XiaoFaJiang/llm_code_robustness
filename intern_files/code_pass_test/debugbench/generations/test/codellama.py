from transformers import AutoTokenizer
import transformers
import torch
import json

model = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/model/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

ans = []


with open("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhaoyicong03/prompt/promptpython3.json", 'r' ) as file:
    data = json.load(file)
    #for item in data:
    item = data[1]


    sequences = pipeline(
        '''import socket\n\ndef ping_exponential_backoff(host: str):''',
        
        #item + "\n" + '<code>',
        do_sample=True,
        top_k=10,
        temperature=0.1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=100,
    )
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

