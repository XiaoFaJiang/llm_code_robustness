import requests
import json
import time
import os

class LocalClient():

    def __init__(self):
        pass

    def generate(self, ip, prompt, wait=False, **gen_kwargs):
        url = os.environ.get("OPENAI_BASE_URL",f"http://{ip}:8800/v1")
        model_id = os.environ.get("MODEL_ID","Qwen2.5-Coder-7B-Instruct")
        if "chat" in url:
            messages = [{"role":"user","content":prompt}]
            data = {
                "model": model_id,
                "messages":messages,
                "max_tokens":gen_kwargs["max_length"],
            }
        else:
            data = {
                "model": model_id,
                "prompt": prompt,
                "max_tokens": gen_kwargs["max_length"],
                "temperature": gen_kwargs["temperature"],
                "top_p": gen_kwargs["top_p"],
                # "do_sample": gen_kwargs["do_sample"],
            }
        payload = json.dumps(data)
        headers = {
            'Content-Type': 'application/json'
        }
        cnt = 0
        total = 1
        if wait:
            total = 20
        while True:
            try:
                response = requests.request("POST", url, headers=headers, data=payload)
                response = response.json()
                #print(f"Server response: {response}")
                if "chat" in url:
                    return response['choices'][0]['message']['content']
                else:
                    return response['choices'][0]['text']
            except Exception as e:
                cnt = cnt + 1
                if cnt < total:
                    print(f"服务器ip: {ip}未返回结果或异常，若这是第一条请求，可能是服务还没启动成功等待1分钟后重试... 已重试次数:{cnt}/{total}", flush=True)
                    print(response)
                    time.sleep(5)  # 等待60秒
                    continue
                raise e