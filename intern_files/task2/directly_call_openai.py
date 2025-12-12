import json
import random
import requests
from requests import Response
import concurrent.futures
from tqdm import tqdm


url = 'https://aigc.sankuai.com/v1/openai/native/chat/completions'
tokens = [
    "1790715889671905303"
]

MODEL = 'gpt-4-turbo-eva'


class Resp(object):
    def __init__(self, data=None, code=0, message=None):
        self.data = data
        self.code = code
        self.message = message

    def is_success(self):
        return self.code == 0


def call(content: str, temperature, max_token) -> Resp:
    """
    调用openai api的直接方法
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token()}"
    }
    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": temperature,
        "max_tokens": max_token
    }

    data = json.dumps(data)

    try:
        raw_resp = requests \
            .post(url=url, json=None, headers=headers, data=data, timeout=240)
    except Exception as e:
        return Resp(message=str(e), code=-1)

    return process_success_resp(raw_resp) if raw_resp.status_code == 200 \
        else handle_fail_resp(raw_resp)

                 
def run_call(data, out_fp, temperture, max_token):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token()}"
    }
    failure_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for question in data:
            future = executor.submit(
                call,
                question,
                None,
                64,
                out_fp
            )
            futures.append(future)
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            try:
                future.result()
            except:
                failure_list.append(future)
        print("done")
    print(f"error num {len(failure_list)}")
    
                 

def token():
    """
    随机选择token
    """
    random.shuffle(tokens)
    return tokens[0]


def handle_fail_resp(raw_resp: Response) -> Resp:
    """
    处理失败的resp
    """
    try:
        resp = raw_resp.json()
        message = json.dumps(resp['error'])
    except Exception:
        message = raw_resp.content if raw_resp and raw_resp.content \
            else "resp is None"
    return Resp(message=message, code=-1)


def process_success_resp(raw_resp: Response) -> Resp:
    """
    处理成功的resp
    """
    try:
        resp = raw_resp.json()
        choices = resp['choices']
        if not choices:
            return Resp("empty", -1)
        data = [choice['message']['content'].strip() for choice in choices]
        return Resp(data=data)
    except Exception as e:
        return Resp(message=e, code=-1)


if __name__ == "__main__":
    sent =['1+1=?', '2+1=?']
    res = call("1+1=", 0, 5)
    # print(res.data, res.code, res.message)