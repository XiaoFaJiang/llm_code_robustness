import json
import os
import time
import openai
from abc import ABC, abstractmethod

import requests


class Responser(ABC):

    @abstractmethod
    def respond(self, system_info: str, user_prompt: str) -> str:
        pass


class GPT4Responser(Responser):
    """ Openai LLM responser """

    def __init__(self, model='gpt-4'):
        """ environment information """
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_base = os.environ.get("OPENAI_API_BASE")
        openai.api_type = 'azure'
        openai.api_version = '2023-07-01-preview'
        self.model = model

    def respond(self, system_info: str, user_prompt: str) -> str:
        """
        respond to system_info and user prompt
        :param system_info: see in openai documentation
        :param user_prompt: see in openai documentation
        :return: response in form of string
        """
        try:
            response = openai.ChatCompletion.create(
                engine=self.model,
                messages=[
                    {"role": "system", "content": system_info},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2000,
                stop=None,
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"{e}\nRate Limit Reached! Sleeping for 20 secs...")
            time.sleep(20)
            response = openai.ChatCompletion.create(
                engine=self.model,
                messages=[
                    {"role": "system", "content": system_info},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=2000,
                stop=None,
            )
            return response['choices'][0]['message']['content']


class TurboResponser(Responser):
    """ Openai LLM responser """

    def __init__(self, model='gpt-3.5-turbo'):
        """ environment information """
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        openai.api_base = os.environ.get("OPENAI_API_BASE")

    def respond(self, system_info: str, user_prompt: str) -> str:
        """
        respond to system_info and user prompt
        :param system_info: see in openai documentation
        :param user_prompt: see in openai documentation
        :return: response in form of string
        """
        messages = [
            {"role": "system", "content": system_info},
            {"role": "user", "content": user_prompt}
        ]
        response = openai.ChatCompletion.create(
            # model='gpt-4',
            model='gpt-3.5-turbo',
            # model='gpt-4-32k',
            # model='gpt-3.5-turbo-16k',
            messages=messages
        )
        return response['choices'][0]['message']['content']

class HttpApiResponser(Responser):
    def __init__(self, ip):
        self.ip = ip

    def respond(self, system_info: str, user_prompt: str) -> str:
        url = f"http://{self.ip}:8080"

        prompt = system_info + user_prompt
        max_new_tokens = 2000
        # 基础模型
        data = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        }
        headers = {
            'Content-Type': 'application/json'
        }
        proxies = {
            "http": "",
            "https": "",
        }
        cnt = 0
        total = 10
        while True:
            try:
                print(f"|| request: {data}",flush=True)
                response = requests.request("POST", url, headers=headers, json=data, proxies=proxies)
                response = response.json()
                print(f"|| response: {response}", flush=True)
                if "completions" in response:
                    return response['completions'][0]['text']
                else:
                    return response["result"]
            except Exception as e:
                cnt = cnt + 1
                if cnt < total:
                    print(f"请求异常: Exception: {e}")
                    print(f"服务器ip: {self.ip}未返回结果或异常，若这是第一条请求，可能是服务还没启动成功等待1分钟后重试..."
                          f" 已重试次数:{cnt}/{total}", flush=True)
                    time.sleep(60)  # 等待60秒
                    continue
                raise e


if __name__ == '__main__':
    # gpt4_responser = GPT4Responser()
    turbo_responser = TurboResponser()
    print(turbo_responser.respond(system_info="Translate the text into English",
                                  user_prompt=f"Elle a dit: \"Je suis une fille\""))
