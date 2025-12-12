# encoding=utf-8
import json
import time
import openai
import logging
import requests

from typing import Any, Dict, List
from bigcode_eval.api_encrypt import auto_decrypt
from bigcode_eval.chatglm3_130b_client import ChatGLM3130BClient, ChatGLM3130BChatClient
from bigcode_eval.ft_client import FTClient
from bigcode_eval.local_client import LocalClient
from bigcode_eval.ratelimit import sleep_and_retry, peak_rate_limit

# from helm.common.cache import Cache, CacheConfig
# from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
# from .baichuan2_53b_client import Baichuan253BClient
# from .chatglm2_66b_client import ChatGLM266BClient
# from .chatglm_pro_client import ChatGLMProClient
# from .client import Client, wrap_request_time, truncate_sequence

# from helm.common.global_args import GlobalArgs
# from .ernie_client import ErnieClient
# from .local_http_client import LocalHttpClient
# from .spark_api_client import SparkClient
# from ...common.hierarchical_logger import print
# from ...common.ratelimit import sleep_and_retry, peak_rate_limit

# from ...common.tokenization_request import DecodeRequest, DecodeRequestResult, TokenizationRequest, \
#     TokenizationRequestResult
# import openai
# from pycat import Cat

# from helm.proxy.clients.ft_client import FTClient

"""
通过http方式调用模型，请求poker或直接请求openAI的接口
"""

class HttpAPIServer:
    def __init__(self, model_name: str, global_args: Dict):
        try:
            self.model_name = model_name.split("/")[1]
            self.global_args = global_args
            # self.appId = "1656975655663071280"
            self.appId = auto_decrypt("!!!机密信息，越权解密将被追责!!!ENCRYPTED_A1V1<<<6ZNgj6K5AyH2z8VcGYCYoUvmmxS8ZovK>>>")
            # self.glm = None
            # self.glm6b = None
            # self.ft = None
        except Exception as e:
            logging.exception(e)
            raise e

    # def init_glm(self):
    #     from .glm_client import GlmClient
    #     self.glm = GlmClient()
    #     # from .glm_6b_client import Glm6BClient
    #     # self.glm6b = Glm6BClient()

    # @staticmethod
    # def model_list() -> [str]:
    #     scales = [5, 20, 50, 200, 1000000]
    #     models = ["OpenAI/ChatGPT", "OpenAI/gpt4"]
    #     return [f"{model}[[EDS={scale}]]" for model in models for scale in scales]

    @staticmethod
    def name_match_list():
        return [
            "openai/chatgpt",
            "openai/gpt4",
            "openai/gpt-4",
            "chatglm3-130b",
        ]

    @staticmethod
    def use_api_client(model_name: str, global_args: Dict = None) -> bool:
        # if model_name in HttpAPIServer.model_list():
        #     return True
        for model in HttpAPIServer.name_match_list():
            if model.lower() in model_name.lower():
                return True
        if "api" in global_args and global_args["api"]:
            return True
        if "chat_mode" in global_args and global_args["chat_mode"] and global_args["chat_mode"] != "chat":
            return True
        if "customize_inference_ip" in global_args and global_args["customize_inference_ip"]:
            return True
        return False

    """
    提交请求 根据模型走不同的方式
    """
    def serve_request(self, raw_request: Dict[str, Any], global_args, **gen_kwargs):
        prompt = raw_request["prompt"]
        # if "prompt_prefix" in global_args:
        #     prompt = global_args["prompt_prefix"] + prompt
        # if "prompt_suffix" in global_args:
        #     prompt = prompt + global_args["prompt_suffix"]
        raw_request["prompt"] = prompt
        # # 设置EOS配置
        # if "eos" in global_args:
        #     raw_request["stop_sequences"] = global_args['eos']

        if "api" in global_args and global_args["api"]:
            # if global_args["api"] == "cfs":
            #     return self.server_by_common_api(raw_request, global_args)
            if global_args["api"] == "ft":
                return self.server_by_ft_api(raw_request, global_args, **gen_kwargs)
        # elif "function_call" in global_args:
        #     if global_args["function_call"] == 'gpt3':
        #         return self.server_by_gpt_functional(raw_request, global_args, "gpt-3.5-turbo-16k-0613")
        #     else:
        #         return self.server_by_gpt_functional(raw_request, global_args, "gpt-4-0613")
        # elif "use_local_inference_file" in global_args:
        #     return self.server_by_local_inference_file(raw_request, global_args)
        elif "chat_mode" in global_args and global_args["chat_mode"]:
            ip = global_args["chat_mode"]
            return self.server_by_local_server(ip, raw_request, global_args, **gen_kwargs)
        elif "customize_inference_ip" in global_args and global_args["customize_inference_ip"]:
            ip = global_args["customize_inference_ip"]
            return self.server_by_local_server(ip, raw_request, global_args, wait=True, **gen_kwargs)
        elif "chatgpt" in self.model_name.lower():
            return self.server_by_chatgpt(raw_request, global_args, **gen_kwargs)
        elif "gpt4" in self.model_name.lower() or "gpt-4" in self.model_name.lower():
            return self.server_by_gpt4(raw_request, global_args, **gen_kwargs)
        elif "chatglm3-130b" in self.model_name.lower():
            return self.server_by_chatglm3_130b(raw_request, global_args)
        # elif "text-davinci-003" in self.model_name.lower() or \
        #         "text-ada-001" in self.model_name.lower() or \
        #         "gpt-3.5-turbo-instruct" in self.model_name.lower():
        #     return self.server_by_openai(raw_request)
        # elif "gpt-3.5-turbo-0613" in self.model_name.lower() or\
        #         "gpt-3.5-turbo-16k-0613" in self.model_name.lower():
        #     return self.server_by_openai_chat(raw_request)
        # elif "minimax" in self.model_name:
        #     return self.server_by_minimax(raw_request, global_args)
        # elif "chatglm2-66b" in self.model_name.lower():
        #     return self.server_by_chatglm2_66b(raw_request, global_args)
        # elif "baichuan2-53b" in self.model_name.lower() or "baichuan53b-api-v2" in self.model_name.lower():
        #     return self.server_by_baichuan2_53b(raw_request, global_args)
        # elif "chatglm-pro" in self.model_name.lower() or "glm-api-v2" in self.model_name.lower():
        #     return self.server_by_chatglm_pro(raw_request, global_args)
        # elif "spark-api-v2" in self.model_name.lower():
        #     return self.server_by_spark(raw_request, global_args)
        # elif "ernie" in self.model_name.lower():
        #     return self.server_by_ernie(raw_request, global_args)
        # else:
        #     return self.server_by_triton_glm(raw_request, global_args)

    """
    封装循环请求，防止限流和模型返回异常
    """
    def do_post(self, url, data, headers, model_name, prompt):
        # 发送POST请求
        response = requests.post(url, json=data, headers=headers)
        print(f"response.text: {response.text}")
        retry_times = 0
        while retry_times < 10:
            # print("网络返回失败：[" + response.status_code + "]" + response.message)
            # print("重试：" + str(data))
            text = json.loads(response.text)
            answer = text

            if response.status_code == 200 and answer is not None and len(answer) > 0:
                return answer

            print(f"触发限流or模型没有返回结果 重试次数:{retry_times} 等待10秒后重试...")
            time.sleep(20)  # 多等一会 一分钟最多请求100次

            retry_times += 1

            response = requests.post(url, data=data, headers=headers)
        raise Exception("OpenAI 没有返回结果")

    """
    获取sleep秒数 sleep到下一分钟+1秒
    """
    def get_sleep_interval(self):
        from datetime import datetime, timedelta
        now = datetime.now()
        next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        delta = next_minute - now
        seconds = delta.seconds
        return seconds + 1

    """
    封装gpt3.5模型循环请求
    """
    @sleep_and_retry
    @peak_rate_limit(calls=1000, peak_calls=5, period=60)
    # @Cat.transaction("Http", "do_post_chatgpt")
    def do_post_chatgpt(self, msg, model_name, **gen_kwargs):
        # 设置要提交的数据
        openai.api_key = self.appId  # 在speech平台申请的
        openai.api_base = "https://aigc.sankuai.com/v1/openai/native"
        try:
            print(f"start query {model_name}..")
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=msg,
                **gen_kwargs,
            )
            print(f"query {model_name} end..")
        except Exception as e:
            print(f"query {model_name} exception: {e}")
            response = None
        retry_times = 0
        while retry_times < 30:
            if response is not None and 'choices' in response and len(response['choices']) > 0:
                break
            retry_times = retry_times + 1
            time.sleep(self.get_sleep_interval()) # sleep到下一整分钟
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=msg,
                    **gen_kwargs,
                )
            except:
                response = None
        return response

    """
    封装gpt4模型循环请求
    """
    @sleep_and_retry
    @peak_rate_limit(calls=1000, peak_calls=5, period=60)
    # @Cat.transaction("Http", "do_post_gpt4")
    def do_post_gpt4(self, msg, model_name, **gen_kwargs):
        # 设置要提交的数据
        openai.api_key = self.appId  # 在speech平台申请的
        openai.api_base = "https://aigc.sankuai.com/v1/openai/native"
        try:
            print(f"start query {model_name}..")
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=msg,
                **gen_kwargs,
            )
            print(f"query {model_name} end..")
        except Exception as e:
            print(f"query {model_name} exception: {e}")
            response = None
        retry_times = 0
        while retry_times < 10:
            if response is not None and 'choices' in response and len(response['choices']) > 0:
                break
            retry_times = retry_times + 1
            time.sleep(self.get_sleep_interval())  # rpm目前只有12，所以每次失败直接sleep1分钟防止浪费
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=msg,
                    **gen_kwargs,
                )
            except:
                response = None
        return response

    """
    请求gpt3的接口
    """
    def server_by_chatgpt(self, raw_request: Dict[str, Any], global_args, **gen_kwargs):
        prompt = raw_request["prompt"]
        # 特殊处理 这个字符会让openai挂掉..替换一下
        prompt = prompt.replace("<|endoftext|>", "<|end_of_text|>")
        if len(prompt) == 0:
            print("prompt长度为0")
        
        # 设置要提交的数据
        msg = []
        if "system_cmd" in raw_request:
            msg.append({"role": "system", "content": raw_request["system_cmd"]})
        msg.append({"role": "user", "content": prompt})

        # model
        model_name = global_args["model_name"].lower().split("/")[-1].split("[[")[0]
        if "-" in model_name:
            model_name = model_name.lower().replace("chatgpt", "gpt-3.5-turbo")
        else:
            model_name = model_name.lower().replace("chatgpt", "gpt-3.5-turbo-0613")

        # 发送POST请求
        answer = self.do_post_chatgpt(msg, model_name, **gen_kwargs)

        # 处理结果
        try:
            if not isinstance(answer, str) and answer is not None:
                answer = answer["choices"][0]["message"]["content"]
            if answer is None:
                print(f"{model_name} 结果返回异常:{answer}, exception: None result")
                answer = f"{model_name} 结果返回异常"
        except Exception as e:
            print(f"{model_name} 结果返回异常:{answer}, exception:{e}")
            answer = f"{model_name} 结果返回异常"
        completions = []
        tokens = []
        completions.append(
            {
                "text": answer,
                "tokens": tokens,
                "logprobs": [0.0] * len(tokens),
                "top_logprobs_dicts": [{token: 0.0} for token in tokens]
            }
        )
        # print(completions)
        return {"completions": completions, "input_length": len(prompt.split("\s"))}

    """
    请求gpt4的接口
    """
    def server_by_gpt4(self, raw_request: Dict[str, Any], global_args, **gen_kwargs):
        prompt = raw_request["prompt"]

        msg = []
        if "system_cmd" in raw_request:
            msg.append({"role": "system", "content": raw_request["system_cmd"]})
        msg.append({"role": "user", "content": prompt})

        # model
        model_name = global_args["model_name"].lower().split("/")[-1].split("[[")[0]
        if "gpt-4-0125" in self.model_name.lower():
            model_name = "gpt-4-0125-preview"
        elif "gpt-4-turbo-2024-04-09" in self.model_name.lower() or "gpt-4-turbo-eva" in self.model_name.lower():
            model_name = "gpt-4-turbo-eva"
            gen_kwargs.pop("top_k", "top_k not found")
            gen_kwargs.pop("top_p", "top_p not found")
            gen_kwargs.pop("do_sample", "do_sample not found")
            gen_kwargs.pop("max_length", "max_length not found")
        elif "gpt-4o" in self.model_name.lower():
            model_name = "gpt-4o-2024-05-13"
            gen_kwargs.pop("top_k", "top_k not found")
            gen_kwargs.pop("top_p", "top_p not found")
            gen_kwargs.pop("do_sample", "do_sample not found")
            gen_kwargs.pop("max_length", "max_length not found")
        else:
            model_name = "gpt-4-0613"

        # 请求结果
        answer = self.do_post_gpt4(msg, model_name, **gen_kwargs)
        # 处理结果
        try:
            if not isinstance(answer, str) and answer is not None:
                answer = answer["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"{model_name} 结果返回异常:{answer}")
            answer = f"{model_name} 结果返回异常"

        if answer is None:
            answer = f"{model_name} 结果返回异常"

        completions = []
        tokens = []
        completions.append(
            {
                "text": answer,
                "tokens": [],
                "logprobs": [0.0] * len(tokens),
                "top_logprobs_dicts": [{token: 0.0} for token in tokens]
            }
        )
        # print(completions)
        return {"completions": completions, "input_length": len(prompt.split("\s"))}

    """
    请求chatglm3-130b的接口
    """
    def server_by_chatglm3_130b(self, raw_request: Dict[str, Any], global_args):
        prompt = raw_request["prompt"]
        # prompt = "<|user|>\n" + prompt + "<|assistant|>\n"
        # 请求结果
        gen_kwargs = {
            "top_k": global_args["top_k"],
            "do_sample": global_args["do_sample"],
            "max_new_tokens": global_args["max_new_tokens"],
        }
        if global_args["temperature"] > 0:
            gen_kwargs["temperature"] = global_args["temperature"]
        if global_args["top_p"] > 0 and global_args["top_p"] < 1:
            gen_kwargs["top_p"] = global_args["top_p"]

        if "chatglm3-130b-chat" in self.model_name.lower():
            print("use chatglm3_130b_chat...")
            answer = ChatGLM3130BChatClient().generate(prompt, **gen_kwargs)
        else:
            print("use chatglm3_130b...")
            answer = ChatGLM3130BClient().generate(prompt, **gen_kwargs)

        # 处理结果
        completions = []
        tokens = []
        completions.append(
            {
                "text": answer,
                "tokens": [],
                "logprobs": [0.0] * len(tokens),
                "top_logprobs_dicts": [{token: 0.0} for token in tokens]
            }
        )
        # print(completions)
        return {"completions": completions, "input_length": len(prompt.split("\s"))}

    # """
    # Triton客户端调用GLM
    # """
    # def server_by_common_api(self, raw_request: Dict[str, Any], global_args):
    #     self.global_args = global_args
    #     ip_port = self.global_args["ip_port"]
    #     prompt = raw_request["prompt"]

    #     data = {
    #         "prompt": prompt,
    #         "max_new_tokens": global_args["max_new_tokens"]
    #     }
    #     data = json.dumps(data)
    #     headers = {
    #         "Content-Type": "application/json"
    #     }

    #     # 发送POST请求
    #     answer = ">>NO RESPONSE<<"
    #     for i in range(10):
    #         try:
    #             response = requests.post(f"http://{ip_port}", data=data, headers=headers)
    #             text = json.loads(response.text)
    #             if response.status_code == 200:
    #                 answer = text["result"]
    #                 break
    #             print(f"Common API request error: no result, retry:({i}/10)")
    #         except Exception as e:
    #             print(f"Common API request error: {e}, retry:({i}/10)")

    #     completions = []
    #     tokens = []
    #     completions.append(
    #         {
    #             "text": answer,
    #             "tokens": tokens,
    #             "logprobs": [0.0] * len(tokens),
    #             "top_logprobs_dicts": [{token: 0.0} for token in tokens]
    #         }
    #     )
    #     return {"completions": completions, "input_length": len(prompt.split("\s"))}

    """
    在线服务ft客户端调用
    """
    def server_by_ft_api(self, raw_request, global_args, **gen_kwargs):
        self.ft = FTClient(global_args)
        prompt = raw_request["prompt"]
        # print(f">>>actual request:{raw_request}")
        answer = self.ft.generate(prompt, raw_request, global_args)

        completions = []
        tokens = []
        completions.append(
            {
                "text": answer,
                "tokens": tokens,
                "logprobs": [0.0] * len(tokens),
                "top_logprobs_dicts": [{token: 0.0} for token in tokens]
            }
        )
        # print(completions)
        return {"completions": completions, "input_length": len(prompt.split("\s"))}

    """
    自己起的服务调用
    """
    def server_by_local_server(self, ip, raw_request: Dict[str, Any], global_args, wait=False, **gen_kwargs):
        prompt = raw_request["prompt"]
        # 请求结果
        try:
            answer = LocalClient().generate(ip, prompt, wait=wait, **gen_kwargs)
        except Exception as e:
            print(f"HTTP API 无结果返回: {e}")
            answer = "<<HTTP API 无结果返回>>"
            raise e
        # 处理结果
        completions = []
        tokens = []
        if isinstance(answer, list):
            completions = answer
        else:
            completions.append(
                {
                    "text": answer,
                    "tokens": [],
                    "logprobs": [0.0] * len(tokens),
                    "top_logprobs_dicts": [{token: 0.0} for token in tokens]
                }
            )
        # print(completions)
        return {"completions": completions, "input_length": len(prompt.split("\s"))}


    # """
    # 请求语音侧的openAI接口
    # """
    # def server_by_openai(self, raw_request: Dict[str, Any]):
    #     prompt = raw_request["prompt"]
    #     model_name = self.model_name
    #     if "text-davinci-003" in self.model_name.lower():
    #         model_name = "text-davinci-003"
    #         # 限流
    #         time.sleep(10)
    #     if "gpt-3.5-turbo-0613" in self.model_name.lower():
    #         model_name = "gpt-3.5-turbo-0613"
    #     if "gpt-3.5-turbo-16k-0613" in self.model_name.lower():
    #         model_name = "gpt-3.5-turbo-16k-0613"

    #     max_tokens = 2048
    #     if "max_tokens" in raw_request:
    #         max_tokens = raw_request["max_tokens"]
    #     # 设置要提交的数据
    #     data = {
    #         "prompt": prompt,
    #         "stream": "false",
    #         "model": model_name,
    #         "max_tokens": max_tokens
    #     }

    #     # 特殊处理需要算logprob
    #     if "gpt-3.5-turbo-instruct" in self.model_name.lower():
    #         model_name = "gpt-3.5-turbo-instruct"
    #         data = {
    #             "prompt": prompt,
    #             "stream": "false",
    #             "logprobs": 5,
    #             "model": model_name,
    #             "max_tokens": max_tokens
    #         }

    #     url = "https://aigc.sankuai.com/v1/openai/native/completions"
    #     # data = json.dumps(data)
    #     headers = {
    #         "Content-Type": "application/json",
    #         "Authorization": f"Bearer {self.appId}"
    #     }

    #     # 请求结果
    #     answer = self.do_post(url, data, headers, model_name, prompt)

    #     correct_logprob = 0.0
    #     # 正确答案的logprobs
    #     if "gpt-3.5-turbo-instruct" in self.model_name.lower() and 'target' in raw_request:
    #         target = raw_request['target']
    #         # 需要计算正确答案的logprobs
    #         # 目前只针对选择题，所以只拿第一个结果
    #         if target is not None:
    #             if "logprobs" in answer["choices"][0]:
    #                 if "top_logprobs" in answer["choices"][0]["logprobs"]:
    #                     top_logprob_dict = answer["choices"][0]["logprobs"]["top_logprobs"][0]
    #                     print(f"top_logprob_dict {top_logprob_dict}")
    #                     # 由于token处理和prompt问题，假设正确答案为'A',实际上我们考虑' A'和'\nA'都算正确答案 于是取这个三个的最大值。
    #                     correct_list = []
    #                     correct_list.append(target)
    #                     correct_list.append(" " + target)
    #                     correct_list.append("\n" + target)
    #                     for correct in correct_list:
    #                         if correct in top_logprob_dict:
    #                             if correct_logprob >= 0.0 or correct_logprob < top_logprob_dict[correct]:
    #                                 correct_logprob = top_logprob_dict[correct]
    #     # 处理结果
    #     if not isinstance(answer, str):
    #         answer = answer["choices"][0]["text"]

    #     completions = []
    #     tokens = []
    #     completions.append(
    #         {
    #             "text": answer,
    #             "tokens": [],
    #             "logprobs": [0.0] * len(tokens),
    #             "top_logprobs_dicts": [{token: 0.0} for token in tokens],
    #             "correct_logprob" : correct_logprob,
    #         }
    #     )
    #     # print(completions)
    #     return {"completions": completions, "input_length": len(prompt.split("\s"))}

    # def server_by_openai_chat(self, raw_request: Dict[str, Any]):
    #     prompt = raw_request["prompt"]
    #     model_name = self.model_name
    #     if "gpt-3.5-turbo-0613" in self.model_name.lower():
    #         model_name = "gpt-3.5-turbo-0613"
    #     if "gpt-3.5-turbo-16k-0613" in self.model_name.lower():
    #         model_name = "gpt-3.5-turbo-16k-0613"

    #     max_tokens = 2048
    #     if "max_tokens" in raw_request:
    #         max_tokens = raw_request["max_tokens"]
    #     # 设置要提交的数据
    #     data = {
    #         "messages": [{"role" : "user", "content": prompt}],
    #         "stream": "false",
    #         "model": model_name,
    #         "max_tokens": max_tokens
    #     }

    #     url = "https://aigc.sankuai.com/v1/openai/native/chat/completions"
    #     # data = json.dumps(data)
    #     headers = {
    #         "Content-Type": "application/json",
    #         "Authorization": f"Bearer {self.appId}"
    #     }

    #     # 请求结果
    #     answer = self.do_post(url, data, headers, model_name, prompt)

    #     correct_logprob = 0.0
    #     # 处理结果
    #     if not isinstance(answer, str):
    #         answer = answer["choices"][0]["message"]["content"]

    #     completions = []
    #     tokens = []
    #     completions.append(
    #         {
    #             "text": answer,
    #             "tokens": [],
    #             "logprobs": [0.0] * len(tokens),
    #             "top_logprobs_dicts": [{token: 0.0} for token in tokens],
    #             "correct_logprob": correct_logprob,
    #         }
    #     )
    #     # print(completions)
    #     return {"completions": completions, "input_length": len(prompt.split("\s"))}

    # def server_by_baichuan2_53b(self, raw_request: Dict[str, Any], global_args):
    #     prompt = raw_request["prompt"]
    #     time.sleep(7) # 限流10rpm ，每次请求sleep一会
    #     # 请求结果
    #     try:
    #         answer = Baichuan253BClient().generate(prompt)
    #     except Exception as e:
    #         print(f"Baichuan2 53B API 无结果返回: {e}")
    #         answer = "<<Baichuan2 53B API 无结果返回>>"
    #     # 处理结果
    #     completions = []
    #     tokens = []
    #     completions.append(
    #         {
    #             "text": answer,
    #             "tokens": [],
    #             "logprobs": [0.0] * len(tokens),
    #             "top_logprobs_dicts": [{token: 0.0} for token in tokens]
    #         }
    #     )
    #     # print(completions)
    #     return {"completions": completions, "input_length": len(prompt.split("\s"))}

    # def server_by_ernie(self, raw_request: Dict[str, Any], global_args):
    #     prompt = raw_request["prompt"]
    #     time.sleep(7)  # 限流10rpm ，每次请求sleep一会
    #     version = "v3.5"
    #     if "ernie-bot-4" in self.model_name.lower():
    #         version = "v4"
    #     # 请求结果
    #     try:
    #         answer = ErnieClient().generate(prompt, version)
    #     except Exception as e:
    #         print(f"ERNIE API 无结果返回: {e}")
    #         raise e
    #     # 处理结果
    #     completions = []
    #     tokens = []
    #     completions.append(
    #         {
    #             "text": answer,
    #             "tokens": [],
    #             "logprobs": [0.0] * len(tokens),
    #             "top_logprobs_dicts": [{token: 0.0} for token in tokens]
    #         }
    #     )
    #     # print(completions)
    #     return {"completions": completions, "input_length": len(prompt.split("\s"))}
    # def server_by_chatglm_pro(self, raw_request: Dict[str, Any], global_args):
    #     prompt = raw_request["prompt"]
    #     time.sleep(7)  # 限流10rpm ，每次请求sleep一会
    #     # 请求结果
    #     try:
    #         answer = ChatGLMProClient().generate(prompt)
    #     except Exception as e:
    #         print(f"ChatGLM PRO API 无结果返回: {e}")
    #         if "敏感内容" in str(e):
    #             answer = "系统检测到输入或生成内容可能包含不安全或敏感内容，请您避免输入易产生敏感内容的提示语，感谢您的配合。"
    #         else:
    #             answer = "ChatGLM PRO API 无结果返回，可能是没token了或者被风控了"
    #     # 处理结果
    #     completions = []
    #     tokens = []
    #     completions.append(
    #         {
    #             "text": answer,
    #             "tokens": [],
    #             "logprobs": [0.0] * len(tokens),
    #             "top_logprobs_dicts": [{token: 0.0} for token in tokens]
    #         }
    #     )
    #     # print(completions)
    #     return {"completions": completions, "input_length": len(prompt.split("\s"))}

    # def server_by_local_inference_file(self, raw_request: Dict[str, Any], global_args):
    #     prompt = raw_request["prompt"]

    #     if prompt not in global_args["local_inference_file"]:
    #         raise Exception(f"当前Prompt未在本地文件中找到对应结果。 Prompt:{prompt}")

    #     answer = global_args["local_inference_file"][prompt]

    #     # 处理结果
    #     completions = []
    #     tokens = []
    #     completions.append(
    #         {
    #             "text": answer,
    #             "tokens": [],
    #             "logprobs": [0.0] * len(tokens),
    #             "top_logprobs_dicts": [{token: 0.0} for token in tokens]
    #         }
    #     )
    #     # print(completions)
    #     return {"completions": completions, "input_length": len(prompt.split("\s"))}

    # def server_by_spark(self, raw_request: Dict[str, Any], global_args):
    #     prompt = raw_request["prompt"]
    #     time.sleep(1)  # 限流1qps ，每次请求sleep一会
    #     # 请求结果
    #     try:
    #         answer = SparkClient().generate(prompt)
    #     except Exception as e:
    #         print(f"Spark 无结果返回: {e}")
    #         raise e
    #         # answer = "<<Spark无结果返回>>"
    #     # 处理结果
    #     completions = []
    #     tokens = []
    #     completions.append(
    #         {
    #             "text": answer,
    #             "tokens": [],
    #             "logprobs": [0.0] * len(tokens),
    #             "top_logprobs_dicts": [{token: 0.0} for token in tokens]
    #         }
    #     )
    #     # print(completions)
    #     return {"completions": completions, "input_length": len(prompt.split("\s"))}

    # def server_by_chatglm2_66b(self, raw_request: Dict[str, Any], global_args):
    #     prompt = raw_request["prompt"]
    #     max_new_tokens = 1500
    #     if 'max_tokens' in raw_request:
    #         max_new_tokens = raw_request["max_tokens"]

    #     # 请求结果
    #     answer = ChatGLM266BClient().generate(prompt, max_new_tokens)
    #     # 处理结果
    #     completions = []
    #     tokens = []
    #     completions.append(
    #         {
    #             "text": answer,
    #             "tokens": [],
    #             "logprobs": [0.0] * len(tokens),
    #             "top_logprobs_dicts": [{token: 0.0} for token in tokens]
    #         }
    #     )
    #     # print(completions)
    #     return {"completions": completions, "input_length": len(prompt.split("\s"))}

    # def server_by_triton_glm(self, raw_request: Dict[str, Any], global_args):
    #     if self.glm or self.glm6b is None:
    #         self.init_glm()
    #     prompt = raw_request["prompt"]
    #     model_name = self.model_name
    #     if "api" in global_args and global_args["api"] == "prototype":
    #         model_name = "prototype"

    #     glm = self.glm6b if '6b' in model_name.lower() else self.glm

    #     print("use global_args.")
    #     self.global_args.argmap = global_args
    #     print(self.global_args.argmap)
    #     # ipport
    #     if "ip-port" in self.global_args.argmap:
    #         ip_port = self.global_args.argmap["ip-port"]
    #     else:
    #         ip_port = None
    #     # appkey
    #     if "appkey" in self.global_args.argmap:
    #         appkey = self.global_args.argmap["appkey"]
    #     else:
    #         appkey = "com.sankuai.data.llminfer.glm130bfp16"

    #     # 请求结果
    #     if model_name == "ChatGLM130B":
    #         appkey = "com.sankuai.data.llminfer.glm130bfp16"
    #         answer = glm.inference(prompt, appkey, ip_port, "glm130b" + "_ft_fp16", raw_request["max_tokens"])
    #     elif model_name == "glm_ft_zhangjiaqi39":
    #         appkey = "com.sankuai.nlp.llm.performance"
    #         answer = glm.inference(prompt, appkey, ip_port, model_name, raw_request["max_tokens"])
    #     else:
    #         answer = glm.inference(prompt, appkey, ip_port, model_name, raw_request["max_tokens"])

    #     completions = []
    #     tokens = []
    #     completions.append(
    #         {
    #             "text": answer,
    #             "tokens": tokens,
    #             "logprobs": [0.0] * len(tokens),
    #             "top_logprobs_dicts": [{token: 0.0} for token in tokens]
    #         }
    #     )
    #     # print(completions)
    #     return {"completions": completions, "input_length": len(prompt.split("\s"))}

    # """
    # 请求poker
    # """

    # def server_by_poker(self, raw_request: Dict[str, Any]):
    #     prompt = raw_request["prompt"]
    #     model_name = self.model_name
    #     # 设置要提交的数据
    #     data = {
    #         "modelName": model_name,
    #         "sessionName": "对话1",
    #         "request": prompt,
    #         "context": [],
    #         "user": "zhaoyunke",
    #         "userKey": "__pokerllm__"
    #     }
    #     url = "http://10.98.69.101:8080/mlapi/llm/system/infer"
    #     data = json.dumps(data)
    #     headers = {
    #         "Content-Type": "application/json"
    #     }

    #     # 发送POST请求
    #     answer = self.do_post(url, data, headers, model_name, prompt)

    #     completions = []
    #     tokens = []
    #     completions.append(
    #         {
    #             "text": answer,
    #             "tokens": tokens,
    #             "logprobs": [0.0] * len(tokens),
    #             "top_logprobs_dicts": [{token: 0.0} for token in tokens]
    #         }
    #     )
    #     # print(completions)
    #     return {"completions": completions, "input_length": len(prompt.split("\s"))}

    # def server_by_gpt4_vision(self, raw_request: Dict[str, Any], global_args):
    #     prompt = raw_request["prompt"]
    #     # 特殊处理 这个字符会让openai挂掉..替换一下
    #     prompt = prompt.replace("<|endoftext|>", "<|end_of_text|>")
    #     if len(prompt) == 0:
    #         print("prompt长度为0")
        
    #     # 设置要提交的数据
    #     msg = []
    #     if raw_request["url"] is not None:
    #         msg.append({"role": "user", 
    #                     "content": [
    #                         {"type":"text", "text": prompt},
    #                         {"type":"image_url", "image_url": raw_request["url"]}
    #                     ]})
    #     else:
    #         msg.append({"role": "user", 
    #                     "content": [
    #                         {"type": "text", "text": prompt}
    #                     ]})

    #     # 发送POST请求
    #     answer = self.do_post_gpt4_vision(msg)

    #     # 处理结果
    #     try:
    #         if not isinstance(answer, str) and answer is not None:
    #             answer = answer["choices"][0]["message"]["content"]
    #         if answer is None:
    #             print(f"GPT4-vision 结果返回异常:{answer}, exception: None result")
    #             answer = "GPT4-vision 结果返回异常"
    #     except Exception as e:
    #         print(f"GPT4-vision 结果返回异常:{answer}, exception:{e}")
    #         answer = "GPT4-vision 结果返回异常"
    #     completions = []
    #     tokens = []
    #     completions.append(
    #         {
    #             "text": answer,
    #             "tokens": tokens,
    #             "logprobs": [0.0] * len(tokens),
    #             "top_logprobs_dicts": [{token: 0.0} for token in tokens]
    #         }
    #     )
    #     # print(completions)
    #     return {"completions": completions, "input_length": len(prompt.split("\s"))}

#     def server_by_gpt_functional(self, raw_request: Dict[str, Any], global_args, gpt_model):
#         """
#         请求chatgpt的functional接口
#         """
#         raw_prompt = raw_request["prompt"]
#         prompt_json = json.loads(raw_prompt)
#         functions = prompt_json["functions"]
#         messages = prompt_json["messages"]
#         max_tokens = 16000
#         if "max_tokens" in raw_request:
#             max_tokens = raw_request["max_tokens"]

#         # 设置要提交的数据
#         data = {
#             "model": gpt_model,
#             "messages": messages,
#             "functions": functions,
#             "max_tokens": max_tokens
#         }

#         url = "https://aigc.sankuai.com/v1/openai/native/chat/completions"
#         data = json.dumps(data)
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {self.appId}"
#         }

#         # 请求结果
#         answer = self.do_post_gpt_function(url, data, headers)
#         # 处理结果
#         if not isinstance(answer, str):
#             if "choices" in answer:
#                 answer = str(answer["choices"])
#             else:
#                 answer = str(answer)
#         completions = []
#         tokens = []
#         completions.append(
#             {
#                 "text": answer,
#                 "tokens": [],
#                 "logprobs": [0.0] * len(tokens),
#                 "top_logprobs_dicts": [{token: 0.0} for token in tokens]
#             }
#         )
#         print("server by gpt functional")
#         # print(completions)
#         return {"completions": completions, "input_length": 0}

#     def do_post_gpt4_vision(self, msg):
#         # 设置要提交的数据
#         openai.api_key = self.appId  # 在speech平台申请的
#         openai.api_base = "https://aigc.sankuai.com/v1/openai/explore"
#         try:
#             print("start query openai gpt4-vision..")
#             response = openai.ChatCompletion.create(
#                 model="gpt-4-vision-preview",
#                 messages=msg,
#                 stream=False
#             )
#             print("query openai end..")
#         except Exception as e:
#             print(f"query openai exception: {e}")
#             response = None
#         retry_times = 0
#         while retry_times < 10:
#             if response is not None and 'choices' in response and len(response['choices']) > 0:
#                 break
#             retry_times = retry_times + 1
#             time.sleep(self.get_sleep_interval())  # sleep到下一整分钟
#             try:
#                 response = openai.ChatCompletion.create(
#                     model="gpt-4-vision-preview",
#                     messages=msg,
#                     stream=False
#                 )
#             except:
#                 response = None
#         return response

#     def do_post_gpt_function(self, url, data, headers):
#         # 发送POST请求
#         response = requests.post(url, data=data, headers=headers)
#         print(response.text)
#         retry_times = 0
#         while retry_times < 10:
#             try:
#                 text = json.loads(response.text)
#             except:
#                 answer = "请求不合法"
#                 break
#             answer = text
#             if response.status_code == 200 and answer is not None and len(answer) > 0:
#                 break

#             if text.get("error") is not None and text["error"].get("type") == "invalid_request_error":
#                 answer = "请求不合法"
#                 break

#             if "status" in text and "message" in text and (text["status"] == 450 or text["message"] == "输入内容涉黄暴政"):
#                 answer = "输入内容涉黄暴政"
#                 break

#             print(f"触发限流or模型没有返回结果 重试次数:{retry_times} 等待10秒后重试...")
#             time.sleep(10)  # 多等一会 一分钟最多请求100次
#             retry_times += 1
#             response = requests.post(url, data=data, headers=headers)
#         return answer

#     def server_by_minimax(self, raw_request, global_args):
#         prompt = raw_request["prompt"]
#         model_name = self.model_name
#         # 设置要提交的数据
#         request_body = {
#             "model": "abab5-chat",
#             "tokens_to_generate": 1024,
#             'messages': [{"sender_type": "USER", "text": prompt}]
#         }
#         headers = {
#             "Authorization": f"Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJTdWJqZWN0SUQiOiIxNjgxOTg2NjAxMzkwNDE4IiwiUGhvbmUiOiIxOTgqKioqNzcxMSIsIkdyb3VwSUQiOiIiLCJQYWdlTmFtZSI6IiIsIk1haWwiOiJ3aG1hZGFuMTk5MEBnbWFpbC5jb20iLCJpc3MiOiJtaW5pbWF4In0.oYZDR9sEwP3XWBjk_hibSrHq25q8SY1V4dN1hmhstDF1BBnVqxN4sLolFOb5K5ysemmB2go_AazAPquBSxm1UxbbMapM_U6CO9kGq8Efqu9uPn9xY6t9cVog1BC5ntPRMzDpScyOweZnOj7FmjU7hrMxbhRhFGbp7vR_ZRUoobzgkw3K60Nhsj7bBriV5Gde7E9nFl73OG_ZmVu0H1s6kLH0lupBM01NtF733bgVOHzp_j-Woe76kn7p2Odw34jlx6OaSOkK2OL146mtYDLPsSkXryCjYjKrJ5arlOBHo1lsN-eN122bfjLnA6v5aBZetbRGRUg-V7jnQIak6suB3Q",
#             "Content-Type": "application/json"
#         }

#         url = f'https://api.minimax.chat/v1/text/chatcompletion?GroupId=1681986601392417'

#         # 请求结果
#         answer = self.do_post_minimax(url, request_body, headers)
#         # 处理结果

#         completions = []
#         tokens = []
#         completions.append(
#             {
#                 "text": answer,
#                 "tokens": [],
#                 "logprobs": [0.0] * len(tokens),
#                 "top_logprobs_dicts": [{token: 0.0} for token in tokens]
#             }
#         )
#         # print(completions)
#         return {"completions": completions, "input_length": len(prompt.split("\s"))}

#     """
#      封装minimax循环请求，防止限流和模型返回异常
#      """

#     def do_post_minimax(self, url, data, headers):
#         # 发送POST请求
#         response = requests.post(url, json=data, headers=headers)
#         retry_times = 0
#         answer = ""
#         while retry_times < 10:
#             try:
#                 answer = response.json()['reply']
#                 if answer is not None and isinstance(answer, str) and len(answer) > 0:
#                     break
#             except:
#                 pass
#             print(f"触发限流or模型没有返回结果 重试次数:{retry_times} 等待10秒后重试...")
#             time.sleep(10)
#             retry_times += 1
#             response = requests.post(url, data=data, headers=headers)
#         return answer


# class HttpAPIClient(Client):

#     def __init__(self, cache_config: CacheConfig):
#         self.cache = Cache(cache_config)
#         self.model_server_instances: Dict[str, HttpAPIServer] = {}

#     def get_model_server_instance(self, model) -> HttpAPIServer:
#         if model not in self.model_server_instances:
#             self.model_server_instances[model] = HttpAPIServer(model)
#         return self.model_server_instances[model]

#     """
#     提交请求 主体逻辑跟Huggingface client一致
#     """

#     def make_request(self, request: Request, global_args) -> RequestResult:
#         # Embedding not supported for this model
#         if request.embedding:
#             return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

#         raw_request = {
#             "prompt": request.prompt,
#             "max_tokens": request.max_tokens,
#             "target": request.target,
#             "url": request.url
#         }

#         # 设置EOS配置
#         if "eos" in global_args:
#             request.stop_sequences = global_args['eos']

#         model_server_instance: HttpAPIServer = self.get_model_server_instance(request.model)

#         try:

#             def do_it():
#                 return model_server_instance.serve_request(raw_request, global_args)

#             cache_key = Client.make_cache_key(raw_request, request)
#             response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
#         except Exception as e:  # Do something if error is encountered.
#             logging.exception(e)
#             error: str = f"http api error: {e}"
#             return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])
#         # 处理结果 由于不需要tokenize，大部分只是为了匹配框架返回格式
#         completions = []
#         for raw_completion in response["completions"]:
#             sequence_logprob: float = 0
#             tokens: List[Token] = []

#             if request.echo_prompt:
#                 # Add prompt to list of generated tokens.
#                 generated_tokens = raw_completion["tokens"][response["input_length"]:]
#                 for token_text in raw_completion["tokens"][: response["input_length"]]:
#                     tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))
#             else:
#                 generated_tokens = raw_completion["tokens"]

#             # Compute logprob for the entire sequence.
#             for token_text, logprob, top_logprobs_dict in zip(
#                     generated_tokens, raw_completion["logprobs"], raw_completion["top_logprobs_dicts"]
#             ):
#                 tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
#                 sequence_logprob += logprob

#             if 'correct_logprob' not in raw_completion:
#                 raw_completion['correct_logprob'] = 0.0

#             completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens, correct_logprob=raw_completion['correct_logprob'])
#             completion = truncate_sequence(completion, request)
#             completions.append(completion)

#         return RequestResult(
#             success=True,
#             cached=cached,
#             request_time=response["request_time"],
#             request_datetime=response.get("request_datetime"),
#             completions=completions,
#             embedding=[],
#         )

#     # 用不到 直接返回none
#     def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
#         return None

#     def decode(self, request: DecodeRequest) -> DecodeRequestResult:
#         return None