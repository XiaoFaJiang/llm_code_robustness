# encoding=utf-8
import json
import time
import openai
import logging
import requests

from typing import Any, Dict, List
from bigcode_eval.local_client import LocalClient
from bigcode_eval.ratelimit import sleep_and_retry, peak_rate_limit

class HttpAPIServer:
    def __init__(self, model_name: str, global_args: Dict):
        try:
            self.model_name = model_name
            self.global_args = global_args

        except Exception as e:
            logging.exception(e)
            raise e

    @staticmethod
    def name_match_list():
        return [
            "openai/chatgpt",
            "openai/gpt4",
            "openai/gpt-4",
            "openai/gpt-4o",
            "deepseek/deepseek-v2-chat",
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
        raw_request["prompt"] = prompt
        ip = global_args.get("api")
        return self.server_by_local_server(ip, raw_request, global_args, wait=True, **gen_kwargs)


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