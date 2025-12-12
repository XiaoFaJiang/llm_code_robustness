import os
import nlp_llm_client as nlc

from pycat import Cat
# from helm.proxy.clients.icetk_glm_130B import _IceTokenizer


class FTClient():
    def __init__(self, global_args):
        model_series = global_args["model_series"]
        model_version = global_args["model_version"]
        appkey = global_args["appkey"]
        ipport = global_args["ip_port"]
        Cat.init_cat("com.sankuai.poker.llm", disable_falcon=True)
        self.get_client(model_series, model_version, appkey, ipport, global_args)

    def get_tokenizer(self):
        try:
            return self.client.get_tokenizer()
        except:
            return self.client.tokenizer

    def get_client(self, model_series, model_version, appkey, ipport, global_args):
        # 1、新建client实例
        model_series = model_series  # 模型系列。这里为固定值"llama"
        model_version = model_version  # 模型版本。llama模型目前支持：7B / 13B / 65B
        appkey = appkey  # APPKEY名称
        ipport = ipport  # 用于直连的"IP:PORT"。可选参数，不指定为默认值：""
        self.tokenizer_path = None
        # if "glm" in model_series and '130B' == model_version:
        #     PWD = os.path.abspath(os.path.dirname(__file__))
        #     self.tokenizer_path = os.path.join(PWD,'icetk_glm_130B/')
        if "tokenizer_path" in global_args:
            self.tokenizer_path = global_args["tokenizer_path"]
        # print(f"API Client init: model_series: {model_series}, model_version: {model_version}, appkey: {appkey}, ipport: {ipport}")
        try:
            # appkey调用，使用此方式
            if ipport is None:
                if self.tokenizer_path:
                    self.client = nlc.new_client(model_series, model_version, appkey, tokenizer_path=self.tokenizer_path)
                else:
                    self.client = nlc.new_client(model_series, model_version, appkey)
            else:
                # ip直连或非生产环境，使用此方式
                if self.tokenizer_path:
                    self.client = nlc.new_client(model_series, model_version, appkey, ipport=ipport, tokenizer_path=self.tokenizer_path)
                else:
                    self.client = nlc.new_client(model_series, model_version, appkey, ipport=ipport)

            # if "glm" in model_series and '130B' == model_version:
            #     self.client.tokenizer = _IceTokenizer()
            # 如果需要自行指定tokenizer，要在new_client中指定关键字参数"tokenizer_path=tokenizer_path"
        except Exception as e:
            raise Exception('API Client get client error, msg[{}]'.format(str(e)))

    def generate(self, prompt, raw_request, global_args):
        # 2、单条文本预测
        model_name = 'prototype'  # config.pbtxt指定的模型名
        if "ft_model_name" in global_args and global_args["ft_model_name"]:
            model_name = global_args["ft_model_name"]
        input_text = prompt  # 请求文本
        maximum_output_length = global_args["max_new_tokens"]
        beam_width = 1  # Beam搜索宽度，确定了搜索中保留最有可能的候选序列个数。如果设为1，则只保留1个。不指定为默认值：1
        temperature = global_args["temperature"]
        sampling_topk = global_args["top_k"]
        sampling_topp = global_args["top_p"]
        do_sample = global_args["do_sample"]
        try:
            # 最简参数形式，超参会保持默认值
            # response = self.client.inference(model_name, input_text)

            # 自定义任意超参，需要哪个指定哪个
            response = self.client.inference(model_name, input_text, maximum_output_length=maximum_output_length,
                                             beam_width=beam_width, sampling_topk=sampling_topk,
                                             sampling_topp=sampling_topp, temperature=temperature)

            # 解析返回结果
            if response.code == 0:
                # 请求成功
                return response.output_text
            else:
                print(f"API评估异常: {response.error_message}")
                raise RuntimeError(f"API评估异常: {response.error_message}")
        except Exception as e:
            print(f"API评估异常: {e}")
            #  抛出异常而不是跳过，否则下次再重跑会认为这个任务成功了
            raise e
