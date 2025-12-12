from text_generation import Client


class ChatGLM3130BClient():

    def __init__(self):
        self.client = Client("http://10.166.58.30:8080", timeout=600) # 设置超时时间为60s

    def generate(self, prompt, **gen_kwargs):
        try:
            return self.client.generate(prompt, **gen_kwargs).generated_text
        except Exception as e:
            truncate_prompt = prompt
            while len(truncate_prompt) > 100:
                truncate_prompt = truncate_prompt[:-100]
                try:
                    return self.client.generate(truncate_prompt, **gen_kwargs).generated_text
                except:
                    pass
            raise e


class ChatGLM3130BChatClient():

    def __init__(self):
        self.client = Client("http://10.166.37.66:8080", timeout=600)  # 设置超时时间为60s

    def generate(self, prompt, **gen_kwargs):
        try:
            return self.client.generate(prompt, **gen_kwargs).generated_text
        except Exception as e:
            truncate_prompt = prompt
            while len(truncate_prompt) > 100:
                truncate_prompt = truncate_prompt[:-100]
                try:
                    return self.client.generate(truncate_prompt, **gen_kwargs).generated_text
                except:
                    pass
            raise e