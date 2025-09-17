"""
该模块提供了一个RequestLLM类，用于与语言模型API进行交互，获取模型的文本生成结果。

类:
    RequestLLM - 负责初始化API客户端，并向语言模型发送请求以获取文本生成的结果。

方法:
    __init__(config: Config) - 使用给定的配置对象初始化RequestLLM类的实例。
    get_response(messages: list) - 向语言模型API发送请求，并获取响应。
    get_llm_result(prompt: str) - 使用指定的提示信息向语言模型请求生成文本。

使用示例:
    # 创建Config对象并初始化RequestLLM类的实例
    config = Config(...)
    request_llm = RequestLLM(config)
    
    # 使用get_llm_result方法传入提示信息来获取模型的生成结果
    prompt = "Hello, world!"
    result = request_llm.get_llm_result(prompt)
    
    # 返回值result是模型生成的文本字符串
"""
import logging
import re
import json
import time
import random

from agentpedia.config import Config
from agentpedia.utils.request_eb_llm import request_eb
from agentpedia.web.local_cache import load_from_cache, save_to_cache

global_token_dict = {"p": 0, "c": 0, "n": 0}

class RequestLLM:
    """
    RequestLLM - 负责初始化API客户端，并向语言模型发送请求以获取文本生成的结果。
    """

    def __init__(self, config: Config):
        """
        初始化RequestLLM类的实例。

        参数:
            config (Config): 包含必要配置信息的配置对象，如API密钥、基础URL以及模型参数等。
        """
        self.config = config
        if self.config.is_debug:
            from openai import OpenAI
            self.clients = [OpenAI(api_key=key, base_url=self.config.base_url) for key in self.config.api_keys]

    def get_response(self, messages):
        """
        向语言模型API发送请求，并根据给定的消息列表获取响应。

        参数:
            messages (list): 包含一个或多个消息字典的列表，每个字典代表一个交互消息。

        返回:
            dict: 包含API响应的字典。
        """
        # print(self.config.model)
        completion = self.clients[0].chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens
        )
        print(f"\033[31m{completion.usage.completion_tokens}-{completion.usage.prompt_tokens}\033[0m")
        return completion

    def get_llm_result(self, prompt, model_name = ""):
        """
        使用指定的提示信息(prompt)向语言模型请求生成文本。

        参数:
            prompt (str): 用于请求语言模型生成文本的提示信息。

        返回:
            str: 语言模型根据提示信息生成的文本结果。
        """
        model_name = model_name if model_name != "" else 'gpt-4o'
        if self.config.model_name == "gpt-4o":
            model_name = self.config.model_name
        if prompt is None or prompt == "" or len(prompt) == 0:
            return ""
        retry_threshold = 2
        if self.config.is_debug:
            retry_threshold = 20

        cache_param = {"prompt": prompt}
        if model_name != "gpt-4o":
            cache_param["model"] = model_name
        if self.config.from_cache:
            cache_str = load_from_cache("get_llm_result", cache_param)
            if cache_str != "":
                return cache_str

        result = ''
        # print('using '+ model_name)

        failed = False
        for _ in range(retry_threshold):
            if model_name == "gpt-4o":
                try:
                    message = [{"role": "user", "content": prompt}]
                    response = self.get_response(message)
                    result = response.choices[0].message.content.strip()
                    if self.config.model in ["deepseek-distill-llama-70b", "deepseek-reasoner", "qwq-32b"]:
                        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
                        result = re.sub(r'.*?</think>', '', result, flags=re.DOTALL)
                    failed = False
                except Exception as e:
                    failed = True
                    if '使用量已用完，请明日再试！' in str(e):
                        print('使用量已用完，请明日再试！')
                    elif '您的请求过于频繁' in str(e):
                        print('您的请求过于频繁')
                    else:
                        print('[Error]', e)
            else:
                exit(-1)
                result = request_eb(prompt, model_name)
                if result:
                    failed = False
                else:
                    failed = True
            if not failed:
                break
            time.sleep(random.uniform(10, 20))

        if failed:
            print('请求失败: ' + prompt)
            if not self.config.is_debug:
                raise Exception('千帆请求结果为空')
            raise Exception('千帆请求结果为空')
        else:
            if self.config.is_debug and self.config.from_cache:
                save_to_cache("get_llm_result", cache_param, result)
        return result

    def fetch_content_result(self, content_prompt, index, model_name = ""):
        """
        并发请求llm，并返回结果和索引。
        """
        return self.get_llm_result(content_prompt, model_name), index

    def fetch_content_dict_result(self, content_prompt, index, key, model_name = ""):
        """
        并发请求llm，并返回结果和索引。
        """
        return self.get_llm_result(content_prompt, model_name), index, key

    # @staticmethod
    def parse_json_response(self, response, logger, retry=0, parse_list=False):
        """
        从模型输出中提取JSON数据。
        """
        response = response.replace('\n', '')
        response = response.replace('\t', '')
        json_pattern_list = [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```']
        if parse_list:
            json_pattern_list.extend([r'```json\s*(\[.*?\])\s*```', r'```\s*(\[.*?\])\s*```'])
        for pattern in json_pattern_list:
            json_pattern = re.compile(pattern, re.DOTALL)
            match = json_pattern.search(response)

            if match:
                # print(f'======={pattern}=======')
                json_string = match.group(1)
                try:
                    data = json.loads(json_string)
                    if parse_list:
                        if isinstance(data, list):
                            return data                        
                    else:
                        if isinstance(data, dict):
                            return data
                except json.JSONDecodeError as e:
                    try:
                        json_string = json_string.replace("\\U", "\\\\U")
                        data = json.loads(json_string)
                        if parse_list:
                            if isinstance(data, list):
                                return data                        
                        else:
                            if isinstance(data, dict):
                                return data
                    except json.JSONDecodeError as e:
                        logger.error(f"解析JSON时出错: {e}")
                        logger.error(f"JSON字符串: {json_string}")
        
        try:
            data = json.loads(response)
            if parse_list:
                if isinstance(data, list):
                    return data                        
            else:
                if isinstance(data, dict):
                    return data
        except json.JSONDecodeError as e:
            logger.error(f"解析JSON时出错: {e}")
            logger.error(f"JSON字符串: {response}")
        
        print("error json :  " + response)
        if retry == 0:
            response = self.get_llm_result(f"以下内容不是合法的JSON格式：{response}\n请你检查并修改成正确的json格式，禁止修改原来的内容，只修改格式。注意回答前后必须要有```json\n回答\n```，隔离开{{}}前后的其他文字)")
            return self.parse_json_response(response, logger, retry=retry + 1)
        raise ValueError
        return None