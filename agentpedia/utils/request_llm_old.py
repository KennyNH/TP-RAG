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
from openai import OpenAI
from agentpedia.config import Config
from agentpedia.context.article import ArticleContext
from agentpedia.utils.request_eb_llm import request_eb
from agentpedia.web.local_cache import load_from_cache, save_to_cache


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
        self.clients = [OpenAI(api_key=key, base_url=self.config.base_url) for key in self.config.api_keys]

    def get_response(self, messages):
        """
        向语言模型API发送请求，并根据给定的消息列表获取响应。

        参数:
            messages (list): 包含一个或多个消息字典的列表，每个字典代表一个交互消息。

        返回:
            dict: 包含API响应的字典。
        """
        completion = self.clients[0].chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens
        )
        return completion

    def get_llm_result(self, prompt):
        """
        使用指定的提示信息(prompt)向语言模型请求生成文本。

        参数:
            prompt (str): 用于请求语言模型生成文本的提示信息。

        返回:
            str: 语言模型根据提示信息生成的文本结果。
        """
        cache_param = {"prompt": prompt}
        if self.config.model_name != "gpt-4o":
            cache_param["model"] = self.config.model_name
        if self.config.from_cache:
            cache_str = load_from_cache("get_llm_result", cache_param)
            if cache_str != "":
                return cache_str

        count_retry = 0
        result = ''
        while count_retry < 10:
            failed = False
            if self.config.model_name == "gpt-4o":
                try:
                    message = [{"role": "user", "content": prompt}]
                    response = self.get_response(message)
                    result = response.choices[0].message.content.strip()
                except Exception as e:
                    failed = True
                    if '使用量已用完，请明日再试！' in str(e):
                        print('使用量已用完，请明日再试！')
                    elif '您的请求过于频繁' in str(e):
                        print('您的请求过于频繁')
                    else:
                        print('[Error]', e)
            else:
                result = request_eb(prompt, self.config.model_name)
                if not result:
                    failed = True

            if not failed:
                break
            count_retry += 1
            time.sleep(2)

        if count_retry >= 10:
            print('请求失败: ' + prompt)
        else:
            if self.config.from_cache:
                save_to_cache("get_llm_result", cache_param, result)
        return result

    def fetch_content_result(self, content_prompt, index):
        """
        并发请求llm，并返回结果和索引。
        """
        return self.get_llm_result(content_prompt), index

    def fetch_content_dict_result(self, content_prompt, index, key):
        """
        并发请求llm，并返回结果和索引。
        """
        return self.get_llm_result(content_prompt), index, key

    @staticmethod
    def parse_json_response(response, logger: logging.Logger):
        """
        从模型输出中提取JSON数据。
        """
        response = response.replace('\n', '')
        response = response.replace('\t', '')
        json_pattern_list = [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```', r'json\s*(\{.*?\})\s*', r'\s*(\{.*?\})\s*']
        for pattern in json_pattern_list:
            json_pattern = re.compile(pattern, re.DOTALL)
            match = json_pattern.search(response)

            if match:
                json_string = match.group(1)
                try:
                    data = json.loads(json_string)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError as e:
                    logger.error(f"解析JSON时出错: {e}")
                    logger.error(f"JSON字符串: {json_string}")
        return None
