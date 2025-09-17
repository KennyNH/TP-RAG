"""
该模块包含用于与数据平台进行通信的RequestDatapower类，以获取与关键词相关的搜索数据。

类:
    RequestDatapower - 通过HTTP请求与数据平台交互，以获取和发送数据。

方法:
    __init__(config: Config) - 初始化RequestDatapower类的实例，接受一个Config对象作为配置。
    sendGetRequest(url: str, param: dict, retry: int) - 向指定的URL发送GET请求，并获取响应。
    search_key(key: str) - 根据提供的关键词(key)，从数据平台检索相关的搜索数据。

使用示例:
    # 首先，创建Config对象并初始化RequestDatapower类的实例。
    config = Config(...)
    request_datapower = RequestDatapower(config)
    
    # 然后，调用search_key方法并传入关键词获取相关数据。
    key = '奥运会'
    result = request_datapower.search_key(key)
    
    # result将包含与关键词相关的搜索数据列表。
"""
import sys
import urllib.parse
import urllib.request
import json
from agentpedia.config import Config


class RequestDatapower:
    """
    通过HTTP请求与数据平台交互，以获取和发送数据。
    """

    def __init__(self, config: Config):
        """
        初始化RequestDatapower类的实例。

        参数:
            config (Config): 一个配置对象，包含了必要的配置信息。
        """
        self.config = config

    def sendGetRequest(self, url, param={}, retry=0):
        """
        向指定的URL发送GET请求，并根据参数和重试次数获取响应。

        参数:
            url (str): 要发送请求的完整URL。
            param (dict, 可选): 要随GET请求一同发送的参数，以键值对的形式，默认为空字典。
            retry (int, 可选): 如果请求失败，重试的次数，默认为0次。

        返回:
            bytes: 从服务器获取的响应数据。
        
        抛出:
            URLError: 如果请求过程中出现了URL相关的错误。
            HTTPError: 如果服务器返回了HTTP错误代码。
        """
        data = urllib.parse.urlencode(param)
        request = urllib.request.Request(url='%s%s%s' % (url, '?', data))
        response = urllib.request.urlopen(request)
        ret = response.read()
        return ret

    def search_key(self, key):
        """
        根据提供的关键词(key)，从数据平台检索相关的搜索数据。

        参数:
            key (str): 要搜索的关键词。

        返回:
            list: 包含与关键词相关的搜索数据条目的列表，每个条目是一个字典。
        """

        result = []
        url = self.config.DataPower + self.config.query_api
        table_name = self.config.table_name
        if table_name == "":
            return -1

        params = {"username": self.config.username, "dataname": table_name, "keyinfo": key}

        ret = self.sendGetRequest(url, params)
        ret = json.loads(ret.decode('gb18030').encode('utf-8'))

        if ret["status"] == 1:
            sys.stderr.write("request Kv query failed")
            return ret

        ret = json.loads(ret["data"])
        table_value = ret["0"]

        if "head" not in table_value["data"] or len(table_value["data"]["head"]) == 0:
            if "loop" in table_value["data"]:
                for loop_data in table_value["data"]["loop"]:
                    result_one = {}
                    result_one[table_value["schema"]["key"]] = table_value["data"]["key"]
                    for index in range(len(table_value["schema"]["loop"])):
                        result_one[table_value["schema"]["loop"][index]] = loop_data[index]
                    result.append(result_one)

        elif "loop" not in table_value["data"] or len(table_value["data"]["loop"]) == 0:
            if "head" in table_value["data"]:
                result_one = {}
                result_one[table_value["schema"]["key"]] = table_value["data"]["key"]
                for head_index in range(len(table_value["data"]["head"])):
                    result_one[table_value["schema"]["head"][head_index]] = table_value["data"]["head"][head_index]
                result.append(result_one)
        else:
            for loop_data in table_value["data"]["loop"]:
                result_one = {}
                result_one[table_value["schema"]["key"]] = table_value["data"]["key"]
                for head_index in range(len(table_value["data"]["head"])):
                    result_one[table_value["schema"]["head"][head_index]] = table_value["data"]["head"][head_index]
                for index in range(len(table_value["schema"]["loop"])):
                    result_one[table_value["schema"]["loop"][index]] = loop_data[index]
                result.append(result_one)
        return result
