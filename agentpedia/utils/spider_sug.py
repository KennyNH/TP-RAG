import codecs
import os
import subprocess
import json
from datetime import datetime
import time

from agentpedia.context.article import ArticleContext
from agentpedia.config import Config
from agentpedia.intent import IntentGenerator
from agentpedia.logger.logger_config import get_logger
from agentpedia.web.local_cache import load_from_cache, save_to_cache


def get_ip_address():
    try:
        # 使用 subprocess.check_output 执行命令并获取输出
        ip_address = subprocess.check_output(['hostname', '-i']).strip()
        # 将输出转换为字符串（Python 2 中获取的是字节串，需要解码）
        ip_address = ip_address.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print("Command failed with error:", e)
        ip_address = None
    except Exception as e:
        print("An error occurred:", e)
        ip_address = None

    return ip_address


class SpiderSug:
    def __init__(self, config: Config):
        """
        初始化函数，用于创建对象并设置其属性。
        
        Args:
            config (Config): 配置文件对象，包含所需的各种配置信息。
        
        Returns:
            None
        
        """
        self.config = config
        self.ip = get_ip_address()
        self.sug_id_dict = eval(self.config.sug_id_dict)
        if self.ip in self.sug_id_dict:
            self.sug_username = self.sug_id_dict[self.ip]["sug_username"]
            self.xhs_task_id = self.sug_id_dict[self.ip]["xhs_task_id"]
            self.dy_task_id = self.sug_id_dict[self.ip]["dy_task_id"]
        else:
            # fake ip数据，启用sug是需要注意
            self.sug_username = "yihuixiong"
            self.xhs_task_id = 33700
            self.dy_task_id = 33700
        self.batch_file_path = self.config.batch_file_path

    def split_list(self, input_list, length):
        """
        将输入列表按照指定长度切割成多个子列表。
        
        Args:
            input_list (list): 需要切割的输入列表。
            length (int): 每个子列表的长度。
        
        Returns:
            list: 由多个子列表组成的列表。
        
        """
        return [input_list[i:i + length] for i in range(0, len(input_list), length)]

    def get_sug(self, query):
        """
            获取关键词的建议，并返回前五个结果。
        参数：
            query (str) - 需要查询的关键词。
        返回值（list）:
            res_sug_list (list) - 包含前五个建议关键词的列表，如果没有建议则为空列表。
        """
        cache_params = {"query": query}
        if self.config.from_cache:
            cache_str = load_from_cache('get_sug', cache_params)
            if cache_str:
                sug_list = json.loads(cache_str)
                if len(sug_list) > 0:
                    return sug_list

        data = {'username': self.sug_username, 'data': [query], 'taskId': self.dy_task_id}
        # 抖音sug请求数据
        subprocess.run(["curl", "-X", "POST", "-H", 'Content-Type: application/json', '-d', json.dumps(data),
                        "http://10.11.11.39:8686/query/stream_request"])
        # 小红书sug请求数据
        # data['taskId'] = self.xhs_task_id
        # subprocess.run(["curl", "-X", "POST", "-H", 'Content-Type: application/json', '-d', json.dumps(data),
        #                 "http://10.11.11.39:8686/query/stream_request"])
        time.sleep(100)
        query_task_sug_dict = self.parse_result(self.batch_file_path)
        res_sug_list = []
        dy_query_task = query + "||" + str(self.dy_task_id)
        # xhs_query_task = query + "||" + str(self.xhs_task_id)
        if dy_query_task in query_task_sug_dict:
            res_sug_list += query_task_sug_dict[dy_query_task][:3]
        # if xhs_query_task in query_task_sug_dict:
        #     res_sug_list += query_task_sug_dict[xhs_query_task][:5]
        if self.config.from_cache:
            save_to_cache('get_sug', cache_params, json.dumps(res_sug_list))
        return res_sug_list

    def get_batch_sug(self, query_list):
        """
            获取批量的关键词建议，每次请求10个关键词，如果不足则继续请求，直到所有关键词都得到了建议。
        返回值为一个字典，包含所有关键词和对应的建议列表，如果没有建议则为空列表。
        
        Args:
            query_list (List[str]): 需要获取建议的关键词列表，每个元素是一个字符串。
        
        Returns:
            Dict[str, List[str]]: 返回一个字典，key是原始关键词，value是对应的建议列表，如果没有建议则为空列表。
        """
        query_len = len(set(query_list))
        query_list = self.split_list(query_list, 200)
        for q_list in query_list:
            data = {}
            data['username'] = self.sug_username
            data['data'] = q_list
            # 抖音sug请求数据
            data['taskId'] = self.dy_task_id
            subprocess.run(["curl", "-X", "POST", "-H", 'Content-Type: application/json', '-d', json.dumps(data),
                            "http://10.11.11.39:8686/query/stream_request"])
            time.sleep(20)
        sug_dict_len = 0
        query_task_sug_dict = {}
        while sug_dict_len < int(query_len * 0.98 + 1):
            time.sleep(30)
            query_task_sug_dict = self.parse_result(self.batch_file_path)
            sug_dict_len = len(query_task_sug_dict)
        res_query_task_sug_dict = dict()
        for query_list in query_task_sug_dict:
            q_list = query_list.split("||")
            if len(q_list) > 0:
                res_query_task_sug_dict[q_list[0]] = query_task_sug_dict[query_list][:3]
        return res_query_task_sug_dict

    def get_sug_change_q_batch(self, query_list):
        """
        根据提供的搜索查询（query），生成需求目录

        参数:
            query (str): 用户的搜索查询字符串。

        返回:
            str: prompt 或者 生成的需求目录prompt
        """
        if self.config.need_sug:
            res_query_task_sug_dict = self.get_batch_sug(query_list)
        else:
            res_query_task_sug_dict = {}
        change_querys_dict = {}
        for query in query_list:
            context = ArticleContext(query)
            intent_generator = IntentGenerator(query, self.config, context)
            change_query_list = intent_generator.get_change_query()
            change_querys_dict[query] = change_query_list
        return res_query_task_sug_dict, change_querys_dict

    def parse_douyin(self, data, query_task_sug_dict):
        """
            解析抖音任务，将获取到的建议内容添加到query_task_sug_dict中
        参数：
            data (dict) - 包含用户信息、任务ID和未解析的数据的字典，格式如下：
                {
                    "userdata": "",  # 用户信息，str类型，可选，默认为""
                    "task_id": 0,  # 任务ID，int类型，必须，默认为0
                    "unparsed_data": ""  # 未解析的数据，str类型，必须，默认为""
                }
            查询任务建议列表（sug_list），并将其添加到query_task_sug_dict中，格式如下：
                {
                    "userdata || task_id": ["建议1", "建议2", ...]  # 用户信息 || 任务ID && 建议列表，str类型，必须
                }
            返回值：
                dict - 更新后的query_task_sug_dict，包含了新添加的建议内容，格式与输入的query_task_sug_dict相同
        异常情况：
            当解析过程中出现异常时，会打印错误信息，但不影响函数的正常运行，返回的结果依然是更新后的query_task_sug_dict
        """
        userdata = data.get("userdata", "")
        task_id = data.get("task_id", 0)
        unparsed_data = data.get("unparsed_data", "")
        unparsed_data = json.loads(unparsed_data)
        sug_list_all = []
        try:
            for up_data in unparsed_data:
                sug_data = up_data.get("data", "")
                for s_data in sug_data:
                    body = s_data.get("body", "")
                    body = json.loads(body)
                    sug_list = body.get("sug_list", [])
                    for sug in sug_list:
                        content = sug.get("content", "")
                        # if userdata in content:
                        sug_list_all.append(content)
        except Exception as e:
            print(e)
        query_task_sug_dict[userdata + "||" + str(task_id)] = sug_list_all
        return query_task_sug_dict

    def parse_xhs(self, data, query_task_sug_dict):
        """
            解析XHS响应数据，获取建议列表并存入字典中
        
        Args:
            data (dict): XHS响应数据，包括用户数据、任务ID、未解析的数据等信息，格式如下：
                {
                    "userdata": "",  # 用户数据，str类型，可选参数，默认为空字符串
                    "task_id": 0,  # 任务ID，int类型，可选参数，默认为0
                    "unparsed_data": ""  # 未解析的数据，str类型，必须参数
                }
            query_task_sug_dict (dict): 一个字典，用于存放每个任务对应的建议列表，格式如下：
                {
                    "userdata1||task_id1": ["建议1", "建议2", ...],  # 用户数据 || 任务ID：建议列表，str类型
                    "userdata2||task_id2": ["建议3", "建议4", ...],  # 用户数据 || 任务ID：建议列表，str类型
                    ...
                }
        
        Returns:
            dict: 返回更新后的query_task_sug_dict字典，包含了每个任务对应的建议列表
        
        Raises:
            无
        """
        userdata = data.get("userdata", "")
        task_id = data.get("task_id", 0)
        unparsed_data = data.get("unparsed_data", "")
        try:
            unparsed_data = json.loads(unparsed_data)
            sug_list_all = []
            sug_data = unparsed_data[0].get("data", "")
            try:
                s_data = sug_data[0]
                body = s_data.get("body", "")
                body = json.loads(body)
                data = body.get("data", "")
                data = json.loads(data)[0]
                sug_items = data.get("data", "")[0]["body"]["data"]["sug_items"]
                for item in sug_items:
                    text = item.get("text", "")
                    # if userdata in text:
                    sug_list_all.append(text)
                query_task_sug_dict[userdata + "||" + str(task_id)] = sug_list_all
            except Exception as e:
                print(e)
        except Exception as e:
            print(e)
        return query_task_sug_dict

    def parse_result(self, file_path=""):
        """
            解析结果，返回一个字典，包含每条任务的词库建议列表。
        如果file_path为空，则默认使用当天日期作为文件名。
        
        Args:
            file_path (str, optional): 结果文件路径，默认为"". Defaults to "".
        
        Returns:
            dict: 包含每条任务的词库建议列表，格式为{task_id: [suggestion1, suggestion2, ...], ...}.
        """
        result_path = "agentpedia/stream_task/log/"
        if file_path == "":
            # 获取当前日期和时间
            now = datetime.now()
            formatted_date = now.strftime('%Y-%m-%d')
            file_path = "result_" + str(formatted_date)
        result_path = os.path.join(result_path, file_path)
        # self.logger.info(f"sug_result_path: {result_path}")
        query_task_sug_dict = {}
        with codecs.open(result_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                try:
                    word = line.rstrip()
                    data = json.loads(word)
                    task_id = data.get("task_id", 0)
                    if str(task_id) == str(self.dy_task_id):
                        query_task_sug_dict = self.parse_douyin(data, query_task_sug_dict)
                    else:
                        query_task_sug_dict = self.parse_xhs(data, query_task_sug_dict)
                except Exception as e:
                    print(e)
        return query_task_sug_dict

# config = Config()
# spider_sug = SpiderSug(config)
# query_list = []
# input_file = sys.argv[1]
# with codecs.open(input_file, 'r', encoding='utf-8') as fin:
#     for line in fin:
#         query = line.rstrip()
#         query_list.append(query)
# print(query_list)
# print(spider_sug.get_batch_sug(query_list))
