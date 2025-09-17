"""
格式化
"""
import ast
import json
import random
import re
from collections import Counter

import requests
import yaml
from agentpedia.config import Config
from agentpedia.utils.request_llm import RequestLLM


class ArticleProcessor:
    """Processes articles for multimedia optimization."""

    def __init__(self):
        """Initializes the ArticleProcessor class."""
        self.config = Config()
        self.request_llm = RequestLLM(self.config)

    def load_json_from_file(self, file_path):
        """
        从文件中读取JSON数据并返回一个Python对象。
        
        Args:
            file_path (str): JSON文件的路径。
        
        Returns:
            dict: 包含JSON文件中键值对的字典。
        
        Raises:
            IOError: 如果指定的文件无法打开。
            ValueError: 如果在解析JSON字符串时出错。
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def insert_after_match(self, json_array, search_string, new_element):
        """
        参考GPT-4o返回的字典，在匹配的字符串后插入新元素
        
        Args:
            json_array (list): 包含字典的列表，每个字典中应包含键'data'，其对应的值也是一个字典，包含键'value'
            search_string (str): 要搜索的字符串
            new_element (dict): 要插入的新元素
        
        Returns:
            list: 插入新元素后的列表
        
        在遍历json_array列表时，检查每个字典中'data'键对应的字典的'value'键是否包含search_string字符串，并且'value'键的值中包含'# '。
        如果满足条件，则在满足条件的元素之后插入new_element，并退出循环。
        返回插入新元素后的列表。
        
        """
        for i, item in enumerate(json_array):
            if search_string in item['data']['value'] and '# ' in item['data']['value']:
                json_array.insert(i + 1, new_element)
                break
        return json_array

    def multimedia_optimization(self, query_name, content, content1):
        """
        对多媒体内容进行优化处理。
        
        Args:
            query_name (str): 查询名称。
            content (dict): 包含多媒体信息的原始数据字典。
            content1 (str): 提示文本。
        
        Returns:
            None
        
        """
        file_name = query_name
        content = json.dumps(content, ensure_ascii=False)
        data_list = json.loads(content)

        matches, valid_url_nos, nums = self.extract_media_info(data_list)
        formatted_json_list = self.prepare_prompts(query_name, content1)
        selected_matches, selected_urls = self.select_indices(matches, valid_url_nos, nums)

        formatted_json_list.extend(self.format_image_url_payload(selected_matches))
        aktion_data = self.process_llm_response(self.request_llm.get_llm_result(formatted_json_list), matches)
        json_array = self.load_json_from_file(f'article_data/{file_name}.json')
        json_array = self.insert_images(json_array, selected_matches, selected_urls, aktion_data)

        with open(f'article_data/{file_name}.json', 'w', encoding='utf-8') as file:
            json.dump(json_array, file, indent=4, ensure_ascii=False)

        self.deduplicate_references(f'article_data/{file_name}.json')
        print("---插图策略优化完成---")

    def extract_media_info(self, data_list):
        """
        从给定的数据列表中提取媒体信息。
        
        Args:
            data_list (list): 包含媒体信息的列表，每个元素是一个字典，
                包含媒体的相关数据，如'data'字段。
        
        Returns:
            tuple: 一个包含三个元素的元组，分别为：
                - matches (list): 匹配到的海报URL列表。
                - valid_url_nos (list): 包含有效URL编号的列表。
                - nums (list): 匹配到的数字列表（具体含义取决于正则表达式的定义）。
        
        """
        pattern = re.compile(r'"poster":\s*"([^"]+)"')
        pattern1 = r"\^(\d+)\^"

        content = json.dumps(data_list, ensure_ascii=False)
        matches = re.findall(pattern, content)
        nums = re.findall(pattern1, content)

        valid_url_nos = [
            item['url_no']
            for data in data_list
            for item in data['data']['value']
            if 'poster' in item and 'url_no' in item
        ]
        return matches, valid_url_nos, nums

    def prepare_prompts(self, query_name, content1):
        """
        准备提示信息。
        
        Args:
            query_name (str): 查询名称。
            content1 (str): 要插入的内容。
        
        Returns:
            list: 包含单个字典的列表，字典包含类型为 'text' 的 'type' 键和包含提示信息的 'text' 键。
        
        """
        with open('agentpedia/config/article_prompt.yaml', 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        prompt_insert = data['multimedia_optimization_prompt']
        text = prompt_insert.format(query_name) + content1
        return [{"type": "text", "text": text}]

    def select_indices(self, matches, valid_url_nos, nums):
        """
        从给定的匹配结果中随机选择指定数量的索引，优先从有效的URL编号中选取，并返回对应的匹配结果和URL编号。
        
        Args:
            matches (list): 匹配结果列表，每个元素代表一个匹配项。
            valid_url_nos (list): 有效的URL编号列表。
            nums (list): 需要从匹配结果中选择的索引对应的数值列表，数值与匹配结果列表中的元素对应。
        
        Returns:
            tuple:
                selected_matches (list): 根据选择条件筛选出的匹配结果列表。
                selected_urls (list): 与selected_matches对应的URL编号列表。
        
        """
        index_list = list(range(len(matches)))
        counter = Counter(nums)
        top_n = [int(item[0]) for item in counter.most_common(len(counter))]
        intersection = [x for x in top_n if x in valid_url_nos]
        intersection_sample = [valid_url_nos.index(x) for x in intersection]

        index_list_sample = [x for x in index_list if x not in intersection_sample[:5]]
        selected_indices = random.sample(index_list_sample, 5) + intersection_sample[:5]
        selected_indices.sort()

        selected_matches = [matches[i] for i in selected_indices]
        selected_urls = [valid_url_nos[i] for i in selected_indices]
        return selected_matches, selected_urls

    def format_image_url_payload(self, selected_matches):
        """
        将选定的匹配项格式化为包含图片URL的载荷列表。
        
        Args:
            selected_matches (List[str]): 包含选定图片URL的列表。
        
        Returns:
            List[Dict[str, Union[str, Dict[str, str]]]]: 包含格式化后的图片URL载荷的列表，
                每个载荷为一个字典，包含字段"type"和"image_url"，其中"type"固定为"image_url"，
                "image_url"为另一个字典，包含字段"url"，值为从selected_matches中选定的图片URL。
        
        """

        return [{"type": "image_url", "image_url": {"url": url}} for url in selected_matches]

    def process_llm_response(self, aktion, matches):
        """
        处理LLM响应。
        
        Args:
            aktion (str): 包含要处理的数据的字符串。
            matches (list): 匹配项列表。
        
        Returns:
            tuple: 包含两个列表的元组，第一个列表是处理后的键，第二个列表是处理后的值。
                键列表：处理后的键列表。
                值列表：处理后的值列表，其中每个值都是原始值减去10后与匹配项列表长度的最小值之间的差值。
        
        """
        data_str = '{' + re.findall(r'\{([^}]*)\}', aktion)[0] + '}'

        if data_str.endswith(','):
            data_str = data_str[:-1]

        data_dict = ast.literal_eval(data_str)
        keys = list(data_dict.keys())
        values = list(data_dict.values())
        values = [int(value) - 10 + min(10, len(matches)) for value in values]

        return keys, values

    def insert_images(self, json_array, selected_matches, selected_urls, aktion_data):
        """
        在json_array中添加图片元素
        
        Args:
            json_array (list): 包含页面组件的列表
            selected_matches (list): 与selected_urls相对应的匹配结果列表
            selected_urls (list): 选中的图片URL列表
            aktion_data (tuple): 包含键值对的元组，键为组件标识，值为对应的URL在selected_urls中的索引
        
        Returns:
            list: 更新后的json_array列表，包含新添加的图片元素
        
        """
        keys, values = aktion_data
        url_no = 0

        json_array = [item for item in json_array if item.get('component') != 'page_image']

        for mem, url in zip(keys, values):
            new_element = {
                "component": "page_image",
                "type": "component",
                "data": {
                    "url_no": selected_urls[values[url_no] - 1],
                    "value": selected_matches[values[url_no] - 1]
                }
            }
            if '封面' in mem:
                json_array.insert(0, new_element)
            else:
                search_string = mem
                json_array = self.insert_after_match(json_array, search_string, new_element)
            url_no += 1

        return json_array

    def deduplicate_references(self, file_path):
        """
        从文件中加载JSON数据，去重视频组件并保存结果
        
        Args:
            file_path (str): JSON文件的路径
        
        Returns:
            None
        
        将指定路径的JSON文件加载到内存中，遍历文件中的每个组件，对于类型为'video'的组件，根据'url_no'字段去重，
        并将去重后的结果以及类型为'reference'的组件保存到新的JSON文件中。
        
        新文件的命名格式为'output_' + 原文件名，并保存在'article_data/'目录下。
        """
        data = self.load_json_from_file(file_path)
        video_components = {}
        result = [data[0]]

        for item in data[1:]:
            if item.get('component') == 'video':
                url_no = item.get('data', {}).get('url_no')
                if url_no is not None:
                    video_components[url_no] = item

        for item in data[1:]:
            component_type = item.get('component')
            url_no = item.get('data', {}).get('url_no')

            if component_type == 'reference':
                result.append(item)
                continue

            if component_type == 'video':
                if url_no in video_components:
                    result.append(video_components[url_no])
                continue

            if url_no not in video_components:
                result.append(item)

        result_json = json.dumps(result, indent=4, ensure_ascii=False)
        file_name = file_path.split("/")[-1]
        output_file_name = 'output_' + file_name
        with open(f'article_data/{output_file_name}', 'w', encoding='utf-8') as file:
            file.write(result_json)