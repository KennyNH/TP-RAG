"""
格式化
"""
import json
import re
from collections import Counter

from agentpedia.logger.logger_config import get_logger


class Format:
    """
    格式化
    """

    def __init__(self, query):
        """初始化"""
        self.query = query
        self.logger = get_logger(query)
        self.table_err_cnt = 0

    def add_image(self, md_str, urls_list):
        """
            将图片添加到 markdown 字符串中，并返回修改后的字符串。
        如果图片路径在 urls_list 中，则会将其插入到第一个标题之前。
        否则，不会进行任何操作。
        
        Args:
            md_str (str): Markdown 格式的字符串，包含多个段落和标题。
            urls_list (List[Dict]): 包含图片路径的列表，每个元素是一个字典，包含键值对 {'poster': str}。
            如果图片路径在此列表中，则会将其插入到第一个标题之前。
        
        Returns:
            str: 返回一个新的 Markdown 字符串，包含所有原始段落和标题，且已经插入了图片。
            如果没有找到合适的图片路径，则不会进行任何操作。
        """
        content_items = self.split_markdown(md_str)
        self.logger.info("add_image before: %s", json.dumps(content_items, ensure_ascii=False))
        if not urls_list:
            self.logger.info("add_image not urls_list: %s", json.dumps(urls_list, ensure_ascii=False))
            return md_str
        res_items = []
        for item in content_items:
            indices = self.parse_all_format_indices(item)
            sorted_indices = self.most_frequent_index(indices)
            sorted_indices = [item[0] for item in sorted_indices]
            insert_path = ""
            for i in sorted_indices:
                if 0 <= i - 1 < len(urls_list) and 'poster' in urls_list[i - 1]:
                    insert_path = urls_list[i - 1]['poster']
                    break
            res_items.append(self.insert_image_before_first_subtitle(item, insert_path))
        md_str = '\n'.join(res_items)
        self.logger.info("add_image after: %s", md_str)
        return md_str

    @staticmethod
    def split_markdown(md_str):
        """
            将Markdown字符串根据一级标题进行拆分，返回一个列表。
        每个元素都是一个字符串，包含一个完整的Markdown部分（包括一级标题）。
        
        Args:
            md_str (str): Markdown字符串，包含一级标题。
        
        Returns:
            list[str]: 一个列表，其中每个元素都是一个字符串，包含一个完整的Markdown部分（包括一级标题）。
        """
        # 按照一级标题拆分
        parts = md_str.split('\n# ')
        # 补齐 标题
        items = ['# ' + part if i != 0 else part for i, part in enumerate(parts)]
        return items

    @staticmethod
    def parse_all_image_indices(response):
        """
        从模型输出中提取所有图片索引。
        """
        image_pattern = r'\[(\d+)\]'
        matches = re.findall(image_pattern, response)
        return [int(match) for match in matches]

    @staticmethod
    def parse_all_video_indices(response):
        """
        从模型输出中提取所有视频索引。
        """
        # 匹配两种格式：[1, 9] 和 [13, 0:00]
        video_pattern = r'\[(\d+), (\d+:\d+|\d+)\]'
        matches = re.findall(video_pattern, response)

        indices = []
        for match in matches:
            video_index = int(match[0])
            # 处理时间戳格式
            if ':' in match[1]:
                # 将时间戳转换为整数秒数
                minutes, seconds = map(int, match[1].split(':'))
                timestamp = minutes * 60 + seconds
            else:
                timestamp = int(match[1])
            indices.append((video_index, timestamp))

        return indices

    @staticmethod
    def parse_all_format_indices(response):
        """
        从模型输出中提取所有格式化后索引。
        """
        format_pattern = r'\^(\d+)\^'
        matches = re.findall(format_pattern, response)
        return [int(match) for match in matches]

    @staticmethod
    def most_frequent_index(indices):
        """
        传入索引列表，返回按出现次数排序的索引列表。
        """
        if not indices:
            return []

        counter = Counter(indices)
        sorted_indices = counter.most_common()

        return sorted_indices

    @staticmethod
    def insert_image_before_first_subtitle(content, image_path):
        """
        在第一个二级标题或第一个三级标题前面插入图片链接。
        """
        if image_path == "":
            return content
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith("## ") or line.startswith("### "):
                lines.insert(i, f"![图片]({image_path})")
                break
        return '\n'.join(lines)

    def format_content_result(self, content):
        """
        格式化内容结果
        """

        self.logger.info("format_content_result before: %s", content)
        content = self.replace_with_original_references(content)
        self.logger.info("format_content_result after: %s", content)

        return content

    def replace_with_original_references(self, input_string):
        """
            将输入字符串中的不标准格式（如：[1, 3]）转换成原始格式（如：^1^）。
        该函数会对输入字符串进行搜索和替换操作，并返回处理后的结果。
        
        Args:
            input_string (str): 需要进行格式转换的字符串。
        
        Returns:
            str: 经过格式转换后的字符串，其中不标准格式的部分已被转换成原始格式。
        """
        # 定义正则表达式模式，匹配不标准的格式
        pattern = re.compile(r'(\^?\[([0-9,: ]+)]\^?)')

        def replacer(match):
            first_number = match.group(2).split(',')[0].strip()
            return f'^{first_number}^'

        # 替换匹配项
        output_string = pattern.sub(replacer, input_string)

        # 移除连续重复的引用
        output_string = re.sub(r'(\^[0-9]+\^)(\s*\1)+', r'\1', output_string)

        # 后移引用
        output_string = self.move_marker_to_tail(output_string)

        # 移除错误引用格式
        output_string = re.sub(r"\^\[[^]]*]\^", "", output_string)
        output_string = re.sub(r"\^\[.*?]。", "。", output_string)
        output_string = re.sub(r"\^\[.*?]", "", output_string)
        output_string = re.sub(r"\[.*?]\^$", "", output_string)

        # 使用正则表达式匹配并替换每对引用之间的标点符号
        output_string = re.sub(r"(\^\d+\^)[、。，]+(?=\^\d+\^)", r"\1", output_string)

        self.logger.info(f"replace_with_original_references before: {input_string}")
        self.logger.info(f"replace_with_original_references after: {output_string}")
        return output_string

    @staticmethod
    def remove_unpaired(md_str):
        """
        移除不成对出现的**。
        """
        # 统计 '**' 的出现次数
        count = md_str.count('**')
        # 如果 '**' 的数量是奇数，说明有一个未配对
        if count % 2 != 0:
            # 找到最后一个 '**' 的位置
            last_index = md_str.rfind('**')
            # 删除最后一个未配对的 '**'
            md_str = md_str[:last_index] + md_str[last_index + 2:]

        return md_str

    @staticmethod
    def move_marker_to_tail(text):
        """
            将^1^移到句号后面。
        """
        # 使用正则表达式来匹配句号前紧挨着的^1^
        pattern = r'((?:\^\d+\^)+)(\。)'
        replacement = r'\2\1'

        # 替换所有匹配的模式，将^1^移到句号后面
        updated_text = re.sub(pattern, replacement, text)

        return updated_text

    def format_return_json(self, md_str, urls_list):
        """
        格式化返回的json数据，包括markdown和参考链接。
        """
        for item in urls_list:
            if 'abstract' in item:
                del item['abstract']
        reference_data = {
            'component': 'reference',
            'type': 'component',
            'data': {
                'value': urls_list
            }
        }

        blocks = self.format_return_md_list(md_str, urls_list)
        blocks.append(reference_data)
        blocks = self.move_repeat_player(blocks)
        blocks = self.move_repeat_reference(blocks)
        self.move_no_reference(blocks)
        # blocks = self.add_official(blocks)

        return blocks

    def add_official(self, blocks):
        """
            添加官方数据，如果查询结果在官方数据库中，则将其插入到第二个位置。
        参数 blocks (list[dict]) - 包含所有块的列表，每个块都是一个字典，包含以下键值对：
            component (str) - 组件类型，例如 'official'
            type (str) - 块类型，例如 'component'
            data (dict) - 包含以下键值对的字典：
                value (str) - 块的值，例如官方数据库 URL
        返回值 (list[dict]) - 与输入相同，但如果查询结果在官方数据库中，则将其插入到第二个位置。
        """
        official_dict = {}
        with open('agentpedia/data/official_data.txt', 'r') as f:
            for line in f.readlines():
                q, a = line.strip().split('\t')
                official_dict[q] = a
        if self.query in official_dict:
            official_url = official_dict[self.query]
            official_block = {
                'component': 'official',
                'type': 'component',
                'data': {
                    'value': official_url
                }
            }
            blocks.insert(1, official_block)
        return blocks

    def move_no_reference(self, blocks):
        """
        移除没有结果的引用。例如，没有结果的引用会被移除：【50】
        """
        reference_list = blocks[-1]['data']['value']
        format_pattern = r'\^(\d+)\^'
        reference_cnt = len(reference_list)
        for block in blocks[:-1]:
            if 'data' in block and 'value' in block['data']:
                original_value = block['data']['value']
                new_value = re.sub(format_pattern,
                                   lambda match: '' if int(match.group(1)) > reference_cnt else match.group(0),
                                   original_value)
                block['data']['value'] = new_value

    def move_repeat_player(self, blocks):
        """
        移除重复的视频、图片
        """
        self.logger.info("move_repeat_player before: %s", blocks)
        show_url_list = []
        res = []
        for block in blocks:
            if block['component'] in ['video', 'page_image']:
                if block['data']['value'] in show_url_list:
                    continue
                show_url_list.append(block['data']['value'])
            res.append(block)

        self.logger.info("move_repeat_player after: %s", res)
        return res

    def move_repeat_reference(self, blocks):
        """
        移除重复的引用
        """
        self.logger.info("move_repeat_reference before: %s", blocks)
        url_to_original_no = {}  # 记录每个URL的原始url_no
        unique_references = []  # 存储去重后的引用
        original_to_new_no = {}  # 原始url_no到新url_no的映射
        reference_list = blocks[-1]['data']['value']
        res = []
        new_no = 1  # 新的url_no从1开始

        for reference in reference_list:
            url = reference['url']
            original_no = reference['url_no']

            if url not in url_to_original_no:
                # 如果URL是新的，则添加到去重后的列表中
                url_to_original_no[url] = original_no
                reference['url_no'] = new_no
                unique_references.append(reference)

                # 更新映射字典
                original_to_new_no[original_no] = new_no

                # 增加新的url_no
                new_no += 1
            else:
                # 如果URL已经存在，更新映射字典
                original_to_new_no[original_no] = original_to_new_no[url_to_original_no[url]]
        self.logger.info("move_repeat_reference original_to_new_no: %s", original_to_new_no)

        def replace_match(match):
            original_no = int(match.group(1))
            # 获取新的引用编号
            new_no = original_to_new_no.get(original_no, original_no)
            return f'^{new_no}^'

        for block in blocks[:-1]:
            block['data']['value'] = re.sub(r'\^(\d+)\^', replace_match, block['data']['value'])
            if 'url_no' in block['data']:
                block['data']['url_no'] = original_to_new_no.get(block['data']['url_no'], block['data']['url_no'])
            res.append(block)

        blocks[-1]['data']['value'] = unique_references
        res.append(blocks[-1])
        self.logger.info("move_repeat_reference after: %s", res)
        return res

    @staticmethod
    def parse_markdown(md_str):
        """
            解析Markdown格式的字符串，返回一个列表，每个元素是一个字典，包含component、type和data三个键值对。
        Args:
            md_str (str): Markdown格式的字符串，可以包含多行。
        
        Returns:
            list[dict]: 解析后的列表，每个元素都是一个字典，包含component、type和'ata三个键值对。
        """
        # Split the input by lines for line-by-line parsing
        lines = md_str.split('\n')

        parsed_data = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                parsed_data.append({
                    'component': 'markdown',
                    'type': 'md_title',
                    'data': {
                        'value': line
                    }
                })
            else:
                parsed_data.append({
                    'component': 'markdown',
                    'type': 'md_content',
                    'data': {
                        'value': line
                    }
                })

        return parsed_data

    @staticmethod
    def is_markdown_table(md_content):
        """
            判断给定的 Markdown 内容是否为表格。
        参数：
            md_content (str) - Markdown 内容字符串。
        返回值（bool）：
            True，如果给定的 Markdown 内容包含一个或多个表格；False，如果不包含。
        """
        # 定义正则表达式匹配表格的结构，支持一个或多个换行符
        table_pattern = re.compile(r'(^\|.*\|$)+', re.MULTILINE)
        return table_pattern.search(md_content) is not None

    def merge_md_list(self, md_list):
        """
            合并多个 Markdown 列表，将相邻的 Markdown 内容合并为一个。
        如果有表格，则会将其与下一个 Markdown 内容合并为一个表格。
        
        Args:
            md_list (List[Dict]): 包含多个 Markdown 元素的列表，每个元素是一个字典，包含以下字段：
                - component (str, optional): 组件名称，默认为 None。
                - type (str): 元素类型，必须为 'md_content' 或 'md_table'。
                - data (Dict): 数据部分，包含以下字段：
                    - value (str): Markdown 内容。
        
        Returns:
            List[Dict]: 返回一个新的列表，包含了合并后的 Markdown 元素，每个元素都是一个字典，包含以下字段：
                - component (str, optional): 组件名称，默认为 None。
                - type (str): 元素类型，可能为 'md_content'、'md_table' 或 'md_content'。
                - data (Dict): 数据部分，包含以下字段：
                    - value (str): Markdown 内容。
        """
        res = []
        for item in md_list:
            if item['type'] != 'md_content':
                res.append(item)
                continue
            if item['data']['value'].startswith('![图片]'):
                continue
            if item['data']['value'].startswith('[观看视频]'):
                continue

            # 去掉前置和后置的换行符
            item['data']['value'] = item['data']['value'].rstrip()
            if not item['data']['value']:
                res.append({
                    "component": item['component'],
                    "type": "md_content",
                    "data": {
                        "value": ''
                    }
                })
                continue
            if self.is_markdown_table(item['data']['value']):
                if len(res) == 0 or res[-1]['type'] != 'md_table':
                    res.append({
                        "component": item['component'],
                        "type": "md_table",
                        "data": {
                            "value": item['data']['value']
                        }
                    })
                else:
                    res[-1]['data']['value'] += '\n' + item['data']['value']
            else:
                if len(res) > 0 and res[-1]['type'] == 'md_table':
                    if item['data']['value'].startswith('|') and item['data']['value'][-1] != '|':
                        res[-1]['data']['value'] += '\n' + item['data']['value'] + '|'
                        continue
                if len(res) == 0 or res[-1]['type'] != 'md_content':
                    res.append({
                        "component": item['component'],
                        "type": "md_content",
                        "data": {
                            "value": item['data']['value']
                        }
                    })
                else:
                    res[-1]['data']['value'] += '\n\n' + item['data']['value']
        res_list = []
        for item in res:
            if item['data']['value'] != "":
                res_list.append(item)
        return res_list

    @staticmethod
    def convert_table_to_list(table_str):
        """
            单行表格转为无序列表
        """
        # 分割表格的行
        lines = table_str.rstrip().split('\n')

        # 检查是否只有一行数据
        if len(lines) == 3:  # 表格包含标题行、分隔符行和数据行
            headers = lines[0].strip().split('|')
            data = lines[2].strip().split('|')

            # 去掉首尾的空格
            headers = [header.strip() for header in headers if header.strip()]
            data = [datum.strip() for datum in data if datum.strip()]

            # 判断最后一列是否为参考
            if headers[-1] in ["参考", "引用"]:
                reference = data.pop()  # 移除并获取最后一列的数据
                headers.pop()  # 移除最后一列的标题
                data[-1] += reference  # 将参考数据附加到前一列的数据后面

            # 构建无序列表
            result = []
            for header, datum in zip(headers, data):
                datum = datum.replace('<br>', '\n')
                items = datum.split('\n')
                if len(items) > 1:
                    items = ['  - ' + item[2:] if item.startswith('- ') else item for item in items]
                    datum = "\n" + '\n'.join(items)
                result.append(f"- {header}: {datum}")

            table_str = '\n'.join(result)
            table_str = table_str
            return table_str
        else:
            return table_str

    def format_return_md_list(self, md_str, urls_list):
        """
            格式化返回的 Markdown 列表，将其转换为 JSON 数组。
        参数：
            md_str (str) - Markdown 字符串，包含标题和内容。
            urls_list (list[str]) - 包含视频 URL 的列表，每个 URL 对应一个视频。
            返回值 (list[dict]) - 包含 JSON 数据的列表，每个元素都是一个包含 component、type 和 data 三个键的字典，用于在 WeCom 中显示。
        """

        parsed_data = self.parse_markdown(md_str)

        res_list = []
        need_add_part = []
        first_index, second_index = 0, 0

        for index, item in enumerate(parsed_data):
            # 移除不成对出现的**
            item['data']['value'] = self.remove_unpaired(item['data']['value'])
            if item['type'] == 'md_title':
                if item['data']['value'].startswith('# '):
                    if len(need_add_part) != 0:
                        need_add_part = self.add_video_precede_image(need_add_part, urls_list,
                                                                     second_index - first_index)
                        res_list.extend(need_add_part)
                        need_add_part = []
                        second_index = 0
                    first_index = index
                if second_index == 0 and (
                        item['data']['value'].startswith('## ') or item['data']['value'].startswith('### ')):
                    second_index = index
            need_add_part.append(item)
        need_add_part = self.add_video_precede_image(need_add_part, urls_list, second_index - first_index)
        res_list.extend(need_add_part)

        self.logger.info(f"merge_md_list before: {res_list}")
        res_list = self.merge_md_list(res_list)
        self.logger.info(f"merge_md_list after: {res_list}")

        res = []
        title_index = 0
        pre_title_list = ["简述", "文章目录", "阅读时间"]
        time_need_trans = False
        for item in res_list:
            if item['type'] == 'md_title' and item['data']['value'] == "# 阅读时间":
                time_need_trans = True
                continue
            if time_need_trans:
                item['type'] = 'md_time'
                time_need_trans = False
            if item["type"] == "md_content":
                # 内容中剔除引用前的竖线
                item["data"]["value"] = re.sub(r"\s*\|\s*(\^\d+\^)", r" \1", item["data"]["value"])
                # 内容中剔除行尾的竖线
                item["data"]["value"] = re.sub(r"\s*\|\s*$", "", item["data"]["value"])
            if item["type"] == "md_table":
                table_str = item["data"]["value"]
                table_str = self.del_empty_table(table_str)
                table_str = self.remove_table_reference(table_str)
                table_str = self.convert_table_to_list(table_str)
                if not self.is_markdown_table(table_str):
                    item["type"] = "md_content"
                    item["component"] = "markdown"
                item["data"]["value"] = table_str
            if item['type'] == 'md_title':
                if item["data"]["value"].startswith('# '):
                    if all(pre_title not in item['data']['value'] for pre_title in pre_title_list):
                        title_index += 1
                        item["data"]["id"] = title_index
                if '# 简述' in item["data"]["value"]:
                    continue
                if item['data']['value'] == "# 来源":
                    break
            res.append(item)

        return res

    def remove_table_reference(self, table_str):
        """
            移除表格最后一列的引用
        """
        lines = table_str.strip().split('\n')
        headers = lines[0].strip().split('|')[1:-1]
        refer_list = ['来源', '引用', '参考']
        if not any(refer_key in headers[-1].strip() for refer_key in refer_list):
            return table_str
        self.logger.info(f"remove_table_reference before: {table_str}")

        headers = headers[:-1]
        rows = [line.strip().split('|')[1:-1] for line in lines[2:]]
        # Process rows
        processed_rows = []
        for row in rows:
            if len(row) > 1:
                new_row = row[:-1]
                last_item = row[-1].strip()
                last_references = re.findall(r'\^\d+\^', last_item)
                # 存储需要追加的引用
                references_to_add = []

                for ref in last_references:
                    # 检查前面的列是否包含该引用
                    if not any(ref in item for item in new_row):
                        references_to_add.append(ref)

                # 将需要追加的引用添加到倒数第二列
                if references_to_add:
                    new_row[-1] += ''.join(references_to_add)
            else:
                new_row = row
            processed_rows.append(new_row)

        # Reconstruct the Markdown table
        new_markdown = '| ' + ' | '.join(headers) + ' |\n'
        new_markdown += '|' + '|'.join(['---' for _ in headers]) + '|\n'
        for row in processed_rows:
            new_markdown += '| ' + ' | '.join(row) + ' |\n'

        self.logger.info(f"remove_table_reference after: {new_markdown}")
        return new_markdown

    @staticmethod
    def format_url(item):
        """
            格式化输出引用的结果

        Args:
            item (dict): aiapi返回结果

        Returns:
            dict: 一个包含以下键值对的字典：
                - title (str): 标题。
                - url (str): URL。
                - abstract (str): 摘要。
                - create_time (int): 创建时间。
                - video_path (str, optional): 视频路径，如果存在。
                - poster (str, optional): 图片路径，如果存在。
        """
        res = {
            'title': item['title'],
            'url': item['url'],
            'publish_time': item.get("page_time", 0)
        }
        if 'sentence' in item:
            sentence = item['sentence']
            if isinstance(sentence, list):
                sentence = ' '.join(sentence)
            res['abstract'] = sentence
        video_info = item.get('video_info', {})
        if video_info:
            video_path = video_info.get('video_play_url', "")
            if video_path != '':
                res['video_path'] = video_path
            image_path = video_info.get('poster', "")
            if image_path != '':
                res['poster'] = image_path
            res['publish_time'] = video_info.get("pub_time", 0)
        image_path = ""
        page_images = item.get('page_images', {}).get('page_image', [])
        if len(page_images) > 0:
            image_path = page_images[0].get('image_url', "")
        if image_path != '':
            res['poster'] = image_path
        return res

    def add_video_precede_image(self, md_list, urls_list, insert_index):
        """
            在指定位置插入视频、图片，优先插入视频，如果找不到视频，则插入图片。
        参数：
            md_list (List[Dict]): markdown列表，包含每个元素为字典格式，包含"component"、"type"和"data"三个key，分别代表组件类型、类型和数据。
            urls_list (List[Dict]): url列表，包含每个元素为字典格式，包含"poster"和"video_url"两个key，分别代表图片路径和视频地址。
            insert_index (int): 插入位置索引，从0开始计算。
        返回值：
            List[Dict]: 更新后的markdown列表，包含每个元素为字典格式，包含"component"、"type"和"data"三个key，分别代表组件类型、类型和数据。
            如果没有找到图片或者视频，则不会插入任何内容。
        """
        self.logger.info("add_video_precede_image before: %s", json.dumps(md_list, ensure_ascii=False))
        value_list = [item['data']['value'] for item in md_list]
        md_str = '\n'.join(value_list)
        indices = self.parse_all_format_indices(md_str)
        sorted_indices = self.most_frequent_index(indices)
        insert_path, url_no = self.get_insert_path(sorted_indices, urls_list, 'video_path')
        insert_type = "video"

        if insert_path == "":
            insert_path, url_no = self.get_insert_path(sorted_indices, urls_list, 'poster')
            insert_type = "page_image"

        if insert_path == "":
            self.logger.info("add_video_precede_image after: %s", md_list)
            return md_list

        c_item = {
            "component": insert_type,
            "type": "component",
            "data": {
                "value": insert_path,
                "url_no": url_no
            }
        }
        if insert_index <= 0:
            md_list.append(c_item)
        else:
            md_list.insert(insert_index, c_item)
        self.logger.info("add_video_precede_image after: %s", md_list)
        return md_list

    def get_insert_path(self, sorted_indices, urls_list, key):
        """
        遍历最多引用索引，获取插入路径。

        Args:
            sorted_indices (list[tuple[str, int]]): 排序后的索引列表。
            urls_list (list[dict]): url列表。
            key (str): 键名。

        Returns:
            str: 插入路径。
        """
        for index in sorted_indices:
            i = index[0]
            cnt = index[1]
            if 0 <= i - 1 < len(urls_list) and key in urls_list[i - 1]:
                if self.is_black_list_url(urls_list[i - 1]['url']):
                    continue
                if cnt < 2:
                    return "", 0
                return urls_list[i - 1][key], i
        return "", 0

    @staticmethod
    def is_black_list_url(url):
        """
        判断是否为黑名单URL。
        """
        black_list = ["jingyan.baidu.com", "zhidao.baidu.com"]
        for i in black_list:
            if i in url:
                return True
        return False

    def del_empty_table(self, md_str):
        """
        删除表格中空列。
        """
        try:

            lines = md_str.strip().split('\n')

            header_cols = lines[0].count('|') - 1
            split_cols = lines[1].count('|') - 1
            content_cols = [line.count('|') - 1 for line in lines[2:]]

            if header_cols != split_cols or header_cols < max(content_cols):
                self.table_err_cnt += 1

            need_del = min(header_cols, split_cols) - max(content_cols)
            if need_del <= 0:
                return md_str

            header_list = lines[0].split('|')
            split_list = lines[1].split('|')
            del header_list[-2: -2 - need_del: -1]
            del split_list[-2: -2 - need_del: -1]
            md_str = '|'.join(header_list) + '\n' + '|'.join(split_list) + '\n' + '\n'.join(lines[2:])
        except Exception as e:
            self.logger.error(e)
            return md_str
        return md_str
