"""
哔哩哔哩爬虫函数包
"""
import os
import sys
import time
import random
import requests
import json
import hashlib
import urllib.parse
from typing import Tuple, Dict

# 获取当前脚本的绝对路径
current_script_path = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录，假设它是当前脚本目录的上两级
project_root = os.path.dirname(os.path.dirname(current_script_path))

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agentpedia.utils.bili_spider_util import WBIHandler
from agentpedia.logger.logger_config import get_logger

# 创建缓存目录
cache_dir = os.path.join(project_root, 'bilibili_data')
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# 共享的 WBIHandler 实例
wbi_handler = WBIHandler()


class BilibiliVideo:
    """
    Bilibili视频类，用于获取Bilibili视频的相关信息。
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0',
        'Referer': 'https://www.bilibili.com/'
    }

    def __init__(self, bvid: str):
        """
        初始化函数，用于创建一个BilibiliVideo实例。
        """
        self.bvid = bvid
        self.cid = None
        self.up_mid = None
        self.info_fetched = False
        self.summary_fetched = False
        self.user_card_fetched = False
        self.wbi_handler = wbi_handler
        self.info = {}
        self.summary = {}
        self.user_card = {}
        self.logger = get_logger()

    @classmethod
    def from_url(cls, url: str) -> 'BilibiliVideo':
        """
        根据给定的Bilibili视频链接，创建一个BilibiliVideo对象。

        Args:
            url (str): Bilibili视频链接。

        Returns:
            BilibiliVideo: 根据链接创建的BilibiliVideo对象。

        """

        if "bvid=" in url:
            bvid = url.split("bvid=")[1].split("/")[0]
        else:
            bvid = url.split("/video/")[1].split("/")[0]
        return cls(bvid)

    def _generate_cache_filename(self, url: str, params: Dict) -> str:
        """
        根据URL和参数生成缓存文件名。
        """
        sorted_params = json.dumps(params, sort_keys=True)
        unique_str = url + sorted_params
        self.logger.info('generate_cache_filename: ' + unique_str)
        hash_str = hashlib.md5(unique_str.encode('utf-8')).hexdigest()
        return os.path.join(cache_dir, f"{hash_str}.json")

    def _load_from_cache(self, url: str, params: Dict) -> Dict:
        """
        尝试从缓存加载数据，并检查其有效性。
        """
        cache_file = self._generate_cache_filename(url, params)
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('code') == 0:
                    return data
        return {}

    def _fetch_and_cache(self, url: str, signed_params: Dict, params: Dict = None) -> Dict:
        """
        进行实际请求并将结果保存到缓存。
        """
        if params is None:
            params = signed_params
        time.sleep(0.1 + random.uniform(-0.02, 0.05))
        response = requests.get(url, headers=self.headers, params=signed_params)
        data = response.json()
        self._save_to_cache(url, params, data)
        return data

    def _save_to_cache(self, url: str, params: Dict, data: Dict):
        """
        将数据保存到缓存。
        """
        cache_file = self._generate_cache_filename(url, params)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def get_info(self):
        """
        获取视频信息
        """
        if not self.info_fetched:
            url = f"https://api.bilibili.com/x/web-interface/view"
            params = {'bvid': self.bvid}
            data = self._load_from_cache(url, params)
            if not data:
                data = self._fetch_and_cache(url, params)
            self.cid = data['data']['cid']
            self.up_mid = data['data']['owner']['mid']
            self.info = data['data']
            self.info_fetched = True
        return self.info

    def get_summary(self) -> Dict:
        """
        获取 ai 总结
        """
        if not self.summary_fetched:
            if self.cid is None or self.up_mid is None:
                self.get_info()
            params = {
                'bvid': self.bvid,
                'cid': self.cid,
                'up_mid': self.up_mid,
                'web_location': '111.232'
            }
            url = "https://api.bilibili.com/x/web-interface/view/conclusion/get"
            data = self._load_from_cache(url, params)
            if len(data) > 0:
                self.summary = data
                self.summary_fetched = True
                return data
            img_key, sub_key = self.wbi_handler.get_wbi_keys()
            signed_params = self.wbi_handler.enc_wbi(
                params=params,
                img_key=img_key,
                sub_key=sub_key
            )
            data = self._fetch_and_cache(url, signed_params, params)
            self.summary = data
            self.summary_fetched = True
            return self.summary

    def get_user_card(self) -> Dict:
        """
        获取用户卡片信息
        """
        if not self.user_card_fetched:
            if self.up_mid is None:
                self.get_info()
            url = f"https://api.bilibili.com/x/web-interface/card"
            params = {'mid': self.up_mid}
            data = self._load_from_cache(url, params)
            if not data:
                data = self._fetch_and_cache(url, params)
            self.user_card = data
            self.user_card_fetched = True
        return self.user_card

    @staticmethod
    def parse_replies(result):
        """解析回复列表"""
        data = result.get("data", {})
        replies = data.get("replies", [])
        replies_list = []
        for reply in replies:
            comment_dict = {}
            content = reply.get("content", {})
            like = reply.get("like", 0)
            message = content.get("message", "")
            comment_dict["main_comment"] = message
            comment_dict["like"] = like
            reps = reply.get("replies", [])
            for rep in reps:
                cont = rep.get("content", {})
                mess = cont.get("message", "")
                comment_dict.setdefault("huifu_comment", [])
                comment_dict["huifu_comment"].append(mess)
            replies_list.append(comment_dict)
        return replies_list

    def get_replies(self, pn: int):
        """
        获取评论
        """
        if not self.info_fetched:
            self.get_info()
        aid = self.info['aid']
        cnt = 0
        offset = {'offset': ""}
        replies = []
        while cnt < pn:
            self.logger.info(f"正在获取第{cnt + 1}页评论")
            offset_str = json.dumps(offset).replace(' ', '')
            url = "https://api.bilibili.com/x/v2/reply/wbi/main"
            params = {
                'oid': aid,
                'type': 1,
                'mode': 3,
                'pagination_str': offset_str,
                'plat': 1,
                'seek_rpid': "",
                'web_location': '1315875'
            }
            data = self._load_from_cache(url, params)
            if len(data) > 0:
                replies_item = self.parse_replies(data)
            else:
                img_key, sub_key = self.wbi_handler.get_wbi_keys()
                signed_params = self.wbi_handler.enc_wbi(
                    params=params,
                    img_key=img_key,
                    sub_key=sub_key
                )
                data = self._fetch_and_cache(url, signed_params, params)
                replies_item = self.parse_replies(data)

            if len(replies_item) == 0:
                break
            replies.extend(self.parse_replies(data))
            data = data.get('data', {})
            cursor = data.get('cursor', {})
            pagination_reply = cursor.get('pagination_reply', {})
            offset_json_str = pagination_reply.get('next_offset', "")
            self.logger.info(f"offset_json_str: {offset_json_str}")
            if len(offset_json_str) == 0:
                break
            offset = {"offset": offset_json_str}
            cnt += 1
        self.logger.info(f"get_replies: {json.dumps(replies, indent=4, ensure_ascii=False)}")
        return replies


class BilibiliSearch:
    """
    BilibiliSearch类，用于搜索视频
    """
    headers = {
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0'
    }

    def __init__(self, keyword: str, page: int, page_size: int):
        """
        初始化函数，用于创建一个BilibiliSearch实例。
        """
        self.keyword = keyword
        self.wbi_handler = wbi_handler
        self.data = None
        self.fetched = False
        self.page = page
        self.page_size = page_size
        self.bvid_list = []
        self.logger = get_logger()

    def _generate_cache_filename(self, url: str, params: Dict) -> str:
        """
        根据URL和参数生成缓存文件名。
        """
        sorted_params = json.dumps(params, sort_keys=True)
        unique_str = url + sorted_params
        hash_str = hashlib.md5(unique_str.encode('utf-8')).hexdigest()
        return os.path.join(cache_dir, f"{hash_str}.json")

    def _load_from_cache(self, url: str, params: Dict) -> Dict:
        """
        尝试从缓存加载数据。
        """
        cache_file = self._generate_cache_filename(url, params)
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_to_cache(self, url: str, params: Dict, data: Dict):
        """
        将数据保存到缓存。
        """
        cache_file = self._generate_cache_filename(url, params)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def search(self):
        """
        根据关键词搜索视频
        """
        if not self.fetched:
            url = 'https://bilibili.com'
            response = requests.get(url, headers=self.headers)
            cookies = response.cookies
            img_key, sub_key = self.wbi_handler.get_wbi_keys()
            signed_params = self.wbi_handler.enc_wbi(
                params={
                    'keyword': self.keyword,
                    'page': self.page,
                    'page_size': self.page_size
                },
                img_key=img_key,
                sub_key=sub_key
            )
            url = "https://api.bilibili.com/x/web-interface/wbi/search/all/v2"
            data = self._load_from_cache(url, signed_params)
            if not data:
                time.sleep(0.1 + random.uniform(-0.02, 0.05))
                response = requests.get(url, headers=self.headers, cookies=cookies, params=signed_params)
                data = response.json()
                self._save_to_cache(url, signed_params, data)
            self.data = data
            self.fetched = True

    def get_bvid_list(self):
        """
        获取搜索到的视频的bvid
        """
        self.search()
        data = self.data.get("data", {})
        results = data.get("result", [])
        bvid_list = []
        for result in results:
            result_type = result.get("result_type", "")
            if result_type == "video":
                data_list = result.get("data", [])
                for dt in data_list:
                    bvid = dt.get("bvid", "")
                    bvid_list.append(bvid)
        self.bvid_list = bvid_list
        return self.bvid_list

    def get_videos_and_replies(self, replies_pn: int):
        """
        获取搜索到的视频的评论
        """
        self.get_bvid_list()
        videos_replies = []
        for bvid in self.bvid_list:
            try:
                video = BilibiliVideo(bvid=bvid)
                info = video.get_info()
                view = info['stat']['view']
                summary = video.get_summary()
                user_card = video.get_user_card()
                is_official = user_card['data']['card']['Official']['type'] >= 0
                up_fans = user_card['data']['card']['fans']
                replies_data = video.get_replies(replies_pn)
                videos_replies.append(
                    {
                        "summary": summary,  # ai总结
                        "is_official": is_official,  # 是否认证账号
                        "view": view,  # 播放量
                        "up_fans": up_fans,  # 粉丝数
                        "replies": replies_data[0:5]  # 评论数据
                    }
                )
            except Exception as e:
                self.logger.error(f"Error get_videos_and_replies: {e}")
        mock_replies = {"official_reply": [], 'user_reply': []}
        for video in videos_replies:
            if video["is_official"]:
                video_summary = video["summary"]
                summary_data = video_summary.get("data", {})
                model_result = summary_data.get("model_result", {})
                summary_str = model_result.get("summary", "")
                if summary_str != "":
                    mock_replies["official_reply"].append(summary_str)
            for reply in video["replies"]:
                mock_replies["user_reply"].append(reply)
        return mock_replies


# # 示例使用
# query = "雅诗兰黛"
# logger = get_logger(query)
# search = BilibiliSearch(query, 1, 40)
# print(search.get_bvid_list())
# replies_data = search.get_videos_and_replies(1)
# print(replies_data)
# with open("reply_"+ query+".json", 'w', encoding='utf-8') as f:
#     json.dump(replies_data, f, ensure_ascii=False, indent=4)