import json
import argparse
from functools import reduce
from hashlib import md5
import urllib.parse
import time
import requests
import os
import sys

# 获取当前脚本的绝对路径
current_script_path = os.path.dirname(os.path.abspath(__file__))

# 获取项目根目录，假设它是当前脚本目录的上两级
project_root = os.path.dirname(os.path.dirname(current_script_path))

# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agentpedia.utils.spider_bili_comment import RequestBiliComment

mixinKeyEncTab = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49,
    33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40,
    61, 26, 17, 0, 1, 60, 51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11,
    36, 20, 34, 44, 52
]

payload = {}
headers = {
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0'
}

class RequestBiliSearch:
    def __init__(self):
        self.bilicomment = RequestBiliComment()

    def getMixinKey(self, orig: str):
        '对 imgKey 和 subKey 进行字符顺序打乱编码'
        return reduce(lambda s, i: s + orig[i], mixinKeyEncTab, '')[:32]

    def encWbi(self, params: dict, img_key: str, sub_key: str):
        '为请求参数进行 wbi 签名'
        mixin_key = self.getMixinKey(img_key + sub_key)
        curr_time = round(time.time())
        params['wts'] = curr_time  # 添加 wts 字段
        params = dict(sorted(params.items()))  # 按照 key 重排参数
        # 过滤 value 中的 "!'()*" 字符
        params = {
            k: ''.join(filter(lambda chr: chr not in "!'()*", str(v)))
            for k, v
            in params.items()
        }
        query = urllib.parse.urlencode(params)  # 序列化参数
        wbi_sign = md5((query + mixin_key).encode()).hexdigest()  # 计算 w_rid
        params['w_rid'] = wbi_sign
        return params

    def getWbiKeys(self):
        '获取最新的 img_key 和 sub_key'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
            'Referer': 'https://www.bilibili.com/'
        }
        resp = requests.get('https://api.bilibili.com/x/web-interface/nav', headers=headers)
        resp.raise_for_status()
        json_content = resp.json()
        img_url: str = json_content['data']['wbi_img']['img_url']
        sub_url: str = json_content['data']['wbi_img']['sub_url']
        img_key = img_url.rsplit('/', 1)[1].split('.')[0]
        sub_key = sub_url.rsplit('/', 1)[1].split('.')[0]
        return img_key, sub_key

    def parse_result(self, data):
        data = data.get("data", {})
        results = data.get("result", [])
        bvid_url_list = []
        for result in results:
            result_type = result.get("result_type", "")
            if result_type == "video":
                data_list = result.get("data", [])
                for dt in data_list:
                    arcurl = dt.get("arcurl", "")
                    bvid = dt.get("bvid", "")
                    bvid_url_list.append([bvid, arcurl])
        return bvid_url_list

    def get_search_comment(self, keyword, page, page_size, pn):
        url = 'https://bilibili.com'
        response = requests.request("GET", url, headers=headers, data=payload)
        cookies = response.cookies
        img_key, sub_key = self.getWbiKeys()
        signed_params = self.encWbi(
            params={
                'keyword': keyword,
                'page': page,
                'page_size': page_size
            },
            img_key=img_key,
            sub_key=sub_key
        )
        query_params = urllib.parse.urlencode(signed_params)
        url = "https://api.bilibili.com/x/web-interface/wbi/search/all/v2?" + query_params
        response = requests.request("GET", url, headers=headers, data=payload, cookies=cookies)
        data = json.loads(response.text)
        bvid_url_list = self.parse_result(data)
        all_comment_list = []
        for bvid_url in bvid_url_list[:20]:
            time.sleep(0.1)
            comment_list = self.bilicomment.get_relay(bvid_url[0], pn)
            all_comment_list += comment_list[:20]
        # 使用 'age' 字段降序排序
        comment_sorted_list = sorted(all_comment_list, key=lambda x: x['like'], reverse=True)
        return comment_sorted_list[:20]

# print(get_search_comment("苹果15灵动岛怎么用", 1, 40, 2))