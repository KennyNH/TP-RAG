import json
import argparse
from functools import reduce
from hashlib import md5
import urllib.parse
import time
import requests

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


class RequestBiliComment:
    def __init__(self):
        self.config = ""

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

    def get_info(self, bvid):
        url = "https://api.bilibili.com/x/web-interface/view?bvid=" + bvid
        response = requests.request("GET", url, headers=headers, data=payload)
        data = json.loads(response.text)
        return data

    def parse_result(self, result):
        data = result.get("data", {})
        replies = data.get("replies", [])
        comment_list = []
        for replie in replies:
            comment_dict = {}
            content = replie.get("content", {})
            like = replie.get("like", 0)
            message = content.get("message", "")
            comment_dict["main_comment"] = message
            comment_dict["like"] = like
            reps = replie.get("replies", [])
            for rep in reps:
                cont = rep.get("content", {})
                mess = cont.get("message", "")
                comment_dict.setdefault("huifu_comment", [])
                comment_dict["huifu_comment"].append(mess)
            comment_list.append(comment_dict)
        return comment_list

    def get_relay(self, bvid, pn):
        info = self.get_info(bvid)
        aid = info['data']['aid']
        img_key, sub_key = self.getWbiKeys()
        cnt = 0
        offset = {'offset': ""}
        comment_list = []
        while cnt < pn:
            offset_str = json.dumps(offset).replace(' ', '')
            signed_params = self.encWbi(
                params={
                    'oid': aid,
                    'type': 1,
                    'mode': 3,
                    'pagination_str': offset_str,
                    'plat': 1,
                    'seek_rpid': "",
                    'web_location': '1315875'
                },
                img_key=img_key,
                sub_key=sub_key
            )
            query_params = urllib.parse.urlencode(signed_params)
            url = "https://api.bilibili.com/x/v2/reply/wbi/main?" + query_params
            response = requests.request("GET", url, headers=headers, data=payload)
            data = json.loads(response.text)
            comment = self.parse_result(data)
            comment_list += comment
            try:
                offset_json_str = data['data']['cursor']['pagination_reply']['next_offset']
            except Exception as e:
                break
            if len(offset_json_str) == 0:
                break
            offset = {"offset": offset_json_str}
            cnt += 1
        return comment_list

    def get_bili_comment(self, url, pn):
        if "bvid=" in url:
            bvid = url.split("bvid=")[1].split("/")[0]
        else:
            bvid = url.split("/video/")[1].split("/")[0]
        comment_list = self.get_relay(bvid, pn)
        return comment_list

# bilicomment = RequestBiliComment()

# url = "https://www.bilibili.com/video/BV1Gs4y1f7oP/"
# pn = 5
# print(bilicomment.get_bili_comment(url, pn))