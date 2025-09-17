"""
哔哩哔哩爬虫工具包
"""
import copy
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


class WBIHandler:
    def __init__(self):
        """
            Initialization function for the ImageInfo class.
        初始化ImageInfo类的函数。
        
        Args:
            None.
        
        Returns:
            None.
            Initializes the img_key and sub_key attributes to empty strings.
        """
        self.img_key = ""
        self.sub_key = ""

    """哔哩哔哩 wbi 签名工具"""

    @staticmethod
    def get_mixin_key(orig: str):
        """对 imgKey 和 subKey 进行字符顺序打乱编码"""
        return reduce(lambda s, i: s + orig[i], mixinKeyEncTab, '')[:32]

    @staticmethod
    def enc_wbi(params: dict, img_key: str, sub_key: str):
        """为请求参数进行 wbi 签名"""
        mixin_key = WBIHandler.get_mixin_key(img_key + sub_key)
        curr_time = round(time.time())
        params = copy.deepcopy(params)
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

    def get_wbi_keys(self) -> tuple[str, str]:
        """获取最新的 img_key 和 sub_key"""
        if self.img_key and self.sub_key:
            return self.img_key, self.sub_key
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/58.0.3029.110 Safari/537.3',
            'Referer': 'https://www.bilibili.com/'
        }
        resp = requests.get('https://api.bilibili.com/x/web-interface/nav', headers=headers)
        resp.raise_for_status()
        json_content = resp.json()
        img_url: str = json_content['data']['wbi_img']['img_url']
        sub_url: str = json_content['data']['wbi_img']['sub_url']
        img_key = img_url.rsplit('/', 1)[1].split('.')[0]
        sub_key = sub_url.rsplit('/', 1)[1].split('.')[0]
        self.img_key = img_key
        self.sub_key = sub_key
        return img_key, sub_key
