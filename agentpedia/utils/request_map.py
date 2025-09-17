# encoding:utf-8
import os
import pickle
import requests
from copy import deepcopy
from geopy.distance import geodesic

from agentpedia.config import Config

class RequestMap:

    def __init__(self, config, multiprocess=True):

        self.config = config
        self.api_keys = self.config.map_api_keys
        self.key_idx = -1
        self.num_keys = len(self.api_keys)
        self.url = "http://api.map.sdns.baidu.com/"
        
        self.map_data_path = 'map_data.pkl'
        if os.path.exists(self.map_data_path):
            self.map_data = pickle.load(open(self.map_data_path, 'rb'))
        else:
            self.map_data = {}

        self.map_data_dump_flag = not multiprocess

    def dump_data(self, k_1, k_2, data):

        # if os.path.exists(self.map_data_path):
        #     d = pickle.load(open(self.map_data_path, 'rb'))
        # else:
        #     d = {}
        # if k_1 not in d:
        #     d[k_1] = {}
        # d[k_1][k_2] = data
        # pickle.dump(d, open(self.map_data_path, 'wb'))
        if k_1 not in self.map_data:
            self.map_data[k_1] = {}
        self.map_data[k_1][k_2] = deepcopy(data)
        if self.map_data_dump_flag:
            self.collect_dump()
    
    def load_data(self, k_1, k_2):
        
        # d = pickle.load(open(self.map_data_path, 'rb'))
        # if k_1 in d and k_2 in d[k_1]:
        #     return d[k_1][k_2]
        # return None

        if k_1 in self.map_data and k_2 in self.map_data[k_1]:
            return self.map_data[k_1][k_2], True
        return None, False

    def collect_dump(self):

        pickle.dump(self.map_data, open(self.map_data_path, 'wb'))

    def get_api_key(self):

        self.key_idx += 1
        return self.api_keys[self.key_idx % self.num_keys]
    
    def api_call(self, url, params):

        num_times = 1
        while True:
            response = requests.get(url=url, params=params).json()
            if response['status'] != 0:
                if response['status'] in [4, 302, 201, 240]:
                    if num_times > 2 * self.num_keys:
                        print(f"API Error {response['status']} {response['message']} " + \
                            f"Time:{num_times}/{self.num_keys} Key:{self.key_idx % self.num_keys}")
                        break
                    num_times += 1
                    params['ak'] = self.get_api_key()
                elif response['status'] in [401]:
                    time.sleep(1)
                else:
                    print(response['status'], response['message'], params)
                    # raise Exception("API Error")
                    exit(-1)
            else:
                return response
        # raise Exception("API Error")
        exit(-1)

    def poi_match(self, query, region):

        url = self.url + "place/v2/suggestion"
        params = {
            "query": query,
            "region": region,
            "city_limit": "true",
            "output": "json",
            "ak": self.get_api_key(),
        }

        # if "poi_match" in self.map_data and f"{query}-{region}" in self.map_data["poi_match"]:
        #     return self.map_data["poi_match"][f"{query}-{region}"]
        check_data = self.load_data("poi_match", f"{query}-{region}")
        if check_data[1]:
            return check_data[0]
        res = self.api_call(url, params)['result']

        try:
            # if "poi_match" not in self.map_data:
            #     self.map_data["poi_match"] = {}
            # self.map_data["poi_match"][f"{query}-{region}"] = res[0]
            self.dump_data("poi_match", f"{query}-{region}", res[0])
            return res[0]
        except:
            print(f'POI match error: {query} - {res}')
            return None
    
    def poi_detail(self, uid):
        
        url = self.url + "place/v2/detail"
        params = {
            "uid": uid,
            "output": "json",
            "scope": "2",
            "ak": self.get_api_key(),
        }

        # if "poi_detail" in self.map_data and uid in self.map_data["poi_detail"]:
        #     return self.map_data["poi_detail"][uid]
        check_data = self.load_data("poi_detail", uid)
        if check_data[1]:
            return check_data[0]
        res = self.api_call(url, params)['result']
        # if "poi_detail" not in self.map_data:
        #     self.map_data["poi_detail"] = {}
        # self.map_data["poi_detail"][uid] = res
        self.dump_data("poi_detail", uid, res)

        return res
    
    def poi_search(self, query, region):

        # print("####", "行政区域搜索")

        url = self.url + "place/v2/search"
        tag = "旅游景点"
        params = {
            "query": query,
            "tag": tag,
            "region": region,
            # "city_limit": "true",
            "output": "json",
            "scope": "2", 
            "ak": self.get_api_key(),
        }

        check_data = self.load_data("poi_search", f"{query}-{region}")
        if check_data[1]:
            return check_data[0]
        else:
            # print(f"### map data not found: {query}-{region}")
            pass
        res = self.api_call(url, params)['results']

        try:
            self.dump_data("poi_search", f"{query}-{region}", res[0])
            return res[0]
        except:
            print(f'POI search error: {query} - {res}')
            return None

    def poi_search_nearby(self, query_key, location, radius, num_candidates, page_size=20):

        url = self.url + "place/v2/search"

        num_pages = num_candidates // page_size if num_candidates % page_size == 0 \
            else num_candidates // page_size + 1
        if query_key == "美食推荐":
            search_query = "中餐厅$外国餐厅$小吃快餐店"
            search_filter = "industry_type:cater|sort_name:overall_rating|sort_rule:0"
        elif query_key == "住宿推荐":
            search_query = "星级酒店$快捷酒店"
            search_filter = "industry_type:hotel|sort_name:overall_rating|sort_rule:0"
        else:
            raise Exception("Invalid query key")

        # if "poi_search_nearby" in self.map_data and f"{query_key}-{location}-{radius}-{num_candidates}" in self.map_data["poi_search_nearby"]:
        #     return self.map_data["poi_search_nearby"][f"{query_key}-{location}-{radius}-{num_candidates}"]
        check_data = self.load_data("poi_search_nearby", f"{query_key}-{location}-{radius}-{num_candidates}")
        if check_data[1]:
            return check_data[0]

        results = []
        for page_idx in range(num_pages):
            params = {
                "query": search_query,
                "location": location,
                "radius": radius,
                "radius_limit": "true", 
                "output": "json",
                "scope": "2", 
                "filter": search_filter, 
                "page_size": page_size, 
                "page_num": page_idx, 
                "ak": self.get_api_key(),
            }
            results.extend(self.api_call(url, params)['results'])
        res = results[:num_candidates]
        # if "poi_search_nearby" not in self.map_data:
        #     self.map_data["poi_search_nearby"] = {}
        # self.map_data["poi_search_nearby"][f"{query_key}-{location}-{radius}-{num_candidates}"] = res
        self.dump_data("poi_search_nearby", f"{query_key}-{location}-{radius}-{num_candidates}", res)
            
        return res

    def poi_search_region(self, query_key, regions, uid_set, num_candidates, page_size=20):

        url = self.url + "place/v2/search"

        if query_key == '景点':
            query = "旅游景点"
            tag = "旅游景点"
        else:
            raise Exception("Invalid query key")

        all_results = []
        for region in regions:
            results = []
            page_idx = 0
            while len(results) < num_candidates:
                params = {
                    "query": query,
                    "tag": tag,
                    "region": region,
                    "output": "json",
                    "scope": "2", 
                    "page_size": page_size, 
                    "page_num": page_idx, 
                    "ak": self.get_api_key(),
                }
                search_res = self.api_call(url, params)['results']
                if uid_set is not None:
                    search_res = [res for res in search_res if res["uid"] not in uid_set]
                results.extend(search_res)
                page_idx += 1
            all_results.extend(results)
        
        return all_results

    def route_plan(self, origin, destination, origin_uid, destination_uid):

        url = self.url + "direction/v2/transit"
        params = {
            "origin": origin,
            "destination": destination,
            "origin_uid": origin_uid, 
            "destination_uid": destination_uid,
            # "departure_date": "20241030", 
            # "departure_time": "05:00", 
            "tactics_incity": "0", # 0 推荐 1 少换乘 2 少步行 3 不坐地铁 4 时间短 5 地铁优先
            "tactics_intercity": "0", # 0 时间短 1 出发早 2 价格低
            "trans_type_intercity": "0", # 0 火车优先 1 飞机优先 2 大巴优先
            "ak": self.get_api_key(),
        }
        route_plan = self.api_call(url, params)['result']
        final_route_plan = {}
        if "taxi" in route_plan:
            final_route_plan["打车"] = {
                "路程": route_plan["taxi"]["distance"], 
                "预计时长": route_plan["taxi"]["duration"], 
                "描述": route_plan["taxi"]["remark"], 
            }
        if len(route_plan["routes"]) > 0:
            plan = route_plan["routes"][0]
            final_route_plan["公交"] = {
                "距离": plan["distance"], 
                "预计时长": plan["duration"], 
                "价格预算": plan["price"], 
                "路线": {},
            }
            for i, step in enumerate(plan["steps"]):
                scheme = step[0]
                final_route_plan["公交"]["路线"][f"第{i + 1}步"] = {
                    "路程": scheme["distance"], 
                    "预计时长": scheme["duration"], 
                    "指引": scheme["instructions"],
                }

        # for k, plan in enumerate(route_plan["routes"]):
        #     public_plan = {}
        #     public_plan["距离"] = plan["distance"]
        #     public_plan["预计时长"] = plan["duration"]
        #     public_plan["价格预算"] = plan["price"]
        #     public_plan["路线"] = {}
        #     for i, step in enumerate(plan["steps"]):
        #         public_plan["路线"][f"第{i + 1}步"] = {}
        #         for j, scheme in enumerate(step):
        #             public_plan["路线"][f"第{i + 1}步"][f"方案{j + 1}"] = {
        #                 "路程": scheme["distance"], 
        #                 "预计时长": scheme["duration"], 
        #                 "指引": scheme["instructions"],
        #             }
        #     final_route_plan["公交方案"][f"方案{k + 1}"] = public_plan
        if len(final_route_plan) == 0:
            final_route_plan["无"] = {}
        
        return {"类型": "交通", "方案": final_route_plan}

    def attraction_detail(self, poi_dict, get_loc=False, mode="search"):

        if mode == "match":
            try:
                matched_res = self.poi_match(poi_dict["名称"], poi_dict["detail"]["城市"])
            except:
                return None
            if matched_res is None:
                return None
            uid = matched_res["uid"]
            # print("###", poi_dict, uid)
            try:
                detailed_poi_dict = self.poi_detail(uid)
            except:
                try:
                    detailed_poi_dict = matched_res
                    if "name" in detailed_poi_dict:
                        poi_dict["名称"] = detailed_poi_dict["name"].strip()
                    if "address" in detailed_poi_dict:
                        poi_dict["detail"]["地址"] = detailed_poi_dict["address"].strip(),
                    poi_dict["detail"].update({
                        "经度": detailed_poi_dict["location"]["lng"],
                        "纬度": detailed_poi_dict["location"]["lat"],
                        # "uid": uid, 
                    })
                    return poi_dict
                except:
                    return None
            # print("###", poi_dict["名称"], detailed_poi_dict["name"], matched_res["tag"])
        elif mode == "search":
            detailed_poi_dict = self.poi_search(poi_dict["名称"], poi_dict["detail"]["城市"])
            # print("#", poi_dict["名称"], detailed_poi_dict["name"])
        else:
            raise ValueError("Invalid mode")
        
        try:
            poi_dict["名称"] = detailed_poi_dict["name"].strip()
            poi_dict["detail"].update({
                # "城市": detailed_poi_dict["city"].strip() if "city" in detailed_poi_dict else poi_dict["detail"]["城市"],
                "经度": detailed_poi_dict["location"]["lng"],
                "纬度": detailed_poi_dict["location"]["lat"],
                # "详情链接": detailed_poi_dict["detail_info"]["detail_url"] if "detail_url" \
                #     in detailed_poi_dict["detail_info"] else "",
                # "uid": uid, 
            })
            if "address" in detailed_poi_dict:
                poi_dict["detail"]["地址"] = detailed_poi_dict["address"].strip()
            # if "area" in detailed_poi_dict:
            #     poi_dict["detail"]["辖区"] = detailed_poi_dict["area"].strip()
            if "shop_hours" in detailed_poi_dict["detail_info"]:
                poi_dict["detail"]["开放时间"] = detailed_poi_dict["detail_info"]["shop_hours"]
            # if "price" in detailed_poi_dict["detail_info"]:
            #     poi_dict["detail"]["预算"] = detailed_poi_dict["detail_info"]["price"]
            # if "overall_rating" in detailed_poi_dict["detail_info"]:
            #     poi_dict["detail"]["评分"] = detailed_poi_dict["detail_info"]["overall_rating"]
        except:
            if get_loc:
                try:
                    if "name" in detailed_poi_dict:
                        poi_dict["名称"] = detailed_poi_dict["name"].strip()
                    if "address" in detailed_poi_dict:
                        poi_dict["detail"]["地址"] = detailed_poi_dict["address"].strip(),
                    poi_dict["detail"].update({
                        "经度": detailed_poi_dict["location"]["lng"],
                        "纬度": detailed_poi_dict["location"]["lat"],
                    })
                except:
                    return None
            else:
                return None

        return poi_dict

    def boarding_detail(self, poi_dict, center_poi_dict):

        matched_res = self.poi_match(poi_dict["名称"], poi_dict["detail"]["城市"])
        if matched_res is None:
            return None
        uid = matched_res["uid"]
        detailed_poi_dict = self.poi_detail(uid)

        poi_dict["名称"] = detailed_poi_dict["name"]
        poi_dict["detail"].update({
            "城市": detailed_poi_dict["city"],
            "辖区": detailed_poi_dict["area"],
            "地址": detailed_poi_dict["address"],
            "经度": detailed_poi_dict["location"]["lng"],
            "纬度": detailed_poi_dict["location"]["lat"],
            "开放时间": detailed_poi_dict["detail_info"]["shop_hours"] if "shop_hours" \
                in detailed_poi_dict["detail_info"] else poi_dict["detail"]["开放时间"],
            "预算": detailed_poi_dict["detail_info"]["price"] if "price" \
                in detailed_poi_dict["detail_info"] else poi_dict["detail"]["预算"],
            "评分": detailed_poi_dict["detail_info"]["overall_rating"] if "overall_rating" \
                in detailed_poi_dict["detail_info"] else "",
            "距离": f"{geodesic(
                (detailed_poi_dict["location"]["lat"], detailed_poi_dict["location"]["lng"]), 
                (center_poi_dict["detail"]["纬度"], center_poi_dict["detail"]["经度"])).m:.2f}米", 
            # "详情链接": detailed_poi_dict["detail_info"]["detail_url"] if "detail_url" \
            #     in detailed_poi_dict["detail_info"] else "",
            "uid": uid, 
        })

        return poi_dict

    def attraction_region_search(self, regions, query_key, uid_set, num_candidates=10):

        search_results = self.poi_search_region(
            query_key=query_key, 
            regions=regions, 
            uid_set=uid_set, 
            num_candidates=num_candidates
        )

        rec_candidates = []
        for idx, res in enumerate(search_results):
            rec_candidates.append(
                {
                    "名称": res["name"],
                    "detail": {
                        "城市": res["city"],
                        "辖区": res["area"],
                        "地址": res["address"],
                        "经度": res["location"]["lng"],
                        "纬度": res["location"]["lat"],
                        "开放时间": res["detail_info"]["shop_hours"] if "shop_hours" \
                            in res["detail_info"] else "",
                        "预算": res["detail_info"]["price"] if "price" \
                            in res["detail_info"] else "",
                        "评分": res["detail_info"]["overall_rating"] if "overall_rating" \
                            in res["detail_info"] else "",
                        "人气值": 0, 
                        # "详情链接": res["detail_info"]["detail_url"] if "detail_url" \
                        #     in res["detail_info"] else "",
                        "uid": res["uid"],
                    }
                }
            )

        return rec_candidates
    
    def attraction_nearby_boarding_search(self, center_poi_dict, query_key, radius, num_candidates):

        search_results = self.poi_search_nearby(
            query_key=query_key, 
            location=f"{center_poi_dict['detail']['纬度']},{center_poi_dict['detail']['经度']}", 
            radius=str(radius), 
            num_candidates=num_candidates
        )
        
        rec_candidates = []
        for idx, res in enumerate(search_results):
            rec_candidates.append(
                {
                    "名称": res["name"],
                    "detail": {
                        "城市": res["city"],
                        "辖区": res["area"],
                        "地址": res["address"],
                        "经度": res["location"]["lng"],
                        "纬度": res["location"]["lat"],
                        "开放时间": res["detail_info"]["shop_hours"] if "shop_hours" \
                            in res["detail_info"] else "",
                        "预算": res["detail_info"]["price"] if "price" \
                            in res["detail_info"] else "",
                        "评分": res["detail_info"]["overall_rating"] if "overall_rating" \
                            in res["detail_info"] else "",
                        "距离": f"{geodesic(
                            (res["location"]["lat"], res["location"]["lng"]), 
                            (center_poi_dict["detail"]["纬度"], center_poi_dict["detail"]["经度"])).m:.2f}米", 
                        # "详情链接": res["detail_info"]["detail_url"] if "detail_url" \
                        #     in res["detail_info"] else "",
                        "uid": res["uid"],
                    }
                }
            )

        return rec_candidates
