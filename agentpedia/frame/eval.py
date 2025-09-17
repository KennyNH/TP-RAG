import os
import re
import time
import json
import math
import codecs
import bisect
import pickle
import argparse
import threading
from tqdm import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
from copy import deepcopy
from scipy.stats import entropy
from collections import Counter
from geopy.distance import geodesic
from datetime import datetime, timedelta
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing
from fuzzywuzzy import process, fuzz
from scipy.stats import kendalltau, spearmanr
from scipy.interpolate import make_interp_spline

from agentpedia.config import Config
from agentpedia.utils.request_llm import RequestLLM
# from agentpedia.utils.request_map import RequestMap
from agentpedia.logger.logger_config import get_logger

plt.rcParams.update({
    "text.usetex": True,        # 启用LaTeX渲染 （1）https://equationliu.github.io/2019-11-14-texlive/ （2）https://docs.pingcode.com/ask/935038.html
    "font.family": 'sans-serif',     # 设置字体族
    "font.sans-serif": "Arial",
    # "font.family": 'monospace',
    # "axes.labelsize": 12,       # 坐标轴标签字号
    # "font.size": 10             # 基础字号
})

class Evaluator:

    def __init__(self, multiprocess, clean_data_flag, llm_eval_model, eval_sample, test_few, 
                 include_mas, include_rag, validate_llm, rebuttal, num_threads=25):

        query = "travel_plan_evaluation"

        self.rebuttal = rebuttal
        self.num_threads = num_threads
        self.logger = get_logger(query)

        self.config = Config()
        self.config.need_full_log = False
        self.config.final_config.update(
            self.config._load_file_config('agentpedia/config/category_config/travel_plan_eval_config.yaml'))

        self.multiprocess = multiprocess
        self.clean_data_flag = clean_data_flag
        self.llm_eval_model = llm_eval_model
        self.config.model = llm_eval_model
        self.eval_sample = eval_sample
        self.test_few = test_few
        self.include_mas = include_mas
        self.include_rag = include_rag
        self.validate_llm = validate_llm

        self.request_llm = RequestLLM(self.config)
    
    def load_result_dict(self, query_list, base_model):

        query_classify_dict = None
        result_dict = {}
        query_info_dict = {}
        self.base_model = base_model
        plan_dir = f"plan_data{f'_mas' if self.include_mas else ''}_{base_model}"
        
        if self.rebuttal:
            plan_dir = f"../rebuttal/plan_data_qwen25-72b"
            method_list = ["reference"]
        elif base_model.startswith("deepseek"):
            method_list = ["given_direct_objective", "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_one", 
                        "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_abstractive",]
        elif self.validate_llm:
            method_list = [
                    "given_direct_objective", "given_cot_objective", "given_refine_objective", 
                    "multi_agent_debate", "multi_agent_collaboration", 
                    "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_one", 
                    "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_abstractive"
                ]
        elif self.include_rag:
            method_list = [
                "given_direct_objective_retrieval_N7", "given_direct_objective_retrieval_N6", "given_direct_objective_retrieval_N5", "given_direct_objective_retrieval_N3",
                "given_direct_objective_retrieval_N2", 
                "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_one", 
                "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_abstractive", 
                "given_direct_objective_retrieval_N7_clean", "given_direct_objective_retrieval_N6_clean", "given_direct_objective_retrieval_N5_clean", "given_direct_objective_retrieval_N3_clean",
                "given_direct_objective_retrieval_N2_clean", 
                "given_direct_objective_retrieval_all_clean", "given_direct_objective_retrieval_half_clean", "given_direct_objective_retrieval_one_clean", 
                "given_direct_objective_retrieval_selective_half_clean", "given_direct_objective_retrieval_selective_one_clean", "given_direct_objective_retrieval_abstractive_clean", 
            ]
        else:
            method_list = [
                "given_direct_objective", "given_cot_objective", "given_refine_objective", 
                "multi_agent_debate", "multi_agent_collaboration", 
                "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_one", 
                "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_abstractive"
            ]
        if self.include_mas:
            # method_list = []
            self.mas_method_list = ["evolutionary_optimize"]
            method_list.extend(self.mas_method_list)
        
        remove_path_list = []
        plan_dir_list = [plan_dir]
        if self.include_mas:
            plan_dir_list.append(f"plan_data_{base_model}")
        for plan_dir in plan_dir_list:
            idx = 0
            for filename in tqdm(os.listdir(plan_dir)):
                # if idx < 14000:
                #     idx += 1
                #     continue
                # idx += 1
                assert filename.endswith('.pkl')
                query, method = filename[:-4].split('-')
                if query not in query_list or method not in method_list:
                    continue
                file_path = os.path.join(plan_dir, filename)
                pkl_data = pickle.load(open(file_path, 'rb'))
                if "evolutionary_optimize" in filename:
                    # print(filename)
                    pkl_data = pkl_data["optimal_plan"]["plan"]
                if query not in result_dict:
                    result_dict[query] = {}
                result_dict[query][method] = pkl_data
                query_info_dict[query] = pickle.load(open(f"construct_data/{query}-data_construct.pkl", 'rb'))

                # clean data
                if self.clean_data_flag:
                    # print(111)
                    try:
                        flag, new_pkl_data = self.clean_data(pkl_data, query_info_dict[query])
                        # if flag:
                        #     # print(111111)
                        #     pickle.dump(new_pkl_data, open(file_path, 'wb'))
                        #     # pickle.dump(query_info_dict[query], open(f"construct_data/{query}-data_construct.pkl", 'wb'))
                    except:
                        assert method in method_list
                        print(file_path)
                        remove_path_list.append(file_path)
            
        print(f"failure {remove_path_list}\n{len(remove_path_list)}\n")
        assert len(remove_path_list) == 0
        # exit(-1)
        for file_path in remove_path_list:
            os.remove(file_path)
        # exit(-1)

        # self.logger.info(f"Loaded results {result_dict}")
        print("合法query数量：", len(result_dict), "out of", len(query_list))
        len_dict = {k: len(v) for k, v in result_dict.items()}
        print("不合法method数量的query：", [(k, v) for k, v in len_dict.items() if v != len(method_list)])
        # exit(-1)
        
        # query classification
        query_classify_dict = {
            "generic": [], 
            "personal": {
                "holiday": [], 
                "season": [],
                "audience": [],
                "category": [],
                "compact": [],
            }
        }
        for query in tqdm(query_info_dict):
            query_info = query_info_dict[query]
            query_details = query_info["query_analysis_result"]
            constraints_gt = query_details["约束"]
            if "季节" in constraints_gt and constraints_gt["季节"] is not None and constraints_gt["季节"] != "":
                if constraints_gt["季节"] not in ["春季", "夏季", "秋季", "冬季"]:
                    print(constraints_gt["季节"])
                    raise ValueError
                query_classify_dict["personal"]["season"].append(query)
            elif "节假日" in constraints_gt and constraints_gt["节假日"] is not None and constraints_gt["节假日"] != "":
                if constraints_gt["节假日"] not in ["春节", "清明", "五一", "端午", "中秋", "国庆"]:
                    print(constraints_gt["节假日"])
                    raise ValueError
                query_classify_dict["personal"]["holiday"].append(query)
            elif "受众" in constraints_gt and constraints_gt["受众"] is not None and constraints_gt["受众"] != "":
                if constraints_gt["受众"] not in ["老年", "单人", "情侣", "亲子"]:
                    print(constraints_gt["受众"])
                    raise ValueError
                query_classify_dict["personal"]["audience"].append(query)
            elif "POI类别" in constraints_gt and constraints_gt["POI类别"] is not None and constraints_gt["POI类别"] != "":
                if constraints_gt["POI类别"] not in ["自然风光", "历史文化", "休闲娱乐", "艺术科技", "城市观光", "宗教文化"]:
                    print(constraints_gt["POI类别"])
                    raise ValueError
                query_classify_dict["personal"]["category"].append(query)
            elif "行程紧凑性" in constraints_gt and constraints_gt["行程紧凑性"] is not None and constraints_gt["行程紧凑性"] != "":
                if constraints_gt["行程紧凑性"] not in ["特种兵"]:
                    print(constraints_gt["行程紧凑性"])
                    raise ValueError
                query_classify_dict["personal"]["compact"].append(query)
            else:
                query_classify_dict["generic"].append(query)
        query_classify_dict["personal"]["total"] = [vv for k, v in query_classify_dict["personal"].items() for vv in v]
        print(f"通用query数量: {len(query_classify_dict['generic'])}, 个性化query数量: {len(query_classify_dict['personal']['total'])}")
        print(f"节假日query数量: {len(query_classify_dict['personal']['holiday'])}, 受众query数量: {len(query_classify_dict['personal']['audience'])}, 季节query数量: {len(query_classify_dict['personal']['season'])}, 类别query数量: {len(query_classify_dict['personal']['category'])}, 紧凑性query数量: {len(query_classify_dict['personal']['compact'])}")
        
        print_poi = False
        if print_poi:
            poi_len_list_classify_dict = {
                "generic": [len(query_info_dict[q]["poi_extract_result_improve"]) for q in query_classify_dict["generic"]], 
                "personal": {
                    "holiday": [len(query_info_dict[q]["poi_extract_result_improve"]) for q in query_classify_dict["personal"]["holiday"]], 
                    "season": [len(query_info_dict[q]["poi_extract_result_improve"]) for q in query_classify_dict["personal"]["season"]], 
                    "audience": [len(query_info_dict[q]["poi_extract_result_improve"]) for q in query_classify_dict["personal"]["audience"]], 
                    "category": [len(query_info_dict[q]["poi_extract_result_improve"]) for q in query_classify_dict["personal"]["category"]], 
                    "compact": [len(query_info_dict[q]["poi_extract_result_improve"]) for q in query_classify_dict["personal"]["compact"]], 
                    "total": [len(query_info_dict[q]["poi_extract_result_improve"]) for q in query_classify_dict["personal"]["total"]], 
                }, 
                "total": [len(query_info_dict[q]["poi_extract_result_improve"]) for q in query_classify_dict["generic"] + query_classify_dict["personal"]["total"]], 
            }
            print(f"通用query数量: {np.mean(poi_len_list_classify_dict['generic'])}, 个性化query数量: {np.mean(poi_len_list_classify_dict['personal']['total'])}, query数量: {np.mean(poi_len_list_classify_dict['total'])}")
            print(f"节假日query数量: {np.mean(poi_len_list_classify_dict['personal']['holiday'])}, 受众query数量: {np.mean(poi_len_list_classify_dict['personal']['audience'])}, 季节query数量: {np.mean(poi_len_list_classify_dict['personal']['season'])}, 类别query数量: {np.mean(poi_len_list_classify_dict['personal']['category'])}, 紧凑性query数量: {np.mean(poi_len_list_classify_dict['personal']['compact'])}")
            print(f"通用query数量: {np.sum(poi_len_list_classify_dict['generic'])}, 个性化query数量: {np.sum(poi_len_list_classify_dict['personal']['total'])}, query数量: {np.sum(poi_len_list_classify_dict['total'])}")
            print(f"节假日query数量: {np.sum(poi_len_list_classify_dict['personal']['holiday'])}, 受众query数量: {np.sum(poi_len_list_classify_dict['personal']['audience'])}, 季节query数量: {np.sum(poi_len_list_classify_dict['personal']['season'])}, 类别query数量: {np.sum(poi_len_list_classify_dict['personal']['category'])}, 紧凑性query数量: {np.sum(poi_len_list_classify_dict['personal']['compact'])}")
            poi_list_classify_dict = {
                "generic": [query_info_dict[q]["poi_extract_result_improve"] for q in query_classify_dict["generic"]], 
                "personal": {
                    "holiday": [query_info_dict[q]["poi_extract_result_improve"] for q in query_classify_dict["personal"]["holiday"]], 
                    "season": [query_info_dict[q]["poi_extract_result_improve"] for q in query_classify_dict["personal"]["season"]], 
                    "audience": [query_info_dict[q]["poi_extract_result_improve"] for q in query_classify_dict["personal"]["audience"]], 
                    "category": [query_info_dict[q]["poi_extract_result_improve"] for q in query_classify_dict["personal"]["category"]], 
                    "compact": [query_info_dict[q]["poi_extract_result_improve"] for q in query_classify_dict["personal"]["compact"]], 
                    "total": [query_info_dict[q]["poi_extract_result_improve"] for q in query_classify_dict["personal"]["total"]], 
                }, 
                "total": [query_info_dict[q]["poi_extract_result_improve"] for q in query_classify_dict["generic"] + query_classify_dict["personal"]["total"]], 
            }
            poi_list = [p["名称"] for q in query_classify_dict["generic"] + query_classify_dict["personal"]["total"] for p in query_info_dict[q]["poi_extract_result_improve"]]
            print(len(poi_list), len(set(poi_list)))
            exit(-1)
        
        if self.eval_sample > 0:
            query_classify_dict = {
                "generic": [], 
                "personal": {
                    "holiday": query_classify_dict["personal"]["holiday"][:self.eval_sample], 
                    "season": query_classify_dict["personal"]["season"][:self.eval_sample],
                    "audience": query_classify_dict["personal"]["audience"][:self.eval_sample],
                    "category": query_classify_dict["personal"]["category"][:self.eval_sample],
                    "compact": query_classify_dict["personal"]["compact"][:self.eval_sample],
                }                
            }
            query_classify_dict["personal"]["total"] = [vv for k, v in query_classify_dict["personal"].items() for vv in v]
            query_list = query_classify_dict["personal"]["total"] + query_classify_dict["generic"]
            result_dict = {k: v for k, v in result_dict.items() if k in query_list}
            query_info_dict = {k: v for k, v in query_info_dict.items() if k in query_list}
        
        self.query_classify_dict = query_classify_dict

        return result_dict, query_info_dict, query_classify_dict
    
    def clean_data(self, result, query_info, test_labels=False):
        
        city = query_info["query_analysis_result"]["地点"]
        
        modify = False
        if test_labels:
            for p in query_info["poi_extract_result_improve"]:
                # if p["detail"]["开放时间"] == "全天开放":
                #     p["detail"]["开放时间"] = "0:00-24:00"
                #     modify = True
                try:
                    # if city == "上海" and p["名称"] == "上海隧道科技馆":
                    #     p["detail"]["开放时间"] = "13:00-16:30"
                    #     modify = True
                    # if city == "北京" and p["名称"] == "鼓楼":
                    #     p["detail"]["开放时间"] = "9:30-16:30"
                    #     modify = True
                    # if city == "北京" and p["名称"] == "鼓楼大街":
                    #     p["detail"]["开放时间"] = "7:30-24:00"
                    #     modify = True
                    # if city == "广州" and p["名称"] == "邮政博览馆-南门":
                    #     p["detail"]["开放时间"] = "10:00-17:00"
                    #     modify = True
                    # if city == "广州" and p["名称"] == "广州地铁博物馆":
                    #     p["detail"]["开放时间"] = "9:00-17:00"
                    #     modify = True
                    # if city == "厦门" and p["名称"] == "隧道涂鸦":
                    #     p["detail"]["开放时间"] = "8:00-16:00"
                    #     modify = True
                    # if city == "厦门" and p["名称"] == "三一堂":
                    #     p["detail"]["开放时间"] = "9:00-20:00"
                    #     modify = True
                    # if city == "厦门" and p["名称"] == "厦门轮渡码头":
                    #     p["detail"]["开放时间"] = "0:00-6:30,7:10-24:00"
                    #     modify = True
                    # if city == "武汉" and p["名称"] == "汉口俄租界巡捕房旧址":
                    #     p["detail"]["开放时间"] = "0:00-24:00"
                    #     modify = True
                    # if city == "福州" and p["名称"] == "天主教西门若瑟堂":
                    #     p["detail"]["开放时间"] = "8:30-11:30,15:00-17:30"
                    #     modify = True
                    # if city == "吉林" and p["名称"] == "长蛇山原始文化遗址":
                    #     p["detail"]["开放时间"] = "0:00-24:00"
                    #     modify = True
                    if city == "肇庆" and p["名称"] == "将军山风景点":
                        p["detail"]["开放时间"] = "0:00-2:00,9:00-17:00,19:00-24:00"
                        modify = True
                    if city == "肇庆" and p["名称"] == "云修台":
                        p["detail"]["开放时间"] = "0:00-2:00,20:00-24:00"
                        modify = True
                    open_time = self.parse_time_string(p["detail"]["开放时间"])
                except:
                    print(city, p["名称"], p["detail"]["开放时间"])
                    raise ValueError

                # if city == "上海" and p["名称"] == "上海隧道科技馆":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': "不推荐", '中午': "13:00-15:00", '下午': "13:00-15:00", '傍晚': "不推荐", '晚上': "不推荐"}
                #     modify = True
                # if city == "日照" and p["名称"] == "小海村":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': "不推荐", '中午': "不推荐", '下午': "不推荐", '傍晚': "不推荐", '晚上': "0:00-24:00"}
                #     modify = True
                # if city == "日照" and p["名称"] == "日照港":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': "不推荐", '中午': "不推荐", '下午': "不推荐", '傍晚': "不推荐", '晚上': "0:00-24:00"}
                #     modify = True
                # if city == "吉林" and p["名称"] == "万科松花湖度假区":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': '9:30-11:00', '中午': '11:00-13:00', '下午': '13:00-15:00', '傍晚': '15:00-17:00', '晚上': '不推荐'}
                #     modify = True
                # if city == "厦门" and p["名称"].startswith("五缘湾"):
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '5:00-7:00', '上午': '不推荐', '中午': '不推荐', '下午': '16:00-17:00', '傍晚': '17:00-19:00', '晚上': '不推荐'}
                #     modify = True
                # if city == "绍兴" and p["名称"] == "古越藏书楼":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': '8:30-10:00', '中午': '不推荐', '下午': '14:00-15:00', '傍晚': '不推荐', '晚上': '不推荐'}
                #     modify = True
                # if city == "南宁" and p["名称"] == "小西湖":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': "不推荐", '中午': "不推荐", '下午': "不推荐", '傍晚': "不推荐", '晚上': "0:00-24:00"}
                #     modify = True
                # if city == "佛山" and p["名称"] == "佛山市祖庙博物馆":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': '8:30-11:00', '中午': '不推荐', '下午': '14:00-16:00', '傍晚': '不推荐', '晚上': '18:00-19:00'}
                #     modify = True
                # if city == "上海" and p["名称"] == "金光外滩中心":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': '不推荐', '中午': '不推荐', '下午': '不推荐', '傍晚': '17:00-21:00', '晚上': '17:00-21:00'}
                #     modify = True
                # if city == "福州" and p["名称"] == "海螺塔":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '8:00-10:00', '上午': '8:00-10:00', '中午': '不推荐', '下午': '12:00-14:00', '傍晚': '不推荐', '晚上': '不推荐'}
                #     modify = True
                # if city == "南宁" and p["名称"] == "邕州阁":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': '不推荐', '中午': '不推荐', '下午': '10:00-18:00', '傍晚': '10:00-18:00', '晚上': '不推荐'}
                #     modify = True
                # if city == "苏州" and p["名称"] == "七里山塘景区":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '6:00-8:00', '上午': '不推荐', '中午': '不推荐', '下午': '17:00-20:00', '傍晚': '17:00-20:00', '晚上': '22:00-24:00'}
                #     modify = True
                # if city == "广州" and p["名称"] == "宏城公园":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': '7:00-9:00', '中午': '不推荐', '下午': '16:00-17:00', '傍晚': '17:00-19:00', '晚上': '不推荐'}
                #     modify = True
                # if city == "苏州" and p["名称"] == "网师园":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': '不推荐', '中午': '不推荐', '下午': '15:00-16:00', '傍晚': '18:00-19:00', '晚上': '20:30-21:10'}
                #     modify = True
                # if city == "上海" and p["名称"].endswith("静安寺"):
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '7:30-9:00', '上午': '9:00-11:00', '中午': '11:00-13:00', '下午': '13:00-15:00', '傍晚': '不推荐', '晚上': '18:00-20:00'}
                #     modify = True
                # if city == "桂林" and p["名称"].endswith("龙脊峡漂流"):
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': '10:00-11:00', '中午': '14:30-15:00', '下午': '14:00-16:00', '傍晚': '不推荐', '晚上': '不推荐'}
                #     modify = True
                # if city == "上海" and p["名称"].endswith("益丰·外滩源"):
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': '不推荐', '中午': '12:00-14:00', '下午': '14:00-17:00', '傍晚': '17:00-19:00', '晚上': '19:00-20:00'}
                #     modify = True
                # if city == "张家界" and p["名称"] == "大观台":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': "不推荐", '中午': "不推荐", '下午': "不推荐", '傍晚': "不推荐", '晚上': "0:00-24:00"}
                #     modify = True
                # if city == "厦门" and p["名称"] == "三一堂":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': '9:00-17:00', '中午': '9:00-17:00', '下午': '9:00-17:00', '傍晚': '16:00-19:00', '晚上': '不推荐'}
                #     modify = True
                # if city == "厦门" and p["名称"] == "环岛海滨浴场":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': '9:30-12:00', '中午': '不推荐', '下午': '不推荐', '傍晚': '18:00-19:00', '晚上': '不推荐'}
                #     modify = True
                # if city == "厦门" and p["名称"] == "各界抗敌后援会会址":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': "不推荐", '中午': "不推荐", '下午': "不推荐", '傍晚': "不推荐", '晚上': "0:00-24:00"}
                #     modify = True
                # if city == "福州" and p["名称"] == "天主教西门若瑟堂":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': '8:30-10:30', '中午': '不推荐', '下午': '15:00-16:30', '傍晚': '不推荐', '晚上': '不推荐'}
                #     modify = True
                # if city == "厦门" and p["名称"] == "赶海文化广场":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': "不推荐", '中午': "不推荐", '下午': "不推荐", '傍晚': "不推荐", '晚上': "0:00-24:00"}
                #     modify = True
                # if city == "苏州" and p["名称"] == "明月湾古村":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '0:00-8:00', '上午': '不推荐', '中午': '不推荐', '下午': '不推荐', '傍晚': '18:00-19:00', '晚上': '19:00-24:00'}
                #     modify = True
                # if city == "杭州" and p["名称"] == "北高峰":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '0:00-8:00', '上午': '不推荐', '中午': '不推荐', '下午': '15:30-17:00', '傍晚': '不推荐', '晚上': '不推荐'}
                #     modify = True
                # if city == "滨州" and p["名称"] == "邹平博物馆":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': '8:30-10:30', '中午': '不推荐', '下午': '14:30-16:30', '傍晚': '不推荐', '晚上': '不推荐'}
                #     modify = True
                # if city == "福州" and p["名称"] == "蛤沙村":
                #     p["detail"]["推荐游玩开始时间"] = {'早晨': '不推荐', '上午': "不推荐", '中午': "不推荐", '下午': "不推荐", '傍晚': "不推荐", '晚上': "0:00-24:00"}
                #     modify = True
                if city == "桂林" and p["名称"] == "遇龙河景区":
                    p["detail"]["推荐游玩开始时间"] = {'早晨': '8:00-9:00', '上午': '不推荐', '中午': '不推荐', '下午': '不推荐', '傍晚': '16:00-17:00', '晚上': '不推荐'}
                    modify = True
                recommend_time_dict = p["detail"]["推荐游玩开始时间"]
                for k, v in recommend_time_dict.items(): 
                    if v != "不推荐":
                        # if v == "全天开放":
                        #     recommend_time_dict[k] = "0:00-24:00"
                        #     modify = True
                        # if v == "不开放":
                        #     recommend_time_dict[k] = "不推荐"
                        #     modify = True
                        #     continue
                        # if v == "根据潮汐表选择":
                        #     recommend_time_dict[k] = "0:00-24:00"
                        #     modify = True
                        try:
                            self.parse_time_string(recommend_time_dict[k])
                        except:
                            print(city, p["名称"], recommend_time_dict)
                            raise ValueError
                duration = p["detail"]["预计游玩时长"]
                # if duration == "随意" or duration == "不适用" or duration == "未知":
                #     p["detail"]["预计游玩时长"] = "0"
                #     modify = True
                #     duration = p["detail"]["预计游玩时长"]
                # if '天' in duration:
                #     p["detail"]["预计游玩时长"] = "0"
                #     modify = True
                #     duration = p["detail"]["预计游玩时长"]
                try:
                    if "-" in duration:
                        duration_min, duration_max = duration.split('-')
                        duration = float(eval(duration_min))
                    else:
                        duration = float(eval(duration))
                except:
                    print(city, p["名称"], duration)
                    raise ValueError
            
        for _, p_l in result.items():
            # # Qwen clean
            try:    
                assert isinstance(p_l, list)
                # if "名称" not in p_l[0] and (len(p_l[0]) == 1 or "午餐" in p_l[0]) and isinstance(list(p_l[0].values())[0], list):
                #     result = {idx: list(r_dict.values())[0] for idx, r_dict in enumerate(p_l)}
                #     return True, result
            except:
                # if isinstance(p_l, str):
                #     result = {k: v for k, v in result.items() if not isinstance(v, str)}
                #     return True, result
                # if isinstance(p_l, list) and len(p_l) == 0:
                #     result = {k: v for k, v in result.items() if len(v) > 0}
                #     return True, result
                # try:
                #     assert len(result) == 1 and isinstance(p_l, dict) and len(p_l) > 0
                #     result = p_l
                #     return True, result
                # except:
                #     # if _ == "黄山休闲娱乐四日游" and '休闲娱乐活动' in result:
                #     #     result = result["黄山休闲娱乐四日游"]
                #     #     return True, result
                #     print(result)
                #     raise ValueError
                print(result)
                raise ValueError
            try:
                assert "名称" in p_l[0]
            except:
                # try:
                #     if len(result) == 2 and "备注" in result:
                #         del result["备注"]
                #     if len(result) == 1 and "行程" in p_l[0] and isinstance(p_l[0]["行程"], list):
                #         result = {idx: ps["行程"] for idx, ps in enumerate(p_l)}
                #         return True, result
                #     if len(result) == 1 and "景点" in p_l[0] and isinstance(p_l[0]["景点"], list):
                #         result = {idx: ps["景点"] for idx, ps in enumerate(p_l)}
                #         return True, result
                #     if len(result) == 1 and "行程安排" in p_l[0] and isinstance(p_l[0]["行程安排"], list):
                #         result = {idx: ps["行程安排"] for idx, ps in enumerate(p_l)}
                #         return True, result
                #     if len(result) == 1 and "景点行程" in p_l[0] and isinstance(p_l[0]["景点行程"], list):
                #         result = {idx: ps["景点行程"] for idx, ps in enumerate(p_l)}
                #         return True, result
                #     if len(result) == 1 and "景点列表" in p_l[0] and isinstance(p_l[0]["景点列表"], list):
                #         result = {idx: ps["景点列表"] for idx, ps in enumerate(p_l)}
                #         return True, result
                #     if len(result) == 1 and "上午" in p_l[0] and "下午" in p_l[0] and isinstance(p_l[0]["上午"], dict):
                #         result = {idx: [v for k, v in ps.items() if isinstance(v, dict)] for idx, ps in enumerate(p_l)}
                #         return True, result
                #     if '描述' in p_l[0] and len(p_l) == 1:
                #         result = {k: v for k, v in result.items() if not (len(v) == 1 and "名称" not in v[0] and "描述" in v[0])}
                #         return True, result
                #     if '活动' in p_l[0] and '时间' in p_l[0] and len(p_l) == 1:
                #         result = {k: v for k, v in result.items() if not (len(v) == 1 and "名称" not in v[0] and "活动" in v[0] and "时间" in v[0])}
                #         return True, result
                # except:
                #     print(result)
                #     raise ValueError
                print(result)
                raise ValueError
            # try:    
            #     assert isinstance(p_l, list)
            # except:
            #     print(result)
            #     raise ValueError
            for p in p_l:
                assert isinstance(p, dict)
                try:
                    # # LLama clean
                    # if "时间段" in p and "时间" in p:
                    #     p["时间段"] = f"{p['时间段']}({p['时间']})"
                    #     # del p["时间"]
                    #     modify = True
                    # elif "时间段" in p and "起止时间" in p:
                    #     p["时间段"] = f"{p['时间段']}({p['起止时间']})"
                    #     modify = True
                    # elif "时间段" in p and "游玩时间" in p:
                    #     p["时间段"] = f"{p['时间段']}({p['游玩时间']})"
                    #     # del p["游玩时间"]
                    #     modify = True
                    # elif "时间段" in p and "具体游玩起止时间" in p:
                    #     p["时间段"] = f"{p['时间段']}({p['具体游玩起止时间']})"
                    #     # del p["具体游玩起止时间"]
                    #     modify = True
                    # elif "时间段" in p and "游玩起止时间" in p:
                    #     p["时间段"] = f"{p['时间段']}({p['游玩起止时间']})"
                    #     # del p['游玩起止时间']
                    #     modify = True
                    # elif "时间段" in p and "具体时间" in p:
                    #     p["时间段"] = f"{p['时间段']}({p['具体时间']})"
                    #     # del p['具体时间']
                    #     modify = True
                    # elif "时间段" in p and "具体游玩时间" in p:
                    #     p["时间段"] = f"{p['时间段']}({p['具体游玩时间']})"
                    #     # del p['具体游玩时间']
                    #     modify = True
                    # elif "时间段" in p and "开始时间" in p and "结束时间" in p:
                    #     p["时间段"] = f"{p['时间段']}({p['开始时间']}-{p['结束时间']})"
                    #     modify = True
                    # elif "访问时间段" in p:
                    #     if '游玩起止时间' in p:
                    #         p["时间段"] = f"{p['访问时间段']}({p['游玩起止时间']})"
                    #     elif "具体游玩起止时间" in p:
                    #         p["时间段"] = f"{p['访问时间段']}({p['具体游玩起止时间']})"
                    #     elif "具体时间" in p:
                    #         p["时间段"] = f"{p['访问时间段']}({p['具体时间']})"
                    #     elif "游玩时间" in p:
                    #         p["时间段"] = f"{p['访问时间段']}({p['游玩时间']})"
                    #     elif "开始时间" in p and "结束时间" in p:
                    #         p["时间段"] = f"{p['访问时间段']}({p['开始时间']}-{p['结束时间']})"
                    #     elif "时间段" not in p:
                    #         p["时间段"] = p['访问时间段']
                    #         # del p["访问时间段"]
                    #         # if '具体游玩时间' in p:
                    #         #     del p['具体游玩时间']
                    #     modify = True
                    # elif "游玩时间" in p:
                    #     if "时间段" not in p:
                    #         p["时间段"] = p['游玩时间']
                    #     modify = True
                    try:
                        assert "名称" in p
                    except:
                        # if '描述' in p and len(p_l) == 1:
                        #     result = {k: v for k, v in result.items() if not (len(v) == 1 and "名称" not in v[0] and "描述" in v[0])}
                        #     return True, result
                        # if '活动' in p and '时间' in p and len(p_l) == 1:
                        #     result = {k: v for k, v in result.items() if not (len(v) == 1 and "名称" not in v[0] and "活动" in v[0] and "时间" in v[0])}
                        #     return True, result
                        print(result)
                        print(p)
                        raise ValueError
                    try:
                        assert "时间段" in p
                    except:
                        # if '描述' in p and len(p_l) == 1:
                        #     result = {k: v for k, v in result.items() if not (len(v) == 1 and "时间段" not in v[0] and "描述" in v[0])}
                        #     return True, result
                        print(result)
                        print(p)
                        raise ValueError
                    self.parse_time_string(self.extract_bracket_content(p["时间段"]), plan=True)
                except:
                    # # LLama clean
                    # if "不推荐" in p["时间段"]:
                    #     result[_] = [p for p in p_l if "不推荐" not in p["时间段"]]
                    #     return True, result
                    # if "不安排" in p["时间段"]:
                    #     result[_] = [p for p in p_l if "不安排" not in p["时间段"]]
                    #     return True, result
                    # if "不建议" in p["时间段"]:
                    #     result[_] = [p for p in p_l if "不建议" not in p["时间段"]]
                    #     return True, result
                    # if "更换景点" in p["时间段"]:
                    #     result[_] = [p for p in p_l if "更换景点" not in p["时间段"]]
                    #     return True, result
                    print(result)
                    print(city, p["名称"], p["时间段"], p_l)
                    raise ValueError
        
        return modify, result

    def eval(self, result_dict, query_info_dict, query_classify_dict=None, baseline_name=None):

        query_list = list(result_dict.keys())
        print(len(query_list), list(result_dict[query_list[0]].keys()))
        self.logger.info(f"Start to evaluate the following queries: {query_list}")
        self.logger.info(f"method list: {list(result_dict[query_list[0]].keys())}")
        print("=========")
        # exit(0)

        # metrics, failure_list = {}, []
        # for query in tqdm(query_list):
        #     local_metrics, local_failure_list = self.eval_query(query, result_dict, query_info_dict, query_classify_dict, baseline_name)
        #     metrics.update(local_metrics)
        #     failure_list.extend(local_failure_list)
        #     exit(-1)
        # print(failure_list, len(failure_list))
        # # exit(-1)
        
        metrics, failure_list = {}, []
        if self.multiprocess:
            batch_size = 10
            num_batch = len(query_list) // batch_size
            if len(query_list) % batch_size > 0:
                num_batch += 1
            for batch_idx in tqdm(range(num_batch)):
                with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                    futures = [executor.submit(self.eval_query, query, result_dict, query_info_dict, query_classify_dict, baseline_name)
                            for query in query_list[batch_idx * batch_size: (batch_idx + 1) * batch_size]]
                    result_list = [future.result() for future in concurrent.futures.as_completed(futures)]
                for local_metrics, local_failure_list in result_list:
                    metrics.update(local_metrics)
                    failure_list.extend(local_failure_list)
                # exit(-1)
        else:
            for query in tqdm(query_list):
                local_metrics, local_failure_list = self.eval_query(query, result_dict, query_info_dict, query_classify_dict, baseline_name)
                metrics.update(local_metrics)
                failure_list.extend(local_failure_list)
        print(failure_list, len(failure_list))
        
        def show_results(metrics):
            # print(metrics)
            new_metrics = {}
            for query, q_res in metrics.items():
                for method, m_res in q_res.items():
                    if method not in new_metrics:
                        new_metrics[method] = {}
                    for criterion, c_res in m_res.items():
                        if isinstance(c_res, dict):
                            if criterion not in new_metrics[method]:
                                new_metrics[method][criterion] = {}
                            for criterion_1, c_res_1 in c_res.items():
                                if isinstance(c_res_1, dict):
                                    if criterion_1 not in new_metrics[method][criterion]:
                                        new_metrics[method][criterion][criterion_1] = {}
                                    for criterion_2, c_res_2 in c_res_1.items():
                                        assert not isinstance(c_res_2, dict)
                                        if criterion_2 not in new_metrics[method][criterion][criterion_1]:
                                            new_metrics[method][criterion][criterion_1][criterion_2] = []
                                        new_metrics[method][criterion][criterion_1][criterion_2].append(c_res_2)
                                else:
                                    if criterion_1 not in new_metrics[method][criterion]:
                                        new_metrics[method][criterion][criterion_1] = []
                                    new_metrics[method][criterion][criterion_1].append(c_res_1)
                        else:
                            if criterion not in new_metrics[method]:
                                new_metrics[method][criterion] = []
                            new_metrics[method][criterion].append(c_res)
            new_new_metrics = {}
            len_record = {}
            for method, m_res in new_metrics.items():
                new_new_metrics[method] = {}
                len_record[method] = {}
                for criterion, c_res in m_res.items():
                    if isinstance(c_res, dict):
                        new_new_metrics[method][criterion] = {}
                        len_record[method][criterion] = {}
                        for criterion_1, c_res_1 in c_res.items():
                            if isinstance(c_res_1, dict):
                                new_new_metrics[method][criterion][criterion_1] = {}
                                len_record[method][criterion][criterion_1] = {}
                                for criterion_2, c_res_2 in c_res_1.items():
                                    assert not isinstance(c_res_2, dict)
                                    len_record[method][criterion][criterion_1][criterion_2] = f"{len([r for r in c_res_2 if r is not None and not math.isnan(r)])}/{len(c_res_2)}"
                                    mean = float(np.array([r for r in c_res_2 if r is not None and not math.isnan(r)]).mean())
                                    new_new_metrics[method][criterion][criterion_1][criterion_2] = mean
                            else:
                                len_record[method][criterion][criterion_1] = f"{len([r for r in c_res_1 if r is not None and not math.isnan(r)])}/{len(c_res_1)}"
                                mean = float(np.array([r for r in c_res_1 if r is not None and not math.isnan(r)]).mean())
                                new_new_metrics[method][criterion][criterion_1] = mean
                    else:
                        len_record[method][criterion] = f"{len([r for r in c_res if r is not None and not math.isnan(r)])}/{len(c_res)}"
                        mean = float(np.array([r for r in c_res if r is not None and not math.isnan(r)]).mean())
                        new_new_metrics[method][criterion] = mean
            return new_new_metrics, len_record
        
        if self.eval_sample > 0:
            eval_dir = f"eval_results{f'_mas' if self.include_mas else ''}{f'_rag' if self.include_rag else ''}_{self.base_model}/eval_sample"
        elif self.llm_eval_model is not None:
            eval_dir = f"eval_results{f'_mas' if self.include_mas else ''}{f'_rag' if self.include_rag else ''}_{self.base_model}/{self.llm_eval_model}"
        else:
            eval_dir = f"eval_results{f'_mas' if self.include_mas else ''}{f'_rag' if self.include_rag else ''}_{self.base_model}"
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)
        metric_dir = os.path.join(eval_dir, "metrics")
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)
            
        final_metrics, final_details = show_results(metrics)
        with open(os.path.join(eval_dir, 'eval_results_all.json'), 'w', encoding='utf-8') as f:
            json.dump({"metrics": final_metrics, "details": final_details}, f, ensure_ascii=False, indent=4)
        with open(os.path.join(metric_dir, 'metrics_all.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)
        # print(f"total {len(metrics)}", final_metrics)
        # exit(-1)

        # query classification
        if query_classify_dict is not None:
            metrics_classify_dict = {
                "generic": {}, 
                "personal": {
                    "holiday": {}, 
                    "season": {},
                    "audience": {},
                    "category": {},
                    "compact": {},
                    "total": {}
                }                
            }
            for query, q_res in metrics.items():
                if query in query_classify_dict["generic"]:
                    metrics_classify_dict["generic"][query] = q_res
                else:
                    assert query in query_classify_dict["personal"]["total"]
                    metrics_classify_dict["personal"]["total"][query] = q_res
                    if query in query_classify_dict["personal"]["holiday"]:
                        metrics_classify_dict["personal"]["holiday"][query] = q_res
                    elif query in query_classify_dict["personal"]["season"]:
                        metrics_classify_dict["personal"]["season"][query] = q_res
                    elif query in query_classify_dict["personal"]["audience"]:
                        metrics_classify_dict["personal"]["audience"][query] = q_res
                    elif query in query_classify_dict["personal"]["category"]:
                        metrics_classify_dict["personal"]["category"][query] = q_res
                    elif query in query_classify_dict["personal"]["compact"]:
                        metrics_classify_dict["personal"]["compact"][query] = q_res
            
            final_metrics, final_details = show_results(metrics_classify_dict["generic"])
            with open(os.path.join(eval_dir, 'eval_results_generic.json'), 'w', encoding='utf-8') as f:
                json.dump({"metrics": final_metrics, "details": final_details}, f, ensure_ascii=False, indent=4)
            with open(os.path.join(metric_dir, 'metrics_generic.json'), 'w', encoding='utf-8') as f:
                json.dump(metrics_classify_dict["generic"], f, ensure_ascii=False, indent=4)
            for title in metrics_classify_dict["personal"]:
                final_metrics, final_details = show_results(metrics_classify_dict["personal"][title])
                with open(os.path.join(eval_dir, f'eval_results_personal_{title}.json'), 'w', encoding='utf-8') as f:
                    json.dump({"metrics": final_metrics, "details": final_details}, f, ensure_ascii=False, indent=4)     
                with open(os.path.join(metric_dir, f'metrics_personal_{title}.json'), 'w', encoding='utf-8') as f:
                    json.dump(metrics_classify_dict["personal"][title], f, ensure_ascii=False, indent=4)         

    def eval_query(self, query, result_dict, query_info_dict, query_classify_dict, baseline_name):
        
        metrics, failure_list = {}, []

        start_time = time.time()
        query_info = query_info_dict[query]
        metrics[query] = {}
        baseline = result_dict[query][baseline_name] if baseline_name in result_dict[query] else None
        if not self.multiprocess:
            for method, result in result_dict[query].items():
                r = self.eval_method(query, query_info, method, result, baseline, baseline_name)
                if r[0] is None:
                    print(f"fail {query}-{r[1]}")
                    failure_list.append(f"{query}-{r[1]}")
                else:
                    metrics[query][r[1]] = r[0]
                # exit(-1)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = [executor.submit(self.eval_method, query, query_info, method, result, baseline, baseline_name)
                        for method, result in result_dict[query].items()]
                result_list = [future.result() for future in concurrent.futures.as_completed(futures)]
            # result = {}
            for r in result_list:
                if r[0] is None:
                    print(f"fail {query}-{r[1]}")
                    failure_list.append(f"{query}-{r[1]}")
                    # exit(-1)
                else:
                    metrics[query][r[1]] = r[0]
        self.logger.info(f"Eval {query} over {baseline_name}: {time.time() - start_time}s")
        # print(f"eval {query} over {baseline_name} done in {time.time() - start_time}s")
        # exit(-1)
        # break
        
        return metrics, failure_list
        
    def eval_method(self, query, query_info, method, result, baseline, baseline_name):
        
        try:
        # if 2 > 1:
            start_time = time.time()
            eval_dir_basic = f"eval_data{f'_mas' if self.include_mas and method in self.mas_method_list else ''}_{self.base_model}"
            if self.eval_sample > 0:
                eval_dir_final = os.path.join(eval_dir_basic, "eval_sample")
            elif self.llm_eval_model is not None:
                eval_dir_final = os.path.join(eval_dir_basic, self.llm_eval_model)
            else:
                eval_dir_final = eval_dir_basic
            if not os.path.exists(eval_dir_final):
                os.makedirs(eval_dir_final)

            metrics_basic, metrics_final = None, None
            eval_path_basic = os.path.join(eval_dir_basic, f"{query}-{method}.pkl")
            if os.path.exists(eval_path_basic):
                metrics_basic = pickle.load(open(eval_path_basic, "rb"))
            if self.llm_eval_model is not None:
                eval_path_final = os.path.join(eval_dir_final, f"{query}-{method}.pkl")
                if os.path.exists(eval_path_final):
                    metrics_final = pickle.load(open(eval_path_final, "rb"))
            
            # if metrics is not None and isinstance(metrics, dict):
            #     return metrics, method
            # else:
            #     raise NotImplementedError

            # print(f"eval {method} {query} over {baseline_name}")
            # print(result)
            # print('-------------------------')

            if metrics_basic is None and metrics_final is None:
                commonsense_metrics = self.commonsense_evaluation(query, query_info, method, result, baseline)
                days_accuracy, failure_rate, repeat_rate, time_disorder_rate = commonsense_metrics
            else:
                metrics_cache = metrics_basic if metrics_basic is not None else metrics_final
                days_accuracy, failure_rate, repeat_rate, time_disorder_rate = metrics_cache["days_accuracy"], metrics_cache["failure_rate"], metrics_cache["repeat_rate"], metrics_cache["time_disorder_rate"]
            if metrics_basic is None and metrics_final is None:
                spatial_metrics = self.spatial_evaluation(query, query_info, method, result, baseline)
                average_distance, local_optimal_distance_margin_ratio, global_optimal_distance_margin_ratio = spatial_metrics
            else:
                metrics_cache = metrics_basic if metrics_basic is not None else metrics_final
                average_distance, local_optimal_distance_margin_ratio, global_optimal_distance_margin_ratio = metrics_cache["average_distance"], metrics_cache["local_optimal_distance_margin_ratio"], metrics_cache["global_optimal_distance_margin_ratio"]
            if metrics_basic is None and metrics_final is None:
                temporal_metrics = self.temporal_evaluation(query, query_info, method, result, baseline)
                shop_hours_overflow_rate, shop_hours_overflow_ratio, recommended_hours_miss_rate, recommended_hours_miss_margin, \
                    duration_underflow_rate, duration_underflow_margin_ratio, total_time, buffer_time, buffer_ratio = temporal_metrics
            else:
                metrics_cache = metrics_basic if metrics_basic is not None else metrics_final
                shop_hours_overflow_rate, shop_hours_overflow_ratio, recommended_hours_miss_rate, recommended_hours_miss_margin, \
                    duration_underflow_rate, duration_underflow_margin_ratio, total_time, buffer_time, buffer_ratio = metrics_cache["shop_hours_overflow_rate"], metrics_cache["shop_hours_overflow_ratio"], \
                        metrics_cache["recommended_hours_miss_rate"], metrics_cache["recommended_hours_miss_margin"], metrics_cache["duration_underflow_rate"], metrics_cache["duration_underflow_margin_ratio"], \
                            metrics_cache["total_time"], metrics_cache["buffer_time"], metrics_cache["buffer_ratio"]
            if metrics_basic is None and metrics_final is None:
                semantic_metrics = self.semantic_evaluation(query, query_info, method, result, baseline)
                poi_number, poi_popularity_recall, poi_popularity_accumulate, poi_category_diversity, \
                    poi_category_relevance, poi_constraint_recall, poi_constraint_penalty = semantic_metrics
            else:
                metrics_cache = metrics_basic if metrics_basic is not None else metrics_final
                poi_number, poi_popularity_recall, poi_popularity_accumulate, poi_category_diversity, \
                    poi_category_relevance, poi_constraint_recall, poi_constraint_penalty = metrics_cache["poi_number"], metrics_cache["poi_popularity_recall"], metrics_cache["poi_popularity_accumulate"], \
                        metrics_cache["poi_category_diversity"], metrics_cache["poi_category_relevance"], metrics_cache["poi_constraint_recall"], metrics_cache["poi_constraint_penalty"]
            if self.eval_sample > 0:
                query_relevance, time_quality, plan_quality, plan_objective_quality = None, None, None, None
                if metrics_final is not None:
                    query_relevance_ratio, query_relevance_win, query_relevance_tie, query_relevance_fail = metrics_final["query_relevance_ratio"], metrics_final["query_relevance_win"], metrics_final["query_relevance_tie"], metrics_final["query_relevance_fail"]
                else:
                    query_relevance_ratio, query_relevance_win, query_relevance_tie, query_relevance_fail = self.eval_sample_supplement_evaluation(query, query_info, method, result, baseline, baseline_name)
            else:
                if self.validate_llm:
                    metrics_validate = self.overall_evaluation(query, query_info, method, result, baseline, baseline_name)
                    metrics_final = {**metrics_validate, **metrics_final}
                query_relevance_ratio, query_relevance_win, query_relevance_tie, query_relevance_fail = None, None, None, None
                if self.llm_eval_model is None:
                    query_relevance, poi_relevance, time_relevance, time_quality, start_time_quality, plan_quality, plan_objective_quality = None, None, None, None, None, None, None
                elif metrics_final is not None:
                    metrics_to_be_test = []
                    for m in ["query_relevance", "poi_relevance", "time_relevance", "time_quality", "start_time_quality", "plan_quality", "plan_objective_quality"]:
                        if m not in metrics_final or metrics_final[m] is None:
                            metrics_to_be_test.append(m)
                    if len(metrics_to_be_test) > 0:
                        test_metrics = self.overall_evaluation(query, query_info, method, result, baseline, baseline_name, metrics_to_be_test)
                        metrics_final.update(test_metrics)
                    query_relevance, poi_relevance, time_relevance, time_quality, start_time_quality, plan_quality, plan_objective_quality = \
                        metrics_final["query_relevance"], metrics_final["poi_relevance"], metrics_final["time_relevance"], metrics_final["time_quality"], metrics_final["start_time_quality"], metrics_final["plan_quality"], metrics_final["plan_objective_quality"]
                else:
                    metrics_final = self.overall_evaluation(query, query_info, method, result, baseline, baseline_name)
                    query_relevance, poi_relevance, time_relevance, time_quality, start_time_quality, plan_quality, plan_objective_quality = \
                        metrics_final["query_relevance"], metrics_final["poi_relevance"], metrics_final["time_relevance"], metrics_final["time_quality"], metrics_final["start_time_quality"], metrics_final["plan_quality"], metrics_final["plan_objective_quality"]

            metrics = {
                # commonsense
                "days_accuracy": days_accuracy, 
                "failure_rate": failure_rate, 
                "repeat_rate": repeat_rate, 
                # "day_overflow_rate": day_overflow_rate, 
                # "time_overlap_rate": time_overlap_rate, 
                "time_disorder_rate": time_disorder_rate, 
                # spatial
                "average_distance": average_distance, 
                "local_optimal_distance_margin_ratio": local_optimal_distance_margin_ratio, 
                "global_optimal_distance_margin_ratio": global_optimal_distance_margin_ratio,  
                # temporal
                "shop_hours_overflow_rate": shop_hours_overflow_rate, 
                "shop_hours_overflow_ratio": shop_hours_overflow_ratio, 
                "recommended_hours_miss_rate": recommended_hours_miss_rate, 
                "recommended_hours_miss_margin": recommended_hours_miss_margin,
                "duration_underflow_rate": duration_underflow_rate, 
                "duration_underflow_margin_ratio": duration_underflow_margin_ratio, 
                "total_time": total_time,
                "buffer_time": buffer_time, 
                "buffer_ratio": buffer_ratio,  
                # semantic & constraints
                "poi_number": poi_number, 
                "poi_popularity_recall": poi_popularity_recall, 
                "poi_popularity_accumulate": poi_popularity_accumulate, 
                "poi_category_diversity": poi_category_diversity,  
                "poi_category_relevance": poi_category_relevance, 
                "poi_constraint_recall": poi_constraint_recall, 
                "poi_constraint_penalty": poi_constraint_penalty, 
                # overall
                "query_relevance": query_relevance, 
                "poi_relevance": poi_relevance, 
                "time_relevance": time_relevance, 
                "time_quality": time_quality, 
                "start_time_quality": start_time_quality, 
                "plan_quality": plan_quality, 
                "plan_objective_quality": plan_objective_quality, 
                # test
                "query_relevance_ratio": query_relevance_ratio, 
                "query_relevance_tie": query_relevance_tie, 
                "query_relevance_fail": query_relevance_fail,
                "query_relevance_win": query_relevance_win,
            }
            # print(metrics)
            # exit(-1)

            eval_path_final = os.path.join(eval_dir_final, f"{query}-{method}.pkl")
            pickle.dump(metrics, open(eval_path_final, 'wb'))

            print(f"eval {query} {method} over {baseline_name} done in {time.time() - start_time}s")

            return metrics, method
        except:
            return None, method

    # main evaluation functions
    def commonsense_evaluation(self, query, query_info, method, result, baseline):

        poi_gt_dict = {p["名称"]: p for p in query_info["poi_extract_result_improve"]}

        # days_accuracy
        days_gt = query_info["query_analysis_result"]["天数"]
        if days_gt == "无":
            days_accuracy = None
        else:
            if isinstance(days_gt, str):
                days_gt = eval(days_gt)
                assert isinstance(days_gt, int)
            days_accuracy = int(days_gt == len(result))

        # failure_rate
        result_poi_list = [p for _, p_l in result.items() for p in p_l]
        poi_check_list = [p for p in result_poi_list if p["名称"] in poi_gt_dict]
        assert len(result_poi_list) > 0
        failure_rate = float(len(result_poi_list) - len(poi_check_list)) / len(result_poi_list)
        # if failure_rate > 0:
        #     print(len(result_poi_list), set([p["名称"] for p in result_poi_list]).difference(set([p["名称"] for p in poi_check_list])))
        #     exit(0)

        # repeat_rate
        name_list = [p["名称"] for _, p_l in result.items() for p in p_l]
        name_set = set(name_list)
        repeat_rate = float(len(name_list) - len(name_set)) / len(name_list)

        # # day_overflow_rate
        # num_overflow = 0
        # for sub_title, p_l in result.items():
        #     s_time = self.parse_time_string(self.extract_bracket_content(p_l[0]["时间段"]), plan=True)[0]
        #     e_time = self.parse_time_string(self.extract_bracket_content(p_l[-1]["时间段"]), plan=True)[1]
        #     if not(s_time >= self.parse_time_string("00:00") and e_time <= self.parse_time_string("24:00")):
        #         num_overflow += 1
        # day_overflow_rate = float(num_overflow) / len(result)

        # time_disorder_rate
        num_disorder = 0
        for sub_title, p_l in result.items():
            # local_flag = False
            time_pair_l = [self.parse_time_string(self.extract_bracket_content(p["时间段"]), plan=True) for p in p_l]
            if time_pair_l[0][0] >= time_pair_l[0][1]:
                # local_flag = True
                num_disorder += 1
                continue
            for i in range(1, len(time_pair_l)):
                if time_pair_l[i][0] <= time_pair_l[i - 1][1] or time_pair_l[i][0] >= time_pair_l[i][1]:
                    # local_flag = True
                    num_disorder += 1
                    break
        time_disorder_rate = float(num_disorder) / len(result)

        return days_accuracy, failure_rate, repeat_rate, time_disorder_rate

    def spatial_evaluation(self, query, query_info, method, result, baseline):
        
        poi_gt_dict = {p["名称"]: p for p in query_info["poi_extract_result_improve"]}
        poi_list = [p for sub_title, p_l in result.items() for p in p_l]
        poi_detail_list = [poi_gt_dict[p["名称"]] if p["名称"] in poi_gt_dict else None for p in poi_list]
        num_total = len(poi_detail_list)
        poi_detail_list = [p for p in poi_detail_list if p is not None]
        if len(poi_detail_list) <= 1:
            # print(result)
            # print(poi_detail_list)
            # raise ValueError
            return None, None, None
        
        # current distance
        cur_dist = 0
        for i in range(len(poi_detail_list)-1):
            cur_dist += self.calculate_distance(poi_detail_list[i], poi_detail_list[i+1])

        # construct distance matrix
        dist_mat = np.zeros((len(poi_detail_list), len(poi_detail_list)))
        for i in range(len(poi_detail_list)):
            for j in range(i, len(poi_detail_list)):
                if i == j:
                    continue
                else:
                    d = self.calculate_distance(poi_detail_list[i], poi_detail_list[j])
                    dist_mat[i][j] = dist_mat[j][i] = d
        dist_mat_ori = deepcopy(dist_mat)

        # optimal distance
        dist_mat = deepcopy(dist_mat_ori)
        dist_mat[:, 0] = 0.
        local_permutation, local_optimal_dist = self.calculate_optimal_distance(dist_mat)
        assert int(local_optimal_dist) <= int(cur_dist)

        global_permutation, global_optimal_dist = None, None
        for i in range(len(poi_detail_list)):
            dist_mat = deepcopy(dist_mat_ori)
            dist_mat[:, i] = 0.
            permutation, optimal_dist = self.calculate_optimal_distance(dist_mat)
            if global_optimal_dist is None or optimal_dist < global_optimal_dist:
                global_permutation = permutation
                global_optimal_dist = optimal_dist
        assert int(global_optimal_dist) <= int(cur_dist)    

        # compute metrics
        average_distance = float(cur_dist) / (len(poi_detail_list) - 1)
        local_optimal_distance_margin_ratio = float((cur_dist - local_optimal_dist) / local_optimal_dist)
        global_optimal_distance_margin_ratio = float((cur_dist - global_optimal_dist) / global_optimal_dist)

        return average_distance, local_optimal_distance_margin_ratio, global_optimal_distance_margin_ratio

    def temporal_evaluation(self, query, query_info, method, result, baseline):

        poi_gt_dict = {p["名称"]: p for p in query_info["poi_extract_result_improve"]}
        result_poi_list = [p for _, p_l in result.items() for p in p_l]
        poi_check_list = [p for p in result_poi_list if p["名称"] in poi_gt_dict]
        plan_time_pair_l = [self.parse_time_string(self.extract_bracket_content(p["时间段"]), plan=True) for p in poi_check_list]
        open_time_pair_l = [self.parse_time_string(poi_gt_dict[p["名称"]]["detail"]["开放时间"]) for p in poi_check_list]
        recommend_time_pair_dict_l = [poi_gt_dict[p["名称"]]["detail"]["推荐游玩开始时间"] for p in poi_check_list]
        duration_time_l = [poi_gt_dict[p["名称"]]["detail"]["预计游玩时长"] for p in poi_check_list]

        # shop_hours_overflow_rate, shop_hours_overflow_ratio
        num_overflow = 0
        shop_hours_overflow_ratio_list = []
        for i in range(len(plan_time_pair_l)):
            time_coverage_ratio = self.calculate_time_coverage_ratio(open_time_pair_l[i], plan_time_pair_l[i])
            overflow_ratio = 1. - time_coverage_ratio
            if time_coverage_ratio < 1.:
                num_overflow += 1
            shop_hours_overflow_ratio_list.append(overflow_ratio)
        if len(plan_time_pair_l) == 0:
            shop_hours_overflow_rate, shop_hours_overflow_ratio = None, None
        else:
            shop_hours_overflow_rate = float(num_overflow) / len(plan_time_pair_l)
            shop_hours_overflow_ratio = float(np.array(shop_hours_overflow_ratio_list).mean())

        # recommended_hours_miss_rate & recommended_hours_miss_margin
        num_miss = 0
        total_miss_margin = 0.
        for i in range(len(plan_time_pair_l)):
            try:
                recommend_deviation_list = [self.calculate_time_deviation(self.parse_time_string(v, plan=True), plan_time_pair_l[i][0]) \
                    for k, v in recommend_time_pair_dict_l[i].items() if v != "不推荐"]
                if len(recommend_deviation_list) == 0:
                    recommend_deviation_list = [self.calculate_time_deviation(self.parse_time_string("0:00-24:00", plan=True), plan_time_pair_l[i][0])]
                miss_margin = min(recommend_deviation_list) * 60
            except:
                print(recommend_time_pair_dict_l[i], plan_time_pair_l[i][0])
                raise ValueError
            total_miss_margin += miss_margin
            if miss_margin > 0.:
                num_miss += 1
        recommended_hours_miss_rate = float(num_miss) / len(plan_time_pair_l) if len(plan_time_pair_l) > 0 else None
        recommended_hours_miss_margin = float(total_miss_margin) / len(plan_time_pair_l) if len(plan_time_pair_l) > 0 else None

        # duration_underflow_rate & duration_underflow_margin_ratio
        underflow_list = []
        duration_underflow_margin_ratio_list = []
        for i in range(len(plan_time_pair_l)):
            duration = duration_time_l[i]
            if "-" in duration:
                duration_min, duration_max = duration.split('-')
                duration = eval(duration_min)
            else:
                duration = eval(duration)
            ref_duration = timedelta(hours=duration).total_seconds() / 60
            if plan_time_pair_l[i][1] <= plan_time_pair_l[i][0]:
                continue
            cur_duration = (plan_time_pair_l[i][1] - plan_time_pair_l[i][0]).total_seconds() / 60
            if cur_duration < ref_duration:
                underflow_list.append(1.)
                duration_underflow_margin_ratio_list.append((ref_duration - cur_duration) / ref_duration)
            else:
                underflow_list.append(0.)
                duration_underflow_margin_ratio_list.append(0.)
        duration_underflow_rate = float(np.array(underflow_list).mean()) if len(underflow_list) > 0 else None
        duration_underflow_margin_ratio = float(np.array(duration_underflow_margin_ratio_list).mean()) if len(duration_underflow_margin_ratio_list) > 0 else None

        # total_time & buffer_time & buffer_ratio
        total_time = 0.
        buffer_time_list = []
        buffer_ratio_list = []
        for sub_title, p_l in result.items():
            local_flag = False
            s_time = self.parse_time_string(self.extract_bracket_content(p_l[0]["时间段"]), plan=True)[0]
            e_time = self.parse_time_string(self.extract_bracket_content(p_l[-1]["时间段"]), plan=True)[1]
            if e_time <= s_time:
                continue
            day_time = min((e_time - s_time).total_seconds() / 60, timedelta(days=1).total_seconds() / 60)
            total_time += day_time

            day_buffer_time = 0.
            local_buffer_time_list = []
            time_pair_l = [self.parse_time_string(self.extract_bracket_content(p["时间段"]), plan=True) for p in p_l]
            for i in range(1, len(time_pair_l)):
                if time_pair_l[i][0] > time_pair_l[i - 1][1]:
                    buffer_time = (time_pair_l[i][0] - time_pair_l[i - 1][1]).total_seconds() / 60
                else:
                    # buffer_time = 0.
                    local_flag = True
                    break
                local_buffer_time_list.append(buffer_time)
                day_buffer_time += buffer_time
            if local_flag:
                continue
            buffer_time_list.extend(local_buffer_time_list)
            if len(time_pair_l) > 1:
                buffer_ratio_list.append(day_buffer_time / day_time)
        total_time = total_time / len(result)
        buffer_time = float(np.array(buffer_time_list).mean()) if len(buffer_time_list) > 0 else None
        buffer_ratio = float(np.array(buffer_ratio_list).mean()) if len(buffer_ratio_list) > 0 else None

        return shop_hours_overflow_rate, shop_hours_overflow_ratio, recommended_hours_miss_rate, recommended_hours_miss_margin, \
               duration_underflow_rate, duration_underflow_margin_ratio, total_time, buffer_time, buffer_ratio

    def semantic_evaluation(self, query, query_info, method, result, baseline):

        poi_gt_dict = {p["名称"]: p for p in query_info["poi_extract_result_improve"]}
        poi_rank_list = [p["名称"] for p in query_info["poi_extract_result_improve"]]
        for rank_id, name in enumerate(poi_rank_list):
            poi_gt_dict[name]['rank'] = rank_id + 1
        result_poi_list = [p for _, p_l in result.items() for p in p_l]
        poi_check_list = [p for p in result_poi_list if p["名称"] in poi_gt_dict]
        rec_poi_list = [p["名称"] for p in poi_check_list]
        constraints_gt = query_info["query_analysis_result"]["约束"]
        
        # poi_number
        poi_number = len(poi_check_list)

        # poi_popularity_recall
        ref_poi_list = poi_rank_list[:poi_number]
        poi_popularity_recall = self.calculate_precision_recall_f1(rec_poi_list, ref_poi_list)["recall"]

        # poi_popularity_accumulate
        poi_popularity_accumulate = 0.
        for p in poi_check_list:
            poi_popularity_accumulate += 1. / poi_gt_dict[p["名称"]]['rank']

        # poi_category_diversity
        category_list = ["自然风光", "历史文化", "休闲娱乐", "艺术科技", "城市观光", "宗教文化"]
        rec_list = []
        for p in poi_check_list:
            rec_list.extend(poi_gt_dict[p["名称"]]["detail"]["分类"])
        poi_category_diversity = self.calculate_entropy(rec_list, category_list)

        # poi_category_relevance (f1-score)
        if "POI类别" in constraints_gt and constraints_gt["POI类别"] is not None and constraints_gt["POI类别"] != "":
            ref_poi_list = [p["名称"] for p in query_info["poi_extract_result_improve"] if constraints_gt["POI类别"] in p["detail"]["分类"]]
            poi_category_relevance = self.calculate_precision_recall_f1(rec_poi_list, ref_poi_list)
        else:
            poi_category_relevance = {}

        # poi_constraint_recall
        poi_constraint_recall = {}
        if "季节" in constraints_gt and constraints_gt["季节"] is not None and constraints_gt["季节"] != "":
            ref_poi_list = [p["名称"] for p in query_info["poi_extract_result_improve"] if p["detail"]["季节"] == "非常满足"]
            poi_constraint_recall["季节"] = self.calculate_precision_recall_f1(rec_poi_list, ref_poi_list)["recall"] \
                if len(ref_poi_list) > 0 else None
        if "节假日" in constraints_gt and constraints_gt["节假日"] is not None and constraints_gt["节假日"] != "":
            ref_poi_list = [p["名称"] for p in query_info["poi_extract_result_improve"] if p["detail"]["节假日"] == "非常满足"]
            poi_constraint_recall["节假日"] = self.calculate_precision_recall_f1(rec_poi_list, ref_poi_list)["recall"] \
                if len(ref_poi_list) > 0 else None
        if "受众" in constraints_gt and constraints_gt["受众"] is not None and constraints_gt["受众"] != "":
            ref_poi_list = [p["名称"] for p in query_info["poi_extract_result_improve"] if p["detail"]["受众"] == "非常满足"]
            poi_constraint_recall["受众"] = self.calculate_precision_recall_f1(rec_poi_list, ref_poi_list)["recall"] \
                if len(ref_poi_list) > 0 else None

        # poi_constraint_penalty
        poi_constraint_penalty = {}
        if "季节" in constraints_gt and constraints_gt["季节"] is not None and constraints_gt["季节"] != "":
            ref_poi_list = [p["名称"] for p in query_info["poi_extract_result_improve"] if p["detail"]["季节"] == "非常不满足"]
            poi_constraint_penalty["季节"] = self.calculate_precision_recall_f1(rec_poi_list, ref_poi_list)["recall"] \
                if len(ref_poi_list) > 0 else None
        if "节假日" in constraints_gt and constraints_gt["节假日"] is not None and constraints_gt["节假日"] != "":
            ref_poi_list = [p["名称"] for p in query_info["poi_extract_result_improve"] if p["detail"]["节假日"] == "非常不满足"]
            poi_constraint_penalty["节假日"] = self.calculate_precision_recall_f1(rec_poi_list, ref_poi_list)["recall"] \
                if len(ref_poi_list) > 0 else None
        if "受众" in constraints_gt and constraints_gt["受众"] is not None and constraints_gt["受众"] != "":
            ref_poi_list = [p["名称"] for p in query_info["poi_extract_result_improve"] if p["detail"]["受众"] == "非常不满足"]
            poi_constraint_penalty["受众"] = self.calculate_precision_recall_f1(rec_poi_list, ref_poi_list)["recall"] \
                if len(ref_poi_list) > 0 else None

        return poi_number, poi_popularity_recall, poi_popularity_accumulate, poi_category_diversity, \
               poi_category_relevance, poi_constraint_recall, poi_constraint_penalty

    def overall_evaluation(self, query, query_info, method, result, baseline, baseline_name, metric_list="all"):
        
        if self.validate_llm:
            validate_dir = "validate_llm_eval"
            if not os.path.exists(validate_dir):
                os.makedirs(validate_dir)
            validate_PR_dir = f"{validate_dir}/{method}/PR/"
            if not os.path.exists(validate_PR_dir):
                os.makedirs(validate_PR_dir)
            validate_TR_dir = f"{validate_dir}/{method}/TR/"
            if not os.path.exists(validate_TR_dir):
                os.makedirs(validate_TR_dir)
            validate_STR_dir = f"{validate_dir}/{method}/STR/"
            if not os.path.exists(validate_STR_dir):
                os.makedirs(validate_STR_dir)
        else:
            validate_dir = None

        poi_gt_info = query_info["poi_extract_result_improve"]
        return_results = {}

        # # query_relevance
        # if metric_list == "all" or "query_relevance" in metric_list:
        #     if query not in self.query_classify_dict["generic"]:
        #         query_relevance = self.multiple_compare(str(self.config.query_relevance_compare_prompt), query, method, result, baseline, baseline_name, metric_name="query_relevance")
        #     else:
        #         query_relevance = None
        #     return_results["query_relevance"] = query_relevance
        return_results["query_relevance"] = None
            
        # poi_relevance
        if self.validate_llm:
            local_logger = get_logger(f"{query}_{method}_PR", validate_PR_dir)
        if metric_list == "all" or "poi_relevance" in metric_list:
            poi_list = [p["名称"] for _, p_l in result.items() for p in p_l]
            if len(poi_list) > 0 and query not in self.query_classify_dict["generic"]:
                prompt = str(self.config.poi_relevance_prompt) % (query, poi_list)
                self.logger.info(f"poi_relevance {method} prompt:\n{prompt}\n")
                if self.validate_llm:
                    local_logger.info(f"poi_relevance {method} prompt:\n{prompt}\n")
                poi_relevance = self.request_llm.get_llm_result(prompt)
                self.logger.info(f"poi_relevance result {method}:\n{poi_relevance}\n")
                if self.validate_llm:
                    local_logger.info(f"poi_relevance result {method}:\n{poi_relevance}\n")
                poi_relevance_dict = self.request_llm.parse_json_response(poi_relevance, self.logger)
                s1 = set(poi_list)
                poi_relevance_dict = {k: v for k, v in poi_relevance_dict.items() if v in ["满足", "不满足"]}
                s2 = set(poi_relevance_dict.keys())
                poi_relevance_dict = {k: v for k, v in poi_relevance_dict.items() if k not in s2.difference(s1)}
                retry = 0
                while len(s1.difference(s2)) > 0 and retry < 10:
                    print(s1.difference(s2), s2.difference(s1), s1.symmetric_difference(s2))
                    prompt = str(self.config.poi_relevance_prompt) % (query, {n for n in s1.difference(s2)})
                    self.logger.info(f"refine poi_relevance {method} prompt:\n{prompt}\n")
                    if self.validate_llm:
                        local_logger.info(f"refine poi_relevance {method} prompt:\n{prompt}\n")
                    poi_relevance_refine = self.request_llm.get_llm_result(prompt)
                    self.logger.info(f"refine poi_relevance result {method}:\n{poi_relevance_refine}\n")
                    if self.validate_llm:
                        local_logger.info(f"refine poi_relevance result {method}:\n{poi_relevance_refine}\n")
                    poi_relevance_refine_dict = self.request_llm.parse_json_response(poi_relevance_refine, self.logger)
                    poi_relevance_dict = {**poi_relevance_dict, **poi_relevance_refine_dict}
                    poi_relevance_dict = {k: v for k, v in poi_relevance_dict.items() if v in ["满足", "不满足"]}
                    s2 = set(poi_relevance_dict.keys())
                    poi_relevance_dict = {k: v for k, v in poi_relevance_dict.items() if k not in s2.difference(s1)}
                    retry += 1
                assert len(s1.difference(s2)) == 0
                poi_relevance = [1. if s == "满足" else 0. for s in list(poi_relevance_dict.values())]
                poi_relevance = float(np.array(poi_relevance).mean()) if len(poi_relevance) > 0 else None
            else:
                poi_relevance = None
            return_results["poi_relevance"] = poi_relevance

        # time_relevance
        if self.validate_llm:
            local_logger = get_logger(f"{query}_{method}_TR", validate_TR_dir)
        if metric_list == "all" or "time_relevance" in metric_list:
            time_list = [f"{_}:" + p["时间段"] for _, p_l in result.items() for p in p_l]
            if len(time_list) > 0 and query in self.query_classify_dict["personal"]["total"]:
                prompt = str(self.config.time_relevance_prompt) % (query, time_list)
                self.logger.info(f"time_relevance {method} prompt:\n{prompt}\n")
                if self.validate_llm:
                    local_logger.info(f"time_relevance {method} prompt:\n{prompt}\n")
                time_relevance = self.request_llm.get_llm_result(prompt)
                self.logger.info(f"time_relevance result {method}:\n{time_relevance}\n")
                if self.validate_llm:
                    local_logger.info(f"time_relevance result {method}:\n{time_relevance}\n")
                time_relevance_dict = self.request_llm.parse_json_response(time_relevance, self.logger)
                s1 = set(time_list)
                time_relevance_dict = {k: v for k, v in time_relevance_dict.items() if v in ["满足", "不满足"]}
                s2 = set(time_relevance_dict.keys())
                time_relevance_dict = {k: v for k, v in time_relevance_dict.items() if k not in s2.difference(s1)}
                retry = 0
                while len(s1.difference(s2)) > 0 and retry < 10:
                    print(s1.difference(s2), s2.difference(s1), s1.symmetric_difference(s2))
                    prompt = str(self.config.time_relevance_prompt) % (query, {n for n in s1.difference(s2)})
                    self.logger.info(f"refine time_relevance {method} prompt:\n{prompt}\n")
                    if self.validate_llm:
                        local_logger.info(f"refine time_relevance {method} prompt:\n{prompt}\n")
                    time_relevance_refine = self.request_llm.get_llm_result(prompt)
                    self.logger.info(f"refine time_relevance result {method}:\n{time_relevance_refine}\n")
                    if self.validate_llm:
                        local_logger.info(f"refine time_relevance result {method}:\n{time_relevance_refine}\n")
                    time_relevance_refine_dict = self.request_llm.parse_json_response(time_relevance_refine, self.logger)
                    time_relevance_dict = {**time_relevance_dict, **time_relevance_refine_dict}
                    time_relevance_dict = {k: v for k, v in time_relevance_dict.items() if v in ["满足", "不满足"]}
                    s2 = set(time_relevance_dict.keys())
                    time_relevance_dict = {k: v for k, v in time_relevance_dict.items() if k not in s2.difference(s1)}
                    retry += 1
                assert len(s1.difference(s2)) == 0
                time_relevance = [1. if s == "满足" else 0. for s in list(time_relevance_dict.values())]
                time_relevance = float(np.array(time_relevance).mean()) if len(time_relevance) > 0 else None
            else:
                time_relevance = None
            return_results["time_relevance"] = time_relevance
        
        # # time_quality
        # if metric_list == "all" or "time_quality" in metric_list:
        #     time_info_dict = {p["名称"]: p["时间段"] for _, p_l in result.items() for p in p_l}
        #     if len(time_info_dict) > 0:
        #         prompt = str(self.config.time_quality_prompt) % (query, time_info_dict)
        #         self.logger.info(f"time quality {method} prompt:\n{prompt}\n")
        #         time_quality = self.request_llm.get_llm_result(prompt)
        #         self.logger.info(f"time quality result {method}:\n{time_quality}\n")
        #         time_quality_dict = self.request_llm.parse_json_response(time_quality, self.logger)
        #         s1 = set(time_info_dict.keys())
        #         time_quality_dict = {k: v for k, v in time_quality_dict.items() if v in ["合适", "不合适"]}
        #         s2 = set(time_quality_dict.keys())
        #         time_quality_dict = {k: v for k, v in time_quality_dict.items() if k not in s2.difference(s1)}
        #         retry = 0
        #         while len(s1.difference(s2)) > 0 and retry < 10:
        #             print(s1.difference(s2), s2.difference(s1), s1.symmetric_difference(s2))
        #             prompt = str(self.config.time_quality_prompt) % (query, {n: time_info_dict[n] for n in s1.difference(s2)})
        #             self.logger.info(f"refine time quality {method} prompt:\n{prompt}\n")
        #             time_quality_refine = self.request_llm.get_llm_result(prompt)
        #             self.logger.info(f"refine time quality result {method}:\n{time_quality_refine}\n")
        #             time_quality_refine_dict = self.request_llm.parse_json_response(time_quality_refine, self.logger)
        #             time_quality_dict = {**time_quality_dict, **time_quality_refine_dict}
        #             time_quality_dict = {k: v for k, v in time_quality_dict.items() if v in ["合适", "不合适"]}
        #             s2 = set(time_quality_dict.keys())
        #             time_quality_dict = {k: v for k, v in time_quality_dict.items() if k not in s2.difference(s1)}
        #             retry += 1
        #         time_quality = [1. if s == "合适" else 0. for s in list(time_quality_dict.values())]
        #         time_quality = float(np.array(time_quality).mean()) if len(time_quality) > 0 else None
        #     else:
        #         time_quality = None
        #     return_results["time_quality"] = time_quality
        return_results["time_quality"] = None
        
        # start_time_quality
        if self.validate_llm:
            local_logger = get_logger(f"{query}_{method}_STR", validate_STR_dir)
        if metric_list == "all" or "start_time_quality" in metric_list:
            try:
                time_info_dict = {p["名称"]: self.parse_time_string(self.extract_bracket_content(p["时间段"]), plan=True)[0].strftime("%H:%M") for _, p_l in result.items() for p in p_l}
            except:
                print(result)
                raise ValueError
            if len(time_info_dict) > 0:
                prompt = str(self.config.start_time_quality_prompt) % (query, time_info_dict)
                self.logger.info(f"start_time_quality {method} prompt:\n{prompt}\n")
                if self.validate_llm:
                    local_logger.info(f"start_time_quality {method} prompt:\n{prompt}\n")
                start_time_quality = self.request_llm.get_llm_result(prompt)
                self.logger.info(f"start_time_quality result {method}:\n{start_time_quality}\n")
                if self.validate_llm:
                    local_logger.info(f"start_time_quality result {method}:\n{start_time_quality}\n")
                start_time_quality_dict = self.request_llm.parse_json_response(start_time_quality, self.logger)
                s1 = set(time_info_dict.keys())
                start_time_quality_dict = {k: v for k, v in start_time_quality_dict.items() if v in ["合适", "不合适"]}
                s2 = set(start_time_quality_dict.keys())
                start_time_quality_dict = {k: v for k, v in start_time_quality_dict.items() if k not in s2.difference(s1)}
                retry = 0
                while len(s1.difference(s2)) > 0 and retry < 10:
                    print(s1.difference(s2), s2.difference(s1), s1.symmetric_difference(s2))
                    prompt = str(self.config.start_time_quality_prompt) % (query, {n: time_info_dict[n] for n in s1.difference(s2)})
                    self.logger.info(f"refine start_time_quality {method} prompt:\n{prompt}\n")
                    if self.validate_llm:
                        local_logger.info(f"refine start_time_quality {method} prompt:\n{prompt}\n")
                    start_time_quality_refine = self.request_llm.get_llm_result(prompt)
                    self.logger.info(f"refine start_time_quality result {method}:\n{start_time_quality_refine}\n")
                    if self.validate_llm:
                        local_logger.info(f"refine start_time_quality result {method}:\n{start_time_quality_refine}\n")
                    start_time_quality_refine_dict = self.request_llm.parse_json_response(start_time_quality_refine, self.logger)
                    start_time_quality_dict = {**start_time_quality_dict, **start_time_quality_refine_dict}
                    start_time_quality_dict = {k: v for k, v in start_time_quality_dict.items() if v in ["合适", "不合适"]}
                    s2 = set(start_time_quality_dict.keys())
                    start_time_quality_dict = {k: v for k, v in start_time_quality_dict.items() if k not in s2.difference(s1)}
                    retry += 1
                assert len(s1.difference(s2)) == 0
                start_time_quality = [1. if s == "合适" else 0. for s in list(start_time_quality_dict.values())]
                start_time_quality = float(np.array(start_time_quality).mean()) if len(start_time_quality) > 0 else None
            else:
                start_time_quality = None
            return_results["start_time_quality"] = start_time_quality

        # # plan_quality
        # if metric_list == "all" or "plan_quality" in metric_list:
        #     plan_quality = self.multiple_compare(str(self.config.plan_quality_compare_prompt), query, method, result, baseline, baseline_name, metric_name="plan_quality")
        #     return_results["plan_quality"] = plan_quality
        return_results["plan_quality"] = None

        # # plan_objective_quality
        # if metric_list == "all" or "plan_objective_quality" in metric_list:
        #     plan_objective_quality = self.multiple_compare(str(self.config.plan_objective_quality_compare_prompt, baseline_name, metric_name="plan_objective_quality"), 
        #         query, method, result, baseline, poi_gt_info)
        #     return_results["plan_objective_quality"] = plan_objective_quality
        return_results["plan_objective_quality"] = None

        return return_results

    def eval_sample_supplement_evaluation(self, query, query_info, method, result, baseline, baseline_name):

        poi_gt_info = query_info["poi_extract_result_improve"]

        # query_relevance
        query_relevance = self.multiple_compare(str(self.config.query_relevance_compare_prompt), query, method, result, baseline, baseline_name, metric_name="query_relevance", print_details=True)
        query_relevance_win, query_relevance_tie, query_relevance_fail = 0., 0., 0.
        if query_relevance == 1.:
            query_relevance_win += 1.
        elif query_relevance == 0.:
            query_relevance_fail += 1.
        elif query_relevance == -1.:
            query_relevance_tie += 1.
        elif query_relevance is None:
            query_relevance_win, query_relevance_tie, query_relevance_fail = None, None, None
        else:
            raise ValueError
        
        # query_relevance_ratio
        poi_list = [p["名称"] for _, p_l in result.items() for p in p_l]
        prompt = str(self.config.query_relevance_ratio_prompt) % (query, poi_list)
        self.logger.info(f"query_relevance_ratio {method} prompt:\n{prompt}\n")
        query_relevance_ratio = self.request_llm.get_llm_result(prompt)
        self.logger.info(f"query_relevance_ratio result {method}:\n{query_relevance_ratio}\n")
        query_relevance_ratio_dict = self.request_llm.parse_json_response(query_relevance_ratio, self.logger)
        s1 = set(poi_list)
        query_relevance_ratio_dict = {k: v for k, v in query_relevance_ratio_dict.items() if v in ["满足", "不满足"]}
        s2 = set(query_relevance_ratio_dict.keys())
        query_relevance_ratio_dict = {k: v for k, v in query_relevance_ratio_dict.items() if k not in s2.difference(s1)}
        retry = 0
        while len(s1.difference(s2)) > 0 and retry < 10:
            print(s1.difference(s2), s2.difference(s1), s1.symmetric_difference(s2))
            prompt = str(self.config.query_relevance_ratio_prompt) % (query, {n for n in s1.difference(s2)})
            self.logger.info(f"refine query_relevance_ratio {method} prompt:\n{prompt}\n")
            query_relevance_ratio_refine = self.request_llm.get_llm_result(prompt)
            self.logger.info(f"refine query_relevance_ratio result {method}:\n{query_relevance_ratio_refine}\n")
            query_relevance_ratio_refine_dict = self.request_llm.parse_json_response(query_relevance_ratio_refine, self.logger)
            query_relevance_ratio_dict = {**query_relevance_ratio_dict, **query_relevance_ratio_refine_dict}
            query_relevance_ratio_dict = {k: v for k, v in query_relevance_ratio_dict.items() if v in ["满足", "不满足"]}
            s2 = set(query_relevance_ratio_dict.keys())
            query_relevance_ratio_dict = {k: v for k, v in query_relevance_ratio_dict.items() if k not in s2.difference(s1)}
            retry += 1
        query_relevance_ratio = [1. if s == "满足" else 0. for s in list(query_relevance_ratio_dict.values())]
        query_relevance_ratio = float(np.array(query_relevance_ratio).mean()) if len(query_relevance_ratio) > 0 else None

        return query_relevance_ratio, query_relevance_win, query_relevance_tie, query_relevance_fail

    # auxiliary evaluation functions
    def extract_bracket_content(self, text):
        
        # text = text.replace('(','（')
        # text = text.replace(')','）')
        # pattern = r'\（(.*?)\）'
        # result = re.findall(pattern, text)
        text = text.replace(' ', '')
        text = text.replace('-下午', '-')
        text = text.replace('-晚上', '-')
        text = text.replace('-傍晚', '-')
        text = text.replace('-中午', '-')
        pattern = r'\d{1,2}:\d{2}-\d{1,2}:\d{2}'
        match = re.search(pattern, text)
        result = re.findall(pattern, text)
        try:
            assert len(result) > 0
        except:
            if "全天" in text:
                result = ["00:00-23:59"]
            else:
                print(text)
                raise ValueError

        return ",".join(result)

    def parse_time_string(self, time_str, plan=False):
        
        if plan:
            if ',' in time_str:
                t_l = time_str.split(',')
                res = [self.parse_time_string(t, plan=True) for t in t_l]
                res_list = []
                for r in res:
                    if isinstance(r, list):
                        for v in r:
                            assert isinstance(v, tuple)
                            res_list.append(v[0])
                            res_list.append(v[1])
                    else:
                        res_list.append(r[0])
                        res_list.append(r[1])
                # print(time_str, min(res_list), max(res_list))
                return min(res_list), max(res_list)
            if '&' in time_str:
                t_l = time_str.split('&')
                res = [self.parse_time_string(t, plan=True) for t in t_l]
                res_list = []
                for r in res:
                    if isinstance(r, list):
                        for v in r:
                            assert isinstance(v, tuple)
                            res_list.append(v[0])
                            res_list.append(v[1])
                    else:
                        res_list.append(r[0])
                        res_list.append(r[1])
                # print(time_str, min(res_list), max(res_list))
                return min(res_list), max(res_list)
            if '-' in time_str:
                start_str, end_str = time_str.split('-')
                if start_str.startswith("24:00"):
                    start_str = "23:59"
                if end_str.startswith("24:00"):
                    end_str = "23:59"
                start_time = datetime.strptime(start_str.strip(), '%H:%M')
                end_time = datetime.strptime(end_str.strip(), '%H:%M')
                # assert start_time < end_time
                if start_time > end_time:
                    assert end_time < datetime.strptime("6:00", '%H:%M')
                    return (start_time, datetime.strptime("23:59", '%H:%M'))
                return start_time, end_time   
            else:
                if time_str.startswith("24:00"):
                    time_str = "23:59"
                return datetime.strptime(time_str.strip(), '%H:%M')    
            
        if ',' in time_str:
            t_l = time_str.split(',')
            res = [self.parse_time_string(t) for t in t_l]
            res_list = []
            for r in res:
                if isinstance(r, list):
                    for v in r:
                        assert isinstance(v, tuple)
                        res_list.append(v)
                else:
                    res_list.append(r)
            return res_list
        
        if '-' in time_str:
            start_str, end_str = time_str.split('-')
            if start_str.startswith("24:00"):
                start_str = "23:59"
            if end_str.startswith("24:00"):
                end_str = "23:59"
            start_time = datetime.strptime(start_str.strip(), '%H:%M')
            end_time = datetime.strptime(end_str.strip(), '%H:%M')
            if start_time >= end_time:
                assert end_time < datetime.strptime("10:00", '%H:%M')
                return [(datetime.strptime("0:00", '%H:%M'), end_time), (start_time, datetime.strptime("23:59", '%H:%M'))]
            return start_time, end_time
        else:
            if time_str.startswith("24:00"):
                time_str = "23:59"
            return datetime.strptime(time_str.strip(), '%H:%M')     

    def calculate_time_deviation(self, time_pair, time_to_check):
        
        if not isinstance(time_pair, tuple):
            time_pair = (time_pair - timedelta(hours=0.5), time_pair + timedelta(hours=0.5),)

        if time_pair[0] <= time_to_check and time_to_check <= time_pair[1]:
            return 0
        elif time_to_check < time_pair[0]:
            deviation = (time_pair[0] - time_to_check).total_seconds() / 60
            return deviation
        elif time_to_check > time_pair[1]:
            deviation = (time_to_check - time_pair[1]).total_seconds() / 60
            return deviation

    def calculate_time_coverage_ratio(self, time_a, time_b):
        
        if isinstance(time_a, list):
            return sum([self.calculate_time_coverage_ratio(t, time_b) for t in time_a])

        start_a, start_b = time_a[0], time_b[0]
        end_a, end_b = time_a[1], time_b[1]
        overlap_start = max(start_a, start_b)
        overlap_end = min(end_a, end_b)

        if overlap_start < overlap_end:
            overlap_duration = (overlap_end - overlap_start).total_seconds() / 60
        else:
            overlap_duration = 0.
        total_duration_b = (end_b - start_b).total_seconds() / 60
        time_coverage_ratio = overlap_duration / total_duration_b if total_duration_b > 0 else 0

        return time_coverage_ratio

    def calculate_distance(self, p_1, p_2):
        return geodesic((p_1["detail"]["纬度"], p_1["detail"]["经度"]), 
                        (p_2["detail"]["纬度"], p_2["detail"]["经度"])).m

    def calculate_optimal_distance(self, dist_mat):
        if len(dist_mat) > 15:
            permutation_1, _ = solve_tsp_simulated_annealing(dist_mat)
            permutation, optimal_dist = solve_tsp_local_search(dist_mat, x0=permutation_1)
        else:
            permutation, optimal_dist = solve_tsp_dynamic_programming(dist_mat)
        
        return permutation, optimal_dist

    def calculate_precision_recall_f1(self, recommended_set, ground_truth_set):
        
        recommended_set = set(recommended_set)
        ground_truth_set = set(ground_truth_set)
        true_positives = recommended_set.intersection(ground_truth_set)
        # print("####\n", ground_truth_set, "\n", true_positives, "\n", recommended_set, "\n####")
        if len(recommended_set) == 0 or len(ground_truth_set) == 0:
            return {
                "precision": None,
                "recall": None,
                "f1_score": None        
            }
        
        precision = len(true_positives) / len(recommended_set)
        recall = len(true_positives) / len(ground_truth_set)
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    def calculate_entropy(self, recommended_labels, all_labels):

        label_counts = Counter(recommended_labels)
        # print("####\n", label_counts, "\n####")
        total_count = len(recommended_labels)
        probabilities = [label_counts[label] / total_count if label in label_counts else 0 for label in all_labels]
        entropy_value = entropy(probabilities, base=2)
        if np.isnan(entropy_value):
            entropy_value = None
        
        return entropy_value

    def multiple_compare(self, prompt_template, query, method, result, baseline, baseline_name, metric_name, poi_gt_info=None, N=1, print_details=False):
        
        if method == baseline_name:
            return None
        
        metric_dict = {
            "result_win": 0, 
            "baseline_win": 0, 
            "tie": 0
        }

        for compare_idx in range(N):
            
            if poi_gt_info is None:
                prompt = prompt_template % (query, result, baseline)
            else:
                prompt = prompt_template % (query, poi_gt_info, result, baseline)
            self.logger.info(f"compare prompt {metric_name} result-baseline idx{compare_idx} {method}:\n{prompt}\n")
            metric = self.request_llm.get_llm_result(prompt)
            self.logger.info(f"compare result {metric_name} result-baseline idx{compare_idx} {method}:\n{metric}\n")
            metric = self.request_llm.parse_json_response(metric, self.logger)
            if metric["评估结果"] == "前者":
                metric_dict["result_win"] += 1
            elif metric["评估结果"] == "后者":
                metric_dict["baseline_win"] += 1
            elif metric["评估结果"] == "平局":
                metric_dict["tie"] += 1
            else:
                raise ValueError

            if poi_gt_info is None:
                prompt = prompt_template % (query, baseline, result)
            else:
                prompt = prompt_template % (query, poi_gt_info, baseline, result)
            self.logger.info(f"compare prompt {metric_name} baseline-result idx{compare_idx} {method}:\n{prompt}\n")
            metric = self.request_llm.get_llm_result(prompt)
            self.logger.info(f"compare result {metric_name} baseline-result idx{compare_idx} {method}:\n{metric}\n")
            metric = self.request_llm.parse_json_response(metric, self.logger)
            if metric["评估结果"] == "前者":
                metric_dict["baseline_win"] += 1
            elif metric["评估结果"] == "后者":
               metric_dict["result_win"] += 1
            elif metric["评估结果"] == "平局":
                metric_dict["tie"] += 1
            else:
                raise ValueError

        # print(metric_dict)
        if print_details:
            if metric_dict["result_win"] > metric_dict["baseline_win"]:
                return 1.
            elif metric_dict["result_win"] < metric_dict["baseline_win"]:
                return 0.
            else:
                return -1.          

        if metric_dict["result_win"] > metric_dict["baseline_win"]:
            return 1.
        elif metric_dict["result_win"] < metric_dict["baseline_win"]:
            return 0.
        else:
            return 0.

    def _fetch_list_results(self, prompt_list):

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.request_llm.fetch_content_result, prompt, index)
                       for index, prompt in enumerate(prompt_list)]
            result_list = [future.result() for future in concurrent.futures.as_completed(futures)]

        result_list = sorted(result_list, key=lambda x: x[1])
        return [result[0] for result in result_list]
    
    def _fetch_dict_results(self, prompt_dict):

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.request_llm.fetch_content_result, prompt, k)
                       for k, prompt in prompt_dict.items()]
            result_list = [future.result() for future in concurrent.futures.as_completed(futures)]

        result = {}
        for r in result_list:
            result[r[1]] = r[0]
        return result

def load_queries(filepath):

    query_list = []
    with codecs.open(filepath, "rb", "utf-8") as fin:
        for line in fin:
            query = line.strip()
            query_list.append(query)

    return query_list

def refine_eval_json(base_model, llm_eval_model, eval_sample, test_few, include_mas, include_rag, rebuttal, specify=False, use_both_eval=True):
    
    use_both_eval = False
    llm_eval_model = "qwen25-72b"
    
    if eval_sample > 0:
        eval_dir = f"eval_results{f'_mas' if include_mas else ''}{f'_rag' if include_rag else ''}_{base_model}/eval_sample"
    elif llm_eval_model is not None:
        eval_dir = f"eval_results{f'_mas' if include_mas else ''}{f'_rag' if include_rag else ''}_{base_model}/{llm_eval_model}"
    else:
        eval_dir = f"eval_results{f'_mas' if include_mas else ''}{f'_rag' if include_rag else ''}_{base_model}"
    json_files = [f for f in os.listdir(eval_dir) if f.endswith(".json")]
    
    def load_df(root_dir, fname):
        with open(os.path.join(root_dir, fname), 'r') as f:
            data = json.load(f)["metrics"]
        data_ori = deepcopy(data)
        for m, m_res in data_ori.items():
            del_list = []
            for k, v in m_res.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        data[m][f"{k}_{kk}"] = vv
                    del_list.append(k)
                elif math.isnan(v):
                    del_list.append(k)
            for k in del_list:
                del data[m][k]
        df = pd.DataFrame.from_dict(data, orient='index').reset_index()
        df = df.rename(columns={'index': 'Method'})   
        
        if rebuttal:
            custom_order = ["reference"]
        elif base_model.startswith("deepseek"):
            custom_order = ["given_direct_objective", "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_one", 
                        "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_abstractive",]
        elif include_rag:
            # custom_order = [
            #     "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_N7", "given_direct_objective_retrieval_N6", "given_direct_objective_retrieval_N5", 
            #     "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_N3", "given_direct_objective_retrieval_N2", "given_direct_objective_retrieval_one", 
            #     "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_abstractive", 
            #     "given_direct_objective_retrieval_all_clean", "given_direct_objective_retrieval_N7_clean", "given_direct_objective_retrieval_N6_clean", "given_direct_objective_retrieval_N5_clean", 
            #     "given_direct_objective_retrieval_half_clean", "given_direct_objective_retrieval_N3_clean", "given_direct_objective_retrieval_N2_clean", "given_direct_objective_retrieval_one_clean", 
            #     "given_direct_objective_retrieval_selective_half_clean", "given_direct_objective_retrieval_selective_one_clean", "given_direct_objective_retrieval_abstractive_clean", 
            # ]
            custom_order = [
                "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_all_clean", 
                "given_direct_objective_retrieval_N7", "given_direct_objective_retrieval_N7_clean", 
                "given_direct_objective_retrieval_N6", "given_direct_objective_retrieval_N6_clean", 
                "given_direct_objective_retrieval_N5", "given_direct_objective_retrieval_N5_clean", 
                "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_half_clean", 
                "given_direct_objective_retrieval_N3", "given_direct_objective_retrieval_N3_clean", 
                "given_direct_objective_retrieval_N2", "given_direct_objective_retrieval_N2_clean", 
                "given_direct_objective_retrieval_one", "given_direct_objective_retrieval_one_clean", 
                "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_half_clean", 
                "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_selective_one_clean", 
                "given_direct_objective_retrieval_abstractive", "given_direct_objective_retrieval_abstractive_clean", 
            ]
        else:
            custom_order = ["given_direct_objective", "given_cot_objective", "given_refine_objective", 
                        "multi_agent_collaboration", "multi_agent_debate",
                        "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_one", 
                        "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_abstractive", ]
        if include_mas:
            # method_list = []
            custom_order.extend(["evolutionary_optimize"])
            
        df = df.set_index('Method').reindex(custom_order).reset_index()
        
        name_mapping = {
            "given_direct_objective": "Direct", 
            "given_cot_objective": "CoT", 
            "given_refine_objective": "Reflextion", 
            "multi_agent_collaboration": "MAC", 
            "multi_agent_debate": "MAD",
            "given_direct_objective_retrieval_all": "RAG (M=8)", 
            "given_direct_objective_retrieval_N7": "RAG (M=7)", 
            "given_direct_objective_retrieval_N6": "RAG (M=6)", 
            "given_direct_objective_retrieval_N5": "RAG (M=5)", 
            "given_direct_objective_retrieval_half": "RAG (M=4)", 
            "given_direct_objective_retrieval_N3": "RAG (M=3)", 
            "given_direct_objective_retrieval_N2": "RAG (M=2)", 
            "given_direct_objective_retrieval_one": "RAG (M=1)", 
            "given_direct_objective_retrieval_selective_half": "RAG + Extr. (M=4)", 
            "given_direct_objective_retrieval_selective_one": "RAG + Extr. (M=1)", 
            "given_direct_objective_retrieval_abstractive": "RAG + Abst.", 
            "given_direct_objective_retrieval_all_clean": "RAG (M=8)", 
            "given_direct_objective_retrieval_N7_clean": "RAG (M=7)", 
            "given_direct_objective_retrieval_N6_clean": "RAG (M=6)", 
            "given_direct_objective_retrieval_N5_clean": "RAG (M=5)", 
            "given_direct_objective_retrieval_half_clean":"RAG (M=4)", 
            "given_direct_objective_retrieval_N3_clean": "RAG (M=3)", 
            "given_direct_objective_retrieval_N2_clean": "RAG (M=2)", 
            "given_direct_objective_retrieval_one_clean": "RAG (M=1)", 
            "given_direct_objective_retrieval_selective_half_clean": "RAG + Extr. (M=4)", 
            "given_direct_objective_retrieval_selective_one_clean": "RAG + Extr. (M=1)", 
            "given_direct_objective_retrieval_abstractive_clean": "RAG + Abst.", 
            "evolutionary_optimize": "Ours",
            "reference": "Reference"
        }
        df['Method'] = df['Method'].map(name_mapping)
        
        return df
    
    for filename in tqdm(json_files):
        df = load_df(eval_dir, filename)
        if use_both_eval and llm_eval_model is not None:
            assert llm_eval_model in ["qwen25-72b", "gpt-4o"]
            other_llm_eval_model = "gpt-4o" if llm_eval_model == "qwen25-72b" else "qwen25-72b"
            df2 = load_df(f"eval_results{f'_mas' if include_mas else ''}{f'_rag' if include_rag else ''}_{base_model}/{other_llm_eval_model}", filename)
            
            if llm_eval_model == "qwen25-72b":
                df = df.rename(columns={
                    "start_time_quality": "start_time_quality_qwen", 
                    "poi_relevance": "poi_relevance_qwen", 
                    "time_relevance": "time_relevance_qwen"
                    })
                columns_to_copy = ["start_time_quality", "poi_relevance", "time_relevance"]
                existing_columns = [col for col in columns_to_copy if col in df2.columns]
                df = pd.concat([df, df2[existing_columns]], axis=1)
                df = df.rename(columns={
                    "start_time_quality": "start_time_quality_gpt", 
                    "poi_relevance": "poi_relevance_gpt", 
                    "time_relevance": "time_relevance_gpt"
                    })
            else:
                df = df.rename(columns={
                    "start_time_quality": "start_time_quality_gpt", 
                    "poi_relevance": "poi_relevance_gpt", 
                    "time_relevance": "time_relevance_gpt"
                    })
                columns_to_copy = ["start_time_quality", "poi_relevance", "time_relevance"]
                existing_columns = [col for col in columns_to_copy if col in df2.columns]
                df = pd.concat([df, df2[existing_columns]], axis=1)     
                df = df.rename(columns={
                    "start_time_quality": "start_time_quality_qwen", 
                    "poi_relevance": "poi_relevance_qwen", 
                    "time_relevance": "time_relevance_qwen"
                    })           
                    
        def highlight_extremes(val, col):
            """将最大值和最小值加粗"""
            if pd.isna(val):
                return val
            if val == df[col].max():
                return f"**{val}**"
            if val == df[col].min():
                return f"<u>{val}</u>"
            return val
        
        # specify
        if specify:
            name_mapping = { 
                "days_accuracy": "DA", 
                "failure_rate": "FR", 
                "repeat_rate": "RR", 
                "time_disorder_rate": "TDR", 
                "shop_hours_overflow_ratio": "SOR", 
                "global_optimal_distance_margin_ratio": "DMR", 
                "duration_underflow_margin_ratio": "DUR", 
                "buffer_ratio": "TBR", 
                "start_time_quality": "STR", 
                "start_time_quality_gpt": "STRG", 
                "start_time_quality_qwen": "STRQ", 
                "poi_popularity_recall": "PP", 
                "poi_relevance": "PR", 
                "poi_relevance_gpt": "PRG", 
                "poi_relevance_qwen": "PRQ", 
                "time_relevance": "TSR", 
                "time_relevance_gpt": "TSRG", 
                "time_relevance_qwen": "TSRQ", 
            }
            df = df.rename(columns=name_mapping)
            if "STRG" in df.columns and "STRQ" in df.columns:
                df["STRA"] = df[["STRG", "STRQ"]].mean(axis=1)
            if "PRG" in df.columns and "PRQ" in df.columns:
                df["PRA"] = df[["PRG", "PRQ"]].mean(axis=1)
            if "TSRG" in df.columns and "TSRQ" in df.columns:
                df["TSRA"] = df[["TSRG", "TSRQ"]].mean(axis=1)
            candidate_cols = ["Method", "FR", "RR", "SOR", "DMR", "DUR", "TBR", "STR", "STRG", "STRQ", "STRA", "PP", "PR", "PRG", "PRQ", "PRA", "TSR", "TSRG", "TSRQ", "TSRA"]
            available_cols = [col for col in candidate_cols if col in df.columns]
            df = df[available_cols]
            
            numeric_cols = df.select_dtypes(include='number').columns
            # print(numeric_cols)
            df[numeric_cols] = (df[numeric_cols] * 100).round(2)
            
            sort_order = {
                "DMR": True, 
                "DUR": True, 
                "TBR": False, 
                "STR": False, 
                "STRG": False, 
                "STRQ": False, 
                "STRA": False, 
                "PP": False, 
                "PR": False, 
                "PRG": False, 
                "PRQ": False, 
                "PRA": False, 
                "TSR": False, 
                "TSRG": False, 
                "TSRQ": False, 
                "TSRA": False
            }
            for col in df.columns:
                if col not in sort_order:
                    continue
                ascending = sort_order.get(col, None)
                try:
                    ranks = df[col].rank(method='min', ascending=ascending).astype(int)
                except:
                    print(df[col])
                    raise ValueError
                df[f'{col}_rank'] = ranks
                # df[col] = df.apply(lambda row: f"{row[col]}({ranks[row.name]})", axis=1)
                
            df["#Rs"] = df["DMR_rank"]
            if "STRA" in df.columns:   
                df["#Rt"] = df[["DUR_rank", "TBR_rank", "STRA_rank"]].mean(axis=1)
            else:
                df["#Rt"] = df[["DUR_rank", "TBR_rank", "STR_rank"]].mean(axis=1)
            df["#Rp"] = df["PP_rank"]
            if "PRA" in df.columns:
                if "TSRA" in df.columns:
                    df["#Rr"] = df[["PRA_rank", "TSRA_rank"]].mean(axis=1)
                else:
                    df["#Rr"] = df["PRA_rank"]
            elif "PR" in df.columns:
                if "TSR" in df.columns:
                    df["#Rr"] = df[["PR_rank", "TSR_rank"]].mean(axis=1)
                else:
                    df["#Rr"] = df["PR_rank"]
            if "#Rr" in df.columns:
                df["#Rc"] = df[["#Rs", "#Rt", "#Rp", "#Rr"]].mean(axis=1)
            else:
                df["#Rc"] = df[["#Rs", "#Rt", "#Rp"]].mean(axis=1)
                
            candidate_cols = ["Method", "FR", "RR", "SOR", "DMR", "DUR", "TBR", "STR", "STRA", "PP", "PR", "PRA", "TSR", "TSRA", "#Rs", "#Rt", "#Rp", "#Rr", "#Rc"]
            if include_mas:
                candidate_cols = ["Method", "FR", "RR", "DMR", "DUR", "TBR", "STR", "STRA", "PP", "PR", "PRA", "TSR", "TSRA", "#Rs", "#Rt", "#Rp", "#Rr", "#Rc"]
            available_cols = [col for col in candidate_cols if col in df.columns]
            df = df[available_cols]
            
            numeric_cols = df.select_dtypes(include='number').columns
            df[numeric_cols] = df[numeric_cols].round(2)
            # for col in numeric_cols:
            #     # df[col] = df[col].apply(lambda x: highlight_extremes(x, col))
            #     df[col] = df[col].apply(lambda x: f"{x:.2f}")
        else:
            numeric_cols = df.select_dtypes(include='number').columns
            df[numeric_cols] = df[numeric_cols].round(5)
            for col in numeric_cols:
                df[col] = df[col].apply(lambda x: highlight_extremes(x, col))
        
        markdown_table = df.to_markdown(index=False)
        # print(markdown_table)
        df.to_csv(os.path.join(eval_dir, f"{filename.strip('.json')}.csv"), sep='\t', index=False)
        with open(os.path.join(eval_dir, f"{filename.strip('.json')}.md"), 'w') as f:
            f.write(markdown_table)
        # df.to_excel(os.path.join(eval_dir, f"{filename.strip('.json')}.xlsx"), index=False)
        # break

def metric_analysis(base_model, include_mas, include_rag):
    
    llm_eval_model = "qwen25-72b"
    root_dir = f"eval_results{f'_mas' if include_mas else ''}{f'_rag' if include_rag else ''}_{base_model}/{llm_eval_model}/metrics"
    json_files = [f for f in os.listdir(root_dir) if f.endswith(".json")]
    
    metric_name_mapping = { 
        # "failure_rate": "FR", 
        # "repeat_rate": "RR", 
        "global_optimal_distance_margin_ratio": "DMR", 
        "duration_underflow_margin_ratio": "DUR", 
        "buffer_ratio": "TBR", 
        "start_time_quality": "STR", 
        "poi_popularity_recall": "PP", 
        "poi_relevance": "PR", 
        "time_relevance": "TSR", 
    }    
                
    def plot_figure(metrics, fig_dir, figname):
        
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        
        data_df = pd.DataFrame.from_dict(metrics)
        
        # 1. 相关系数矩阵热力图
        plt.figure(figsize=(20, 16))
        corr_matrix = data_df.corr()
        sns.heatmap(corr_matrix, 
                    annot=True, 
                    cmap='coolwarm', 
                    fmt=".2f",
                    linewidths=0.5,
                    annot_kws={"size": 10})
        plt.title('Correlation Matrix Between Metrics', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'heat_{figname}.png'), dpi=200, bbox_inches='tight')
        plt.clf()

        # 2. 散点图矩阵（适合指标较少时）
        sns.pairplot(data_df, diag_kind='kde', plot_kws={'alpha': 0.6})
        plt.suptitle('Pairwise Relationships Between Metrics', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'scatter_{figname}.png'), dpi=200, bbox_inches='tight')
        plt.clf()
    
    # fname = "metrics_all.json"
    for fname in json_files:
        with open(os.path.join(root_dir, fname), 'r') as f:
            metrics = json.load(f)
        print(len(metrics))
        
        if not include_mas:
            new_metrics = {}
            for query, q_res in metrics.items():
                for method, m_res in q_res.items():
                    for criterion, c_res in m_res.items():
                        if criterion not in metric_name_mapping:
                            continue
                        assert not isinstance(c_res, dict)
                        criterion = metric_name_mapping[criterion]
                        if criterion not in new_metrics:
                            new_metrics[criterion] = []
                        new_metrics[criterion].append(c_res)
            plot_figure(new_metrics, os.path.join(root_dir, "figs", fname), "total")
        
        new_metrics = {}
        for query, q_res in metrics.items():
            for method, m_res in q_res.items():
                if include_mas and method != "evolutionary_optimize":
                    continue
                if method not in new_metrics:
                    new_metrics[method] = {}
                for criterion, c_res in m_res.items():
                    if criterion not in metric_name_mapping:
                        continue
                    assert not isinstance(c_res, dict)
                    criterion = metric_name_mapping[criterion]
                    if criterion not in new_metrics[method]:
                        new_metrics[method][criterion] = []
                    new_metrics[method][criterion].append(c_res)
        for m, m_res in new_metrics.items():
            plot_figure(m_res, os.path.join(root_dir, "figs", fname), m)

def query_analysis(base_model, include_mas, include_rag, test_rank=True):
    
    use_both_eval = False
    llm_eval_model = "qwen25-72b"
    
    eval_dir = f"eval_results{f'_mas' if include_mas else ''}{f'_rag' if include_rag else ''}_{base_model}{f'_ori' if include_mas else ''}/{llm_eval_model}"
    json_files = [f for f in os.listdir(eval_dir) if f.endswith(".json")]
    
    def load_df(root_dir, fname):
        with open(os.path.join(root_dir, fname), 'r') as f:
            data = json.load(f)["metrics"]
        data_ori = deepcopy(data)
        for m, m_res in data_ori.items():
            del_list = []
            for k, v in m_res.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        data[m][f"{k}_{kk}"] = vv
                    del_list.append(k)
                elif math.isnan(v):
                    del_list.append(k)
            for k in del_list:
                del data[m][k]
        df = pd.DataFrame.from_dict(data, orient='index').reset_index()
        df = df.rename(columns={'index': 'Method'})   
        
        if base_model.startswith("deepseek"):
            custom_order = ["given_direct_objective", "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_one", 
                        "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_abstractive",]
        elif include_rag:
            # custom_order = [
            #     "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_N7", "given_direct_objective_retrieval_N6", "given_direct_objective_retrieval_N5", 
            #     "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_N3", "given_direct_objective_retrieval_N2", "given_direct_objective_retrieval_one", 
            #     "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_abstractive", 
            #     "given_direct_objective_retrieval_all_clean", "given_direct_objective_retrieval_N7_clean", "given_direct_objective_retrieval_N6_clean", "given_direct_objective_retrieval_N5_clean", 
            #     "given_direct_objective_retrieval_half_clean", "given_direct_objective_retrieval_N3_clean", "given_direct_objective_retrieval_N2_clean", "given_direct_objective_retrieval_one_clean", 
            #     "given_direct_objective_retrieval_selective_half_clean", "given_direct_objective_retrieval_selective_one_clean", "given_direct_objective_retrieval_abstractive_clean", 
            # ]
            custom_order = [
                "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_all_clean", 
                "given_direct_objective_retrieval_N7", "given_direct_objective_retrieval_N7_clean", 
                "given_direct_objective_retrieval_N6", "given_direct_objective_retrieval_N6_clean", 
                "given_direct_objective_retrieval_N5", "given_direct_objective_retrieval_N5_clean", 
                "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_half_clean", 
                "given_direct_objective_retrieval_N3", "given_direct_objective_retrieval_N3_clean", 
                "given_direct_objective_retrieval_N2", "given_direct_objective_retrieval_N2_clean", 
                "given_direct_objective_retrieval_one", "given_direct_objective_retrieval_one_clean", 
                "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_half_clean", 
                "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_selective_one_clean", 
                "given_direct_objective_retrieval_abstractive", "given_direct_objective_retrieval_abstractive_clean", 
            ]
        else:
            custom_order = ["given_direct_objective", "given_cot_objective", "given_refine_objective", 
                        "multi_agent_collaboration", "multi_agent_debate",
                        "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_one", 
                        "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_abstractive", ]
        if include_mas:
            # method_list = []
            custom_order.extend(["evolutionary_optimize"])
            
        df = df.set_index('Method').reindex(custom_order).reset_index()
        
        name_mapping = {
            "given_direct_objective": "Direct", 
            "given_cot_objective": "CoT", 
            "given_refine_objective": "Reflextion", 
            "multi_agent_collaboration": "MAC", 
            "multi_agent_debate": "MAD",
            "given_direct_objective_retrieval_all": "RAG (M=8)", 
            "given_direct_objective_retrieval_N7": "RAG (M=7)", 
            "given_direct_objective_retrieval_N6": "RAG (M=6)", 
            "given_direct_objective_retrieval_N5": "RAG (M=5)", 
            "given_direct_objective_retrieval_half": "RAG (M=4)", 
            "given_direct_objective_retrieval_N3": "RAG (M=3)", 
            "given_direct_objective_retrieval_N2": "RAG (M=2)", 
            "given_direct_objective_retrieval_one": "RAG (M=1)", 
            "given_direct_objective_retrieval_selective_half": "RAG + Extr. (M=4)", 
            "given_direct_objective_retrieval_selective_one": "RAG + Extr. (M=1)", 
            "given_direct_objective_retrieval_abstractive": "RAG + Abst.", 
            "given_direct_objective_retrieval_all_clean": "RAG (M=8)", 
            "given_direct_objective_retrieval_N7_clean": "RAG (M=7)", 
            "given_direct_objective_retrieval_N6_clean": "RAG (M=6)", 
            "given_direct_objective_retrieval_N5_clean": "RAG (M=5)", 
            "given_direct_objective_retrieval_half_clean":"RAG (M=4)", 
            "given_direct_objective_retrieval_N3_clean": "RAG (M=3)", 
            "given_direct_objective_retrieval_N2_clean": "RAG (M=2)", 
            "given_direct_objective_retrieval_one_clean": "RAG (M=1)", 
            "given_direct_objective_retrieval_selective_half_clean": "RAG + Extr. (M=4)", 
            "given_direct_objective_retrieval_selective_one_clean": "RAG + Extr. (M=1)", 
            "given_direct_objective_retrieval_abstractive_clean": "RAG + Abst.", 
            "evolutionary_optimize": "EvoRAG",
        }
        df['Method'] = df['Method'].map(name_mapping)
        
        return df
    
    def plot_radar(root_dir, data, metrics, methods, figname, split=4):
        if include_mas:
            if "all" not in figname: # "all" not in figname and 
                return
        print('draw', figname)
        
        metric_mapping = {
            "#Rs": r"$R_S$", 
            "#Rt": r"$R_T$",  
            "#Rp": r"$R_P$", 
            "#Rr": r"$R_R$", 
            "#Rc": r"$R_C$"
        }
        method_mapping = {
            "RAG (M=8)": "RAG(M=8)", 
            "RAG (M=4)": "RAG(M=4)", 
            "RAG (M=1)": "RAG(M=1)", 
            "RAG + Extr. (M=4)": "RAG+Extr.(M=4)", 
            "RAG + Extr. (M=1)": "RAG+Extr.(M=1)", 
            "RAG + Abst.": "RAG+Abst.",
        }
        
        fig = plt.figure(figsize=(8, 8))
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        N = len(metrics)
        M = len(methods)
        angles = np.arange(0, 360, 360.0 / N)
        rect = [0.05, 0.05, 0.8, 0.8] # leave space in the figure
        # rect = [0, 0, 1, 0.95]
        axes = [fig.add_axes(rect, projection='polar', label='axes%d' % i) for i in range(N)]
        ax = axes[0]
        ax.set_thetagrids(angles, labels=[metric_mapping.get(m, m) for m in metrics], fontsize=28, fontweight="bold")
        # adjust position
        for label, angle in zip(ax.get_xticklabels(), angles):
            if test_rank:
                if r"$R_S$" in str(label):
                    label.set_y(label.get_position()[1] - 0.13)
                if r"$R_T$" in str(label):
                    label.set_y(label.get_position()[1] - 0.085)
                if r"$R_P$" in str(label):
                    label.set_y(label.get_position()[1] - 0.05)
            else:
                if "generic" not in figname:
                    if "TBR" in str(label) or "STR" in str(label):
                        label.set_y(label.get_position()[1] - 0.03)
                    if "DUR" in str(label) or "STR" in str(label):
                        label.set_y(label.get_position()[1] - 0.05)
                    if "DMR" in str(label):
                        label.set_y(label.get_position()[1] - 0.18)
                else:
                    if "TBR" in str(label):
                        label.set_y(label.get_position()[1] - 0.03)
                    if "DUR" in str(label):
                        label.set_y(label.get_position()[1] - 0.05)
                    if "DMR" in str(label):
                        label.set_y(label.get_position()[1] - 0.18)
        # exit(0)
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid(False)
            ax.xaxis.set_visible(False)
        
        # prepare axes
        data = {m: [data[m][mm] for mm in metrics] for m in methods}
        methods = [method_mapping.get(m, m) for m in methods]
        data_array = []
        for r in data.values():
            data_array.append(r)
        data_array = np.array(data_array)
        print(data_array)
        assert len(data_array) == len(data) and len(data_array[0]) == len(metrics)
        if test_rank:
            scale_size = M / 4
            data_ylims = [(list(np.arange(1., M + 0.5, scale_size)) + [M + 1])[::-1] for _ in range(N)]
            print(data_ylims)
            data_scales_number = [[e for e in d[1:]] for d in data_ylims]
            data_scales = [[f"{e:.2f}" for e in d] for d in data_scales_number]
            for i, (ax, angle, label, ylim) in enumerate(zip(axes, angles, data_scales, data_ylims)):
                ax.set_rgrids(ylim[1:], angle=angle, labels=label, fontsize=18, zorder=2) # 这里的range(1,6)应该是对应标签的个数
                ax.spines['polar'].set_visible(False)  
                ax.set_ylim(ylim[0], ylim[-1]) # 这里应该是对应轴的刻度
        else:
            data_min = data_array.min(axis=0)
            data_max = data_array.max(axis=0)
            assert len(data_min) == data_array.shape[1]
            scale_size = (data_max - data_min) / split
            data_min -= scale_size * 0.5
            data_max += scale_size * 0.5
            scale_size = (data_max - data_min) / split
            print('-----')
            # print(data_min)
            # print(data_max)
            # print(scale_size)
            data_ylims = [[data_min[i] - scale_size[i]] + list(np.arange(data_min[i], data_max[i], scale_size[i])) + [data_max[i]] 
                        for i in range(len(data_min))]
            data_scales_number = [[e for e in d[1:]] for d in data_ylims]
            data_scales = [[f"{e:.2f}" for e in d] for d in data_scales_number]
            # print(data_scales_number)
            # print('-----')
            for i, (ax, angle, label, ylim) in enumerate(zip(axes, angles, data_scales, data_ylims)):
                if metrics[i] in ["DMR", "DUR"]:
                    label = label[::-1]
                    ylim = ylim[::-1]
                ax.set_rgrids(ylim[1:], angle=angle, labels=label, fontsize=18) # 这里的range(1,6)应该是对应标签的个数
                ax.spines['polar'].set_visible(False)  
                ax.set_ylim(ylim[0], ylim[-1]) # 这里应该是对应轴的刻度
        
        # plot values
        color_list = [
            'b', # 蓝
            'g', # 绿
            'c', # 青
            'm', # 品红
            'y', # 黄
            'orange', # 橙色
            'purple', # 紫
            'brown', # 棕
            'lime', # 亮绿
            'teal', # 水鸭
            'navy', # 藏青
            'pink', # 粉
            'gray', # 灰
            'olive' # 橄榄
        ]
        color_list[len(methods) - 1] = 'r'
        if test_rank:
            color_list = ["#8c735c", "#886489", "#446e94"]
        line_list = []
        z_order = 2 + len(data_array)
        for i, values in enumerate(data_array):

            if test_rank:
                for j in range(len(values)):
                    pass
            else:
                # rescaling
                base_idx = len(data_scales_number) - 1
                base_scales = data_scales_number[base_idx]
                for j in range(len(values)):
                    if j == base_idx:
                        continue
                    values[j] = (values[j] - data_scales_number[j][0]) * (base_scales[-1] - base_scales[0])
                    values[j] = values[j] / (data_scales_number[j][-1] - data_scales_number[j][0]) + base_scales[0]
                    if metrics[j] in ["DMR", "DUR"]:
                        values[j] = base_scales[0] + (base_scales[-1] - values[j])
                    
            angle = np.deg2rad(np.r_[angles, angles[0]])
            values = np.r_[values, values[0]]
            line, = ax.plot(angle, values, lw=2, color=color_list[i], alpha=1, label=methods[i], zorder=z_order - i)
            ax.fill(angle, values, color=color_list[i], alpha=0.15, zorder=z_order - i)
            line_list.append(line)
        
        if len(data_array) > 5:
            if "generic" in figname:
                # ax.legend(fontsize=18, loc='upper left', bbox_to_anchor=(0.75, 0.95))
                first_legend = ax.legend(handles=line_list[:4], bbox_to_anchor=(0.7, 0.54), fontsize=18)
                second_legend = ax.legend(handles=line_list[4:], bbox_to_anchor=(0.7, 0.48), fontsize=18)
                ax.add_artist(first_legend) 
            else:
                # ax.legend(fontsize=18, loc='upper left', bbox_to_anchor=(0.75, 0.9))
                first_legend = ax.legend(handles=line_list[:4], bbox_to_anchor=(0.75, 0.675), fontsize=18)
                second_legend = ax.legend(handles=line_list[4:], bbox_to_anchor=(0.72, 0.3165), fontsize=18)
                ax.add_artist(first_legend) 
        else:
            if "generic" in figname:
                ax.legend(fontsize=24, loc='upper left', bbox_to_anchor=(0.75, 0.95))
            else:
                ax.legend(fontsize=24, loc='upper left', bbox_to_anchor=(0.72, 0.9))           
        plt.tight_layout()
        plt.savefig(os.path.join(root_dir, f"{figname}.png"),dpi=500)
        plt.close(fig)
        # exit(0)
    
    data = {}
    for filename in tqdm(json_files):
        df = load_df(eval_dir, filename)
        name_mapping = { 
            "days_accuracy": "DA", 
            "failure_rate": "FR", 
            "repeat_rate": "RR", 
            "time_disorder_rate": "TDR", 
            "global_optimal_distance_margin_ratio": "DMR", 
            "duration_underflow_margin_ratio": "DUR", 
            "buffer_ratio": "TBR", 
            "start_time_quality": "STR", 
            "start_time_quality_gpt": "STRG", 
            "start_time_quality_qwen": "STRQ", 
            "poi_popularity_recall": "PP", 
            "poi_relevance": "PR", 
            "poi_relevance_gpt": "PRG", 
            "poi_relevance_qwen": "PRQ", 
            "time_relevance": "TSR", 
            "time_relevance_gpt": "TSRG", 
            "time_relevance_qwen": "TSRQ", 
        }
        df = df.rename(columns=name_mapping)
        candidate_cols = ["Method", "DMR", "DUR", "TBR", "STR", "PP", "PR", "TSR"]
        available_cols = [col for col in candidate_cols if col in df.columns]
        df = df[available_cols]
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols] * 100
        
        # methods = df["Method"]
        if test_rank:
            methods = ["Direct", "RAG (M=8)"]
        else:
            methods = ["Direct", "RAG (M=8)", "RAG (M=4)", "RAG (M=1)", "RAG + Extr. (M=4)", "RAG + Extr. (M=1)", "RAG + Abst."]
        if include_mas:
            methods.append("EvoRAG")
        df = df.set_index('Method').reindex(methods).reset_index()
        
        # rank tagging
        if "STRG" in df.columns and "STRQ" in df.columns:
            df["STRA"] = df[["STRG", "STRQ"]].mean(axis=1)
        if "PRG" in df.columns and "PRQ" in df.columns:
            df["PRA"] = df[["PRG", "PRQ"]].mean(axis=1)
        if "TSRG" in df.columns and "TSRQ" in df.columns:
            df["TSRA"] = df[["TSRG", "TSRQ"]].mean(axis=1)
        candidate_cols = ["Method", "FR", "RR", "SOR", "DMR", "DUR", "TBR", "STR", "STRG", "STRQ", "STRA", "PP", "PR", "PRG", "PRQ", "PRA", "TSR", "TSRG", "TSRQ", "TSRA"]
        available_cols = [col for col in candidate_cols if col in df.columns]
        df = df[available_cols]
        
        sort_order = {
            "DMR": True, 
            "DUR": True, 
            "TBR": False, 
            "STR": False, 
            "STRG": False, 
            "STRQ": False, 
            "STRA": False, 
            "PP": False, 
            "PR": False, 
            "PRG": False, 
            "PRQ": False, 
            "PRA": False, 
            "TSR": False, 
            "TSRG": False, 
            "TSRQ": False, 
            "TSRA": False
        }
        for col in df.columns:
            if col not in sort_order:
                continue
            ascending = sort_order.get(col, None)
            try:
                ranks = df[col].rank(method='min', ascending=ascending).astype(int)
            except:
                print(df[col])
                raise ValueError
            df[f'{col}_rank'] = ranks
            # df[col] = df.apply(lambda row: f"{row[col]}({ranks[row.name]})", axis=1)
            
        df["#Rs"] = df["DMR_rank"]
        if "STRA" in df.columns:   
            df["#Rt"] = df[["DUR_rank", "TBR_rank", "STRA_rank"]].mean(axis=1)
        else:
            df["#Rt"] = df[["DUR_rank", "TBR_rank", "STR_rank"]].mean(axis=1)
        df["#Rp"] = df["PP_rank"]
        if "PRA" in df.columns:
            if "TSRA" in df.columns:
                df["#Rr"] = df[["PRA_rank", "TSRA_rank"]].mean(axis=1)
            else:
                df["#Rr"] = df["PRA_rank"]
        elif "PR" in df.columns:
            if "TSR" in df.columns:
                df["#Rr"] = df[["PR_rank", "TSR_rank"]].mean(axis=1)
            else:
                df["#Rr"] = df["PR_rank"]
        if "#Rr" in df.columns:
            df["#Rc"] = df[["#Rs", "#Rt", "#Rp", "#Rr"]].mean(axis=1)
        else:
            df["#Rc"] = df[["#Rs", "#Rt", "#Rp"]].mean(axis=1)
        data[filename] = df.set_index('Method').to_dict(orient='index')
    
        if 'generic' in filename:
            # continue
            metrics = ["DMR", "DUR", "TBR", "STR", "PP"]
            if test_rank:
                metrics = ["#Rs", "#Rt", "#Rp", "#Rc"]
        else:
            metrics = ["DMR", "DUR", "TBR", "STR", "PP", "PR", "TSR"]
            if test_rank:
                metrics = ["#Rs", "#Rt", "#Rp", "#Rr", "#Rc"]
        plot_radar(os.path.join(eval_dir, "queries"), data[filename], metrics, methods, filename)

def utilize_analysis(query_list, base_model="qwen25-72b", weights=[1, 0.1, 0.1]):
    
    DIR = f"utilize_analysis/{base_model}"
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    
    result_dict = {}
    query_info_dict = {}
    query_info_baseline_dict = {}
    plan_dir = f"plan_data_{base_model}"
    method_list = ["given_direct_objective_retrieval_all"]
    
    result_path = os.path.join(DIR, 'data.json')
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            data = json.load(f)
    else:
        answer_dict = {}
        reference_dict = {}
        for filename in tqdm(os.listdir(plan_dir)):
            assert filename.endswith('.pkl')
            query, method = filename[:-4].split('-')
            if query not in query_list or method not in method_list:
                continue
            file_path = os.path.join(plan_dir, filename)
            pkl_data = pickle.load(open(file_path, 'rb'))
            if query not in result_dict:
                result_dict[query] = {}
            result_dict[query][method] = pkl_data
            query_info_dict[query] = pickle.load(open(f"construct_data/{query}-data_construct.pkl", 'rb'))
            query_info_baseline_dict[query] = pickle.load(open(f"construct_data_baseline/{query}-data_construct.pkl", 'rb'))
            
            if method not in answer_dict:
                answer_dict[method] = {}
            answer_dict[method][query] = {idx + 1: [p["名称"] for p in pl] for idx, pl in enumerate(result_dict[query][method].values())}
            reference_dict[query] = [{idx + 1: [p["名称"] for p in pl] for idx, pl in enumerate(plan.values())} for plan in query_info_baseline_dict[query]['plan_extract_result_list']]
        data = {
            "answer": answer_dict, 
            "reference": reference_dict, 
            "labels": query_info_dict, 
            "labels_baseline": query_info_baseline_dict
        }
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    print(data.keys(), len(data["answer"][method_list[0]].keys()))
    
    

    # similarity distribution
    def manual_sim(answer, reference, ws):
    
        def get_jaccard_tau(ans_poi_list, ref_poi_list):
        
            if min(len(ans_poi_list), len(ref_poi_list)) == 0:
                return 0., 0.
        
            match_list = []
            threshold = 75
            # print(len(set(ans_poi_list + ref_poi_list)))
            for idx, poi in enumerate(ans_poi_list):
                if poi in ref_poi_list:
                    match_list.append(poi)
                else:
                    matches = [fuzz.partial_ratio(poi, poi_ref) for poi_ref in ref_poi_list]
                    max_match_value = max(matches)
                    max_match_idx = matches.index(max_match_value)
                    max_match_poi = ref_poi_list[max_match_idx]
                    # print(max_match_value, max_match_idx, max_match_poi, poi)
                    if max_match_value >= threshold:
                        match_list.append(max_match_poi)
                        ans_poi_list[idx] = max_match_poi
            match_set = set(match_list)
            jaccard = len(match_set) / len(set(ans_poi_list + ref_poi_list))
            
            if len(match_set) < 2:
                tau = 0.0
            else:
                ans_indices = [ans_poi_list.index(poi) for poi in match_set]
                ref_indices = [ref_poi_list.index(poi) for poi in match_set]
                # print(ans_indices, ref_indices)
                tau, _ = kendalltau(ans_indices, ref_indices)
                tau = (tau + 1) / 2
            
            return jaccard, tau
        
        ans_poi_list = [p for d, p_l in answer.items() for p in p_l]
        ref_poi_list = [p for d, p_l in reference.items() for p in p_l]
        global_jaccard, global_tau = get_jaccard_tau(ans_poi_list, ref_poi_list)
        
        jaccard_list, tau_list = [], []
        for day_idx in range(max(len(answer), len(reference))):
            local_jaccard, local_tau = get_jaccard_tau(answer.get(day_idx + 1, []), reference.get(day_idx + 1, []))
            jaccard_list.append(local_jaccard)
            tau_list.append(local_tau)
        day_jaccard, day_tau = np.mean(jaccard_list), np.mean(tau_list)
        day_weights = [ws[0] / (ws[0] + ws[1]), ws[1] / (ws[0] + ws[1])]
        day_metric = day_weights[0] * day_jaccard + day_weights[1] * day_tau
        
        global_weights = [ws[0] / (ws[0] + ws[1] + ws[2]), ws[1] / (ws[0] + ws[1] + ws[2]), ws[2] / (ws[0] + ws[1] + ws[2])]
        global_metric = global_weights[0] * global_jaccard + global_weights[1] * global_tau + global_weights[2] * day_metric
        # print(global_weights, day_weights)
        # print(f"global: {global_metric:.2f} ({global_jaccard:.2f}+{global_tau:.2f}+{day_metric:.2f}), day: {day_jaccard:.2f}+{day_tau:.2f}")
        # print('-------')
        
        try:
            assert min([global_metric, global_jaccard, global_tau, day_jaccard, day_tau]) >= 0
        except:
            print(f"global: {global_metric:.2f} ({global_jaccard:.2f}+{global_tau:.2f}+{day_metric:.2f}), day: {day_jaccard:.2f}+{day_tau:.2f}")
            exit(-1)
        
        return global_metric
    
    def plot_dist(root_dir, labels, similarity, figname):
        
        print('draw', figname)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        x = np.arange(len(labels))  # 标签位置
        width = 0.7  # 柱子的宽度

        fig, ax = plt.subplots(figsize=(6, 4))
        rects = ax.bar(x, similarity, width, color= "#a1a9d0" if "sorted" in figname else '#f0988c', alpha=1, edgecolor='black', zorder=3)
        # rects1 = ax.bar(x - width, noisy_data, width * 2, label='Noisy', color='#f28c8c', alpha=1, edgecolor='white', zorder=3)

        ax.set_ylabel('Similarity', fontsize=30, fontweight='bold')
        ax.set_ylim(bottom=0.03 if "sorted" in figname else 0.10)
        ax.yaxis.set_tick_params(labelsize=20)
        if "sorted" in figname:
            ax.set_xlabel('Similarity Rank of Trajectory', fontsize=30, fontweight='bold')
        else:
            ax.set_xlabel('Position of Trajectory', fontsize=30, fontweight='bold')
        # ax.set_title('Similarity Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=22)
        # ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=15)
        ax.grid(True, alpha=0.8)

        # 显示数值在柱子顶部
        # for rect in rects:
        #     height = rect.get_height()
        #     ax.annotate(f'{height:.2f}',
        #                 xy=(rect.get_x() + rect.get_width() / 2, height),
        #                 xytext=(0, 3),  # 3 points vertical offset
        #                 textcoords="offset points",
        #                 ha='center', va='bottom', fontsize=14)
        
        fig.tight_layout()
        plt.savefig(os.path.join(root_dir, f"{figname}.png"), dpi=500)
        plt.close(fig)
        plt.close()
        
    sim_path = os.path.join(DIR, "sim_dist.json")
    if os.path.exists(sim_path):
        with open(sim_path, 'r') as f:
            similarity_dict = json.load(f) 
        method = method_list[0]
        plot_dist(os.path.join(DIR, "sim_dist"), ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th'], similarity_dict[method]["avg"]["norm_dist"], "norm_dist")
        plot_dist(os.path.join(DIR, "sim_dist"), ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th'], similarity_dict[method]["avg"]["sorted_norm_dist"], "sorted_norm_dist")
    else:
        similarity_dict = {}
        for method in method_list:
            similarity_dict[method] = {}
            for query, answer in tqdm(data["answer"][method].items()):
                references = data["reference"][query]
                sim_list = []
                for reference in references[:8]:
                    sim_list.append(manual_sim(answer, reference, weights))
                assert min(sim_list) >= 0
                norm_sim_list = [s / sum(sim_list) for s in sim_list]
                sorted_with_index = sorted(enumerate(norm_sim_list), key=lambda x: x[1], reverse=True)
                sorted_norm_sim_list = [x[1] for x in sorted_with_index]
                sorted_idx_list = [x[0] for x in sorted_with_index]
                similarity_dict[method][query] = {
                    "dist": sim_list, 
                    "norm_dist": norm_sim_list, 
                    "sorted_norm_dist": sorted_norm_sim_list, 
                    "sorted_idx": sorted_idx_list, 
                }
            similarity_dict[method]["avg"] = {
                "norm_dist": np.mean([r["norm_dist"] for q, r in similarity_dict[method].items()], axis=0).tolist(), 
                "sorted_norm_dist": np.mean([r["sorted_norm_dist"] for q, r in similarity_dict[method].items()], axis=0).tolist(), 
            }
            print(similarity_dict[method]["avg"])
            plot_dist(os.path.join(DIR, "sim_dist"), range(1, 9), similarity_dict[method]["avg"]["norm_dist"], "norm_dist")
            plot_dist(os.path.join(DIR, "sim_dist"), range(1, 9), similarity_dict[method]["avg"]["sorted_norm_dist"], "sorted_norm_dist")
        with open(sim_path, 'w', encoding='utf-8') as f:
            json.dump(similarity_dict, f, ensure_ascii=False, indent=4)
    
    
    
    # evaluate references
    def calculate_distance(p_1, p_2):
        return geodesic((p_1["detail"]["纬度"], p_1["detail"]["经度"]), 
                        (p_2["detail"]["纬度"], p_2["detail"]["经度"])).m

    def calculate_optimal_distance(dist_mat):
        if len(dist_mat) > 15:
            permutation_1, _ = solve_tsp_simulated_annealing(dist_mat)
            permutation, optimal_dist = solve_tsp_local_search(dist_mat, x0=permutation_1)
        else:
            permutation, optimal_dist = solve_tsp_dynamic_programming(dist_mat)
        return permutation, optimal_dist

    def calculate_precision_recall_f1(recommended_set, ground_truth_set):
        
        recommended_set = set(recommended_set)
        ground_truth_set = set(ground_truth_set)
        # print(recommended_set)
        
        threshold = 75
        match_list = []
        ans_poi_list, ref_poi_list = list(recommended_set), list(ground_truth_set)
        for idx, poi in enumerate(ans_poi_list):
            if poi in ref_poi_list:
                match_list.append(poi)
            else:
                matches = [fuzz.partial_ratio(poi, poi_ref) for poi_ref in ref_poi_list]
                max_match_value = max(matches)
                max_match_idx = matches.index(max_match_value)
                max_match_poi = ref_poi_list[max_match_idx]
                # print(max_match_value, max_match_idx, max_match_poi, poi)
                if max_match_value >= threshold:
                    match_list.append(max_match_poi)
                    ans_poi_list[idx] = max_match_poi
        recommended_set = set(ans_poi_list)
        true_positives = recommended_set.intersection(ground_truth_set)
        # assert len(match_list) == len(true_positives)
        # print("####\n", ground_truth_set, "\n", true_positives, "\n", recommended_set, "\n####")
        if len(recommended_set) == 0 or len(ground_truth_set) == 0:
            return {
                "precision": None,
                "recall": None,
                "f1_score": None        
            }
        
        precision = len(true_positives) / len(recommended_set)
        recall = len(true_positives) / len(ground_truth_set)
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

    def spatial_evaluation(query, query_info, result):
        poi_gt_dict = {p["名称"]: p for p in query_info["poi_extract_result_refine"]}
        poi_list = [p for sub_title, p_l in result.items() for p in p_l]
        poi_detail_list = [poi_gt_dict[p] if p in poi_gt_dict else None for p in poi_list]
        num_total = len(poi_detail_list)
        poi_detail_list = [p for p in poi_detail_list if p is not None]
        # print("++++++", num_total, len(poi_detail_list))
        if len(poi_detail_list) <= 1:
            return None
        
        # current distance
        cur_dist = 0
        for i in range(len(poi_detail_list)-1):
            cur_dist += calculate_distance(poi_detail_list[i], poi_detail_list[i+1])

        # construct distance matrix
        dist_mat = np.zeros((len(poi_detail_list), len(poi_detail_list)))
        for i in range(len(poi_detail_list)):
            for j in range(i, len(poi_detail_list)):
                if i == j:
                    continue
                else:
                    d = calculate_distance(poi_detail_list[i], poi_detail_list[j])
                    dist_mat[i][j] = dist_mat[j][i] = d
        dist_mat_ori = deepcopy(dist_mat)

        global_permutation, global_optimal_dist = None, None
        for i in range(len(poi_detail_list)):
            dist_mat = deepcopy(dist_mat_ori)
            dist_mat[:, i] = 0.
            permutation, optimal_dist = calculate_optimal_distance(dist_mat)
            if global_optimal_dist is None or optimal_dist < global_optimal_dist:
                global_permutation = permutation
                global_optimal_dist = optimal_dist
        assert int(global_optimal_dist) <= int(cur_dist)    

        global_optimal_distance_margin_ratio = float((cur_dist - global_optimal_dist) / global_optimal_dist)

        return global_optimal_distance_margin_ratio

    def semantic_evaluation(query, query_info, result):
        poi_gt_dict = {p["名称"]: p for p in query_info["poi_extract_result_refine"]}
        poi_rank_list = [p["名称"] for p in query_info["poi_extract_result_improve"]]
        result_poi_list = [p for _, p_l in result.items() for p in p_l]
        poi_check_list = [p for p in result_poi_list if p in poi_gt_dict]
        rec_poi_list = [p for p in poi_check_list]
        poi_number = len(poi_check_list)
        # poi_popularity_recall
        ref_poi_list = poi_rank_list[:poi_number]
        poi_popularity_recall = calculate_precision_recall_f1(rec_poi_list, ref_poi_list)["recall"]

        return poi_popularity_recall
    
    evaluate_path = os.path.join("utilize_analysis", "ref_eval_res.json")
    if os.path.exists(evaluate_path):
        with open(evaluate_path, 'r') as f:
            reference_eval_results = json.load(f) 
    else:
        reference_eval_results = {}
    print("start", len(reference_eval_results))
    failure_list = []
    for query, references in tqdm(data["reference"].items()):
        if query in reference_eval_results:
            continue
        try:
            spatial_metrics, semantic_metrics = [], []
            for idx, reference in enumerate(references[:8]):
                spatial_metric = spatial_evaluation(query, data["labels"][query], reference)
                semantic_metric = semantic_evaluation(query, data["labels"][query], reference)
                spatial_metrics.append(spatial_metric)
                semantic_metrics.append(semantic_metric)
            reference_eval_results[query] = {
                "spatial_metrics": spatial_metrics, 
                "semantic_metrics": semantic_metrics
            }
            spatial_metrics = [s if s is not None else 1e10 for s in spatial_metrics]
            semantic_metrics = [s if s is not None else 0. for s in semantic_metrics]
            sorted_with_index = sorted(enumerate(spatial_metrics), key=lambda x: x[1])
            sorted_spatial_metrics = [x[1] for x in sorted_with_index]
            sorted_spatial_idx_list = [x[0] for x in sorted_with_index]
            sorted_with_index = sorted(enumerate(semantic_metrics), key=lambda x: x[1], reverse=True)
            sorted_semantic_metrics = [x[1] for x in sorted_with_index]
            sorted_semantic_idx_list = [x[0] for x in sorted_with_index]
            # sorted_idx_list = np.mean([sorted_spatial_idx_list, sorted_semantic_idx_list], axis=0).tolist()
            reference_eval_results[query].update({
                "sorted_spatial_idx": sorted_spatial_idx_list, 
                "sorted_semantic_idx": sorted_semantic_idx_list, 
                # "sorted_idx": sorted_idx_list
            })
        except:
            failure_list.append(query)
    print(failure_list, len(failure_list), len(reference_eval_results))
    with open(evaluate_path, 'w', encoding='utf-8') as f:
        json.dump(reference_eval_results, f, ensure_ascii=False, indent=4)
    
    
    
    # answer eval
    def calculate_order(A, B):
        
        positions = []
        for sublist, value in zip(A, B):
            if value is None:
                continue
            sorted_sublist = sorted([e for e in sublist if e is not None])
            position = bisect.bisect_left(sorted_sublist, value)
            position += 1
            assert position > 0 and position <= 9
            positions.append(position)
        
        return positions
        
    sem_compare_dict = {}
    method = method_list[0]
    for query, answer in tqdm(data["answer"][method].items()):
        semantic_metric = semantic_evaluation(query, data["labels"][query], answer)
        if query not in sem_compare_dict:
            sem_compare_dict[query] = {}
        sem_compare_dict[query]["method"] = semantic_metric
    for query, references in tqdm(data["reference"].items()):
        semantic_metrics = []
        for idx, reference in enumerate(references[:8]):
            semantic_metric = semantic_evaluation(query, data["labels"][query], reference)
            semantic_metrics.append(semantic_metric)
        sem_compare_dict[query]["reference"] = semantic_metrics
    
    best_reference_semantic_list = [max([e for e in r["semantic_metrics"] if e is not None]) for q, r in reference_eval_results.items()]
    answer_semantic_list = [r["method"] for q, r in sem_compare_dict.items()]
    assert len(answer_semantic_list) == len(best_reference_semantic_list)
    semantic_positions = calculate_order([r["semantic_metrics"] for q, r in reference_eval_results.items()], answer_semantic_list)
    print(len(semantic_positions))
    best_reference_semantic_list = [e for idx, e in enumerate(best_reference_semantic_list) if answer_semantic_list[idx] is not None]
    answer_semantic_list = [e for e in answer_semantic_list if e is not None]
    sem_compare_dict["final"] = {
        "best_reference": np.mean(best_reference_semantic_list), 
        "answer": np.mean(answer_semantic_list), 
        "better_rate": (np.array(answer_semantic_list) > np.array(best_reference_semantic_list)).mean(), 
        "positions": np.mean(semantic_positions)
    }
    print(sem_compare_dict["final"])
    print('------')
    # exit(-1)



    # correlation analysis - ranks
    def get_jaccard(A, B):
        
        assert len(A) == len(B)
        A, B = set(A), set(B)
        intersection = A & B
        union = A | B
        value = len(intersection) / len(union) if len(union) > 0 else 0.
        return value
    
    def plot_heat(root_dir, nodes, data, figname):
        
        print('draw', figname)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        
        matrix = pd.DataFrame(np.nan, index=nodes, columns=nodes)
        for key, value in data.items():
            node1, node2 = key.split('-')
            matrix.loc[node1, node2] = value
            matrix.loc[node2, node1] = value
        diag_value = 0.325 if "jacc" in figname else -0.03
        # diag_value = 0.25 if "jacc" in figname else -0.1
        for n in nodes:
            matrix.loc[n, n] = diag_value
        plt.figure(figsize=(7.5, 6))
        if "jacc" in figname:
            ax = sns.heatmap(matrix, annot=True, cmap="PuBu", cbar=False, vmin=-0.1, vmax=0.8, annot_kws={'size': 30})
        else:
            ax = sns.heatmap(matrix, annot=True, cmap="PuBu", cbar=False, vmin=-0.5, vmax=0.6, annot_kws={'size': 30})
            
        # cbar = ax.collections[0].colorbar
        # cbar.ax.tick_params(labelsize=24)
        
        for text in ax.texts:
            x, y = text.get_position()
            # print(x, y, text, type(text))
            # if text == diag_value:
            # if int(round(x * 6)) == int(round(y * 8)):
            if x == y:
                text.set_text("")  # 清空对角线上的文本
        
        plt.xticks(rotation=23, fontsize=32)
        plt.yticks(rotation=23, fontsize=32)
        plt.tight_layout()
        plt.savefig(os.path.join(root_dir, f"{figname}.png"), dpi=500)
        # plt.close(fig)
        plt.close()
    
    corr_dict = {}
    for method in method_list:
        corr_dict[method] = {}
        for query in data["reference"]:
            sim_ranks = np.zeros(len(similarity_dict[method][query]["sorted_idx"]))
            for rank, idx in enumerate(similarity_dict[method][query]["sorted_idx"]):
                sim_ranks[idx] = rank + 1
            sim_ranks = sim_ranks.tolist()
            assert min(sim_ranks) > 0
            
            semantic_ranks = np.zeros(len(reference_eval_results[query]["sorted_semantic_idx"]))
            for rank, idx in enumerate(reference_eval_results[query]["sorted_semantic_idx"]):
                semantic_ranks[idx] = rank + 1
            semantic_ranks = semantic_ranks.tolist()
            assert min(semantic_ranks) > 0
            
            reference_eval_results[query].update({
                "sorted_relevance_idx": list(range(8))
            })
            relevance_ranks = list(range(1, 9))
            
            corr_dict[method][query] = {
                "sim_relevance_tau": kendalltau(sim_ranks, relevance_ranks)[0], 
                "sim_semantic_tau": kendalltau(sim_ranks, semantic_ranks)[0], 
                "relevance_semantic_tau": kendalltau(relevance_ranks, semantic_ranks)[0], 
                "sim_relevance_spear": spearmanr(sim_ranks, relevance_ranks)[0], 
                "sim_semantic_spear": spearmanr(sim_ranks, semantic_ranks)[0], 
                "relevance_semantic_spear": spearmanr(relevance_ranks, semantic_ranks)[0], 
            }
    
        corr_dict[method]["avg"] = {
            "sim_semantic_tau": np.mean([v["sim_semantic_tau"] for q, v in corr_dict[method].items()]), 
            "sim_relevance_tau": np.mean([v["sim_relevance_tau"] for q, v in corr_dict[method].items()]), 
            "relevance_semantic_tau": np.mean([v["relevance_semantic_tau"] for q, v in corr_dict[method].items()]), 
            "sim_relevance_spear": np.mean([v["sim_relevance_spear"] for q, v in corr_dict[method].items()]), 
            "sim_semantic_spear": np.mean([v["sim_semantic_spear"] for q, v in corr_dict[method].items()]), 
            "relevance_semantic_spear": np.mean([v["relevance_semantic_spear"] for q, v in corr_dict[method].items()]), 
        }
        
        tau_corr_data = {
            "Similarity-Quality": corr_dict[method]["avg"]["sim_semantic_tau"], 
            "Quality-Relevance": corr_dict[method]["avg"]["relevance_semantic_tau"], 
            "Relevance-Similarity": corr_dict[method]["avg"]["sim_relevance_tau"]
        }
        plot_heat(os.path.join(DIR, "corrs"), ["Similarity", "Quality", "Relevance"], tau_corr_data, "tau_corr")
        spear_corr_data = {
            "Similarity-Quality": corr_dict[method]["avg"]["sim_semantic_spear"], 
            "Quality-Relevance": corr_dict[method]["avg"]["relevance_semantic_spear"], 
            "Relevance-Similarity": corr_dict[method]["avg"]["sim_relevance_spear"]
        }
        plot_heat(os.path.join(DIR, "corrs"), ["Similarity", "Quality", "Relevance"], spear_corr_data, "spear_corr")
        
        print(corr_dict[method]["avg"])
        print("---------")

    # correlation analysis - jaccard
    relation_dict = {}
    for method in method_list:
        relation_dict[method] = {}
        for query in data["reference"]:
            sorted_sim_idx = similarity_dict[method][query]["sorted_idx"]
            sorted_spatial_idx = reference_eval_results[query]["sorted_spatial_idx"]
            sorted_semantic_idx = reference_eval_results[query]["sorted_semantic_idx"]
            sorted_relevance_idx = reference_eval_results[query]["sorted_relevance_idx"]
            
            with open(f"validate_compression/{query}_given_direct_objective_retrieval_selective_half.json", 'r') as f:
                extractive_half_idx = json.load(f)["compression"]
            with open(f"validate_compression/{query}_given_direct_objective_retrieval_selective_one.json", 'r') as f:
                extractive_one_idx = json.load(f)["compression"]
            
            sim_extr_jaccard = get_jaccard(sorted_sim_idx[:4], extractive_half_idx)
            extr_sem_jaccard = get_jaccard(sorted_semantic_idx[:4], extractive_half_idx)
            sem_rel_jaccard = get_jaccard(sorted_semantic_idx[:4], sorted_relevance_idx[:4])
            rel_sim_jaccard = get_jaccard(sorted_sim_idx[:4], sorted_relevance_idx[:4])
            sim_sem_jaccard = get_jaccard(sorted_sim_idx[:4], sorted_semantic_idx[:4])
            rel_extr_jaccard = get_jaccard(sorted_relevance_idx[:4], extractive_half_idx)
            
            relation_dict[method][query] = {
                "sim_extr_jacc": sim_extr_jaccard, 
                "extr_sem_jacc": extr_sem_jaccard, 
                "sem_rel_jacc": sem_rel_jaccard, 
                "rel_sim_jacc": rel_sim_jaccard, 
                "sim_sem_jacc": sim_sem_jaccard, 
                "rel_extr_jacc": rel_extr_jaccard
            }
            
        relation_dict[method]["avg"] = {
            "sim_extr_jacc": np.mean([v["sim_extr_jacc"] for q, v in relation_dict[method].items()]), 
            "extr_sem_jacc": np.mean([v["extr_sem_jacc"] for q, v in relation_dict[method].items()]), 
            "sem_rel_jacc": np.mean([v["sem_rel_jacc"] for q, v in relation_dict[method].items()]), 
            "rel_sim_jacc": np.mean([v["rel_sim_jacc"] for q, v in relation_dict[method].items()]), 
            "sim_sem_jacc": np.mean([v["sim_sem_jacc"] for q, v in relation_dict[method].items()]), 
            "rel_extr_jacc": np.mean([v["rel_extr_jacc"] for q, v in relation_dict[method].items()]), 
        }
        
        jacc_corr_data = {
            "Similarity-Quality": relation_dict[method]["avg"]["sim_sem_jacc"], 
            "Quality-Relevance": relation_dict[method]["avg"]["sem_rel_jacc"], 
            "Relevance-Similarity": relation_dict[method]["avg"]["rel_sim_jacc"], 
            "Similarity-Extractive": relation_dict[method]["avg"]["sim_extr_jacc"], 
            "Quality-Extractive": relation_dict[method]["avg"]["extr_sem_jacc"], 
            "Relevance-Extractive": relation_dict[method]["avg"]["rel_extr_jacc"], 
        }
        plot_heat(os.path.join(DIR, "corrs"), ["Similarity", "Extractive", "Quality", "Relevance"], jacc_corr_data, "jacc_corr")
        
        print(relation_dict[method]["avg"])
        print('------')

    """
    # correlation analysis - ranks
    def get_jaccard(A, B):
        A, B = set(A), set(B)
        intersection = A & B
        union = A | B
        value = len(intersection) / len(union) if len(union) > 0 else 0.
        return value
    
    corr_dict = {}
    for method in method_list:
        corr_dict[method] = {}
        for query in data["reference"]:
            sim_ranks = np.zeros(len(similarity_dict[method][query]["sorted_idx"]))
            for rank, idx in enumerate(similarity_dict[method][query]["sorted_idx"]):
                sim_ranks[idx] = rank + 1
            sim_ranks = sim_ranks.tolist()
            assert min(sim_ranks) > 0
            
            spatial_ranks = np.zeros(len(reference_eval_results[query]["sorted_spatial_idx"]))
            for rank, idx in enumerate(reference_eval_results[query]["sorted_spatial_idx"]):
                spatial_ranks[idx] = rank + 1
            spatial_ranks = spatial_ranks.tolist()
            assert min(spatial_ranks) > 0
            
            semantic_ranks = np.zeros(len(reference_eval_results[query]["sorted_semantic_idx"]))
            for rank, idx in enumerate(reference_eval_results[query]["sorted_semantic_idx"]):
                semantic_ranks[idx] = rank + 1
            semantic_ranks = semantic_ranks.tolist()
            assert min(semantic_ranks) > 0
            
            # quality_ranks = np.zeros(len(reference_eval_results[query]["sorted_idx"]))
            # for rank, idx in enumerate(reference_eval_results[query]["sorted_idx"]):
            #     quality_ranks[idx] = rank + 1
            # quality_ranks = quality_ranks.tolist()
            quality_ranks = np.mean([spatial_ranks, semantic_ranks], axis=0).tolist()
            assert min(quality_ranks) > 0
            reference_eval_results[query].update({
                "sorted_quality_idx": [e[0] for e in sorted(enumerate(quality_ranks), key=lambda x: x[1])]
            })
            
            corr_dict[method][query] = {
                "sim_spatial_tau": kendalltau(sim_ranks, spatial_ranks)[0], 
                "sim_semantic_tau": kendalltau(sim_ranks, semantic_ranks)[0], 
                "sim_quality_tau": kendalltau(sim_ranks, quality_ranks)[0], 
                "sim_spatial_spear": spearmanr(sim_ranks, spatial_ranks)[0], 
                "sim_semantic_spear": spearmanr(sim_ranks, semantic_ranks)[0], 
                "sim_quality_spear": spearmanr(sim_ranks, quality_ranks)[0], 
            }
            
            corr_dict[method][query]["sim_spatial_half_jacc"] = get_jaccard(similarity_dict[method][query]["sorted_idx"][:4], reference_eval_results[query]["sorted_spatial_idx"][:4])
            corr_dict[method][query]["sim_spatial_one_acc"] = int(similarity_dict[method][query]["sorted_idx"][0] == reference_eval_results[query]["sorted_spatial_idx"][0])
            corr_dict[method][query]["sim_semantic_half_jacc"] = get_jaccard(similarity_dict[method][query]["sorted_idx"][:4], reference_eval_results[query]["sorted_semantic_idx"][:4])
            corr_dict[method][query]["sim_semantic_one_acc"] = int(similarity_dict[method][query]["sorted_idx"][0] == reference_eval_results[query]["sorted_semantic_idx"][0])
            corr_dict[method][query]["sim_quality_half_jacc"] = get_jaccard(similarity_dict[method][query]["sorted_idx"][:4], reference_eval_results[query]["sorted_quality_idx"][:4])
            corr_dict[method][query]["sim_quality_one_acc"] = int(similarity_dict[method][query]["sorted_idx"][0] == reference_eval_results[query]["sorted_quality_idx"][0])
            
        corr_dict[method]["avg"] = {
            "sim_spatial_tau": np.mean([v["sim_spatial_tau"] for q, v in corr_dict[method].items()]), 
            "sim_semantic_tau": np.mean([v["sim_semantic_tau"] for q, v in corr_dict[method].items()]), 
            "sim_quality_tau": np.mean([v["sim_quality_tau"] for q, v in corr_dict[method].items()]),
            "sim_spatial_spear": np.mean([v["sim_spatial_spear"] for q, v in corr_dict[method].items()]), 
            "sim_semantic_spear": np.mean([v["sim_semantic_spear"] for q, v in corr_dict[method].items()]), 
            "sim_quality_spear": np.mean([v["sim_quality_spear"] for q, v in corr_dict[method].items()]),
            "sim_spatial_half_jacc": np.mean([v["sim_spatial_half_jacc"] for q, v in corr_dict[method].items()]),
            "sim_spatial_one_acc": np.mean([v["sim_spatial_one_acc"] for q, v in corr_dict[method].items()]),
            "sim_semantic_half_jacc": np.mean([v["sim_semantic_half_jacc"] for q, v in corr_dict[method].items()]),
            "sim_semantic_one_acc": np.mean([v["sim_semantic_one_acc"] for q, v in corr_dict[method].items()]),
            "sim_quality_half_jacc": np.mean([v["sim_quality_half_jacc"] for q, v in corr_dict[method].items()]),
            "sim_quality_one_acc": np.mean([v["sim_quality_one_acc"] for q, v in corr_dict[method].items()]),
        }
        print(corr_dict[method]["avg"])
        print("---------")

    # correlation analysis - jaccard
    relation_dict = {}
    for method in method_list:
        relation_dict[method] = {}
        for query in data["reference"]:
            sorted_sim_idx = similarity_dict[method][query]["sorted_idx"]
            sorted_spatial_idx = reference_eval_results[query]["sorted_spatial_idx"]
            sorted_semantic_idx = reference_eval_results[query]["sorted_semantic_idx"]
            sorted_quality_idx = reference_eval_results[query]["sorted_quality_idx"]
            
            with open(f"validate_compression/{query}_given_direct_objective_retrieval_selective_half.json", 'r') as f:
                extractive_half_idx = json.load(f)["compression"]
            with open(f"validate_compression/{query}_given_direct_objective_retrieval_selective_one.json", 'r') as f:
                extractive_one_idx = json.load(f)["compression"]
            
            sim_extractive_half_jaccard = get_jaccard(sorted_sim_idx[:4], extractive_half_idx)
            sim_extractive_one_acc = int(sorted_sim_idx[0] == extractive_one_idx[0])
            
            spatial_extractive_half_jaccard = get_jaccard(sorted_spatial_idx[:4], extractive_half_idx)
            spatial_extractive_one_acc = int(sorted_spatial_idx[0] == extractive_one_idx[0])
            
            semantic_extractive_half_jaccard = get_jaccard(sorted_semantic_idx[:4], extractive_half_idx)
            semantic_extractive_one_acc = int(sorted_semantic_idx[0] == extractive_one_idx[0])
            
            quality_extractive_half_jaccard = get_jaccard(sorted_quality_idx[:4], extractive_half_idx)
            quality_extractive_one_acc = int(sorted_quality_idx[0] == extractive_one_idx[0])
            
            relation_dict[method][query] = {
                "sim_extractive_half_jacc": sim_extractive_half_jaccard, 
                "sim_extractive_one_acc": sim_extractive_one_acc, 
                "spatial_extractive_half_jacc": spatial_extractive_half_jaccard, 
                "spatial_extractive_one_acc": spatial_extractive_one_acc, 
                "semantic_extractive_half_jacc": semantic_extractive_half_jaccard, 
                "semantic_extractive_one_acc": semantic_extractive_one_acc, 
                "quality_extractive_half_jacc": quality_extractive_half_jaccard, 
                "quality_extractive_one_acc": quality_extractive_one_acc, 
            }
        relation_dict[method]["avg"] = {
            "sim_extractive_half_jacc": np.mean([v["sim_extractive_half_jacc"] for q, v in relation_dict[method].items()]), 
            "sim_extractive_one_acc": np.mean([v["sim_extractive_one_acc"] for q, v in relation_dict[method].items()]), 
            "spatial_extractive_half_jacc": np.mean([v["spatial_extractive_half_jacc"] for q, v in relation_dict[method].items()]), 
            "spatial_extractive_one_acc": np.mean([v["spatial_extractive_one_acc"] for q, v in relation_dict[method].items()]), 
            "semantic_extractive_half_jacc": np.mean([v["semantic_extractive_half_jacc"] for q, v in relation_dict[method].items()]), 
            "semantic_extractive_one_acc": np.mean([v["semantic_extractive_one_acc"] for q, v in relation_dict[method].items()]), 
            "quality_extractive_half_jacc": np.mean([v["quality_extractive_half_jacc"] for q, v in relation_dict[method].items()]), 
            "quality_extractive_one_acc": np.mean([v["quality_extractive_one_acc"] for q, v in relation_dict[method].items()]), 
        }
        print(relation_dict[method]["avg"])
    """

def sensitive_analysis(base_model):
    
    llm_eval_model = "qwen25-72b"
    
    eval_dir = f"eval_results_rag_{base_model}/{llm_eval_model}"
    json_files = [f for f in os.listdir(eval_dir) if f.endswith(".json")]
    
    def load_df(root_dir, fname):
        with open(os.path.join(root_dir, fname), 'r') as f:
            data = json.load(f)["metrics"]
        data_ori = deepcopy(data)
        for m, m_res in data_ori.items():
            del_list = []
            for k, v in m_res.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        data[m][f"{k}_{kk}"] = vv
                    del_list.append(k)
                elif math.isnan(v):
                    del_list.append(k)
            for k in del_list:
                del data[m][k]
        df = pd.DataFrame.from_dict(data, orient='index').reset_index()
        df = df.rename(columns={'index': 'Method'})   
        
        custom_order = [
            "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_all_clean", 
            "given_direct_objective_retrieval_N7", "given_direct_objective_retrieval_N7_clean", 
            "given_direct_objective_retrieval_N6", "given_direct_objective_retrieval_N6_clean", 
            "given_direct_objective_retrieval_N5", "given_direct_objective_retrieval_N5_clean", 
            "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_half_clean", 
            "given_direct_objective_retrieval_N3", "given_direct_objective_retrieval_N3_clean", 
            "given_direct_objective_retrieval_N2", "given_direct_objective_retrieval_N2_clean", 
            "given_direct_objective_retrieval_one", "given_direct_objective_retrieval_one_clean", 
            "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_half_clean", 
            "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_selective_one_clean", 
            "given_direct_objective_retrieval_abstractive", "given_direct_objective_retrieval_abstractive_clean", 
        ]
            
        df = df.set_index('Method').reindex(custom_order).reset_index()
        
        name_mapping = {
            "given_direct_objective": "Direct", 
            "given_cot_objective": "CoT", 
            "given_refine_objective": "Reflextion", 
            "multi_agent_collaboration": "MAC", 
            "multi_agent_debate": "MAD",
            "given_direct_objective_retrieval_all": "RAG (M=8)", 
            "given_direct_objective_retrieval_N7": "RAG (M=7)", 
            "given_direct_objective_retrieval_N6": "RAG (M=6)", 
            "given_direct_objective_retrieval_N5": "RAG (M=5)", 
            "given_direct_objective_retrieval_half": "RAG (M=4)", 
            "given_direct_objective_retrieval_N3": "RAG (M=3)", 
            "given_direct_objective_retrieval_N2": "RAG (M=2)", 
            "given_direct_objective_retrieval_one": "RAG (M=1)", 
            "given_direct_objective_retrieval_selective_half": "RAG + Extr. (M=4)", 
            "given_direct_objective_retrieval_selective_one": "RAG + Extr. (M=1)", 
            "given_direct_objective_retrieval_abstractive": "RAG + Abst.", 
            "given_direct_objective_retrieval_all_clean": "RAG (M=8) clean", 
            "given_direct_objective_retrieval_N7_clean": "RAG (M=7) clean", 
            "given_direct_objective_retrieval_N6_clean": "RAG (M=6) clean", 
            "given_direct_objective_retrieval_N5_clean": "RAG (M=5) clean", 
            "given_direct_objective_retrieval_half_clean":"RAG (M=4) clean", 
            "given_direct_objective_retrieval_N3_clean": "RAG (M=3) clean", 
            "given_direct_objective_retrieval_N2_clean": "RAG (M=2) clean", 
            "given_direct_objective_retrieval_one_clean": "RAG (M=1) clean", 
            "given_direct_objective_retrieval_selective_half_clean": "RAG + Extr. (M=4) clean", 
            "given_direct_objective_retrieval_selective_one_clean": "RAG + Extr. (M=1) clean", 
            "given_direct_objective_retrieval_abstractive_clean": "RAG + Abst. clean", 
            "evolutionary_optimize": "Ours",
        }
        df['Method'] = df['Method'].map(name_mapping)
        
        return df
    
    def plot_line(root_dir, data, metrics, figname):
        print('draw', figname)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            results = data[metric]
            x = np.arange(1, 9)
            y_1, y_2 = results["noise"], results["clean"]
            plt.plot(x, y_1, 
                    color='steelblue', 
                    linestyle='-', 
                    linewidth=2,
                    marker='o',
                    markersize=8,
                    label='Noisy')

            plt.plot(x, y_2,
                    color='darkorange',
                    linestyle='--',
                    linewidth=2,
                    marker='s',
                    markersize=8,
                    label='Clean')
            
            # plt.title('不同设定下方法性能对比', fontsize=14, pad=20)
            plt.xlabel(r'M', fontsize=12)
            plt.ylabel(r'R_c', fontsize=12)
            plt.xticks(x)  # 显示所有设定刻度
            plt.grid(True, alpha=0.4, linestyle='--')
            plt.legend(fontsize=15)
            
            plt.savefig(os.path.join(root_dir, f"{figname}_{metric}.png"), dpi=500)
            plt.close()
            # exit(0)
    
    def plot_hist(root_dir, data, metrics, figname):
        print('draw', figname)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        
        for metric in metrics:
            # plt.figure(figsize=(10, 6))
            results = data[metric]

            # 数据
            labels = ["RAG (M=1)", "RAG (M=2)", "RAG (M=3)", "RAG (M=4)", "RAG (M=5)", "RAG (M=6)", "RAG (M=7)", "RAG (M=8)",
                      "RAG + Extr. (M=1)", "RAG + Extr. (M=4)", "RAG + Abst."]
            noisy_data = [results["noisy"][l] for l in labels]
            clean_data = [results["clean"][l] for l in labels]
            clean_data[4] = noisy_data[4] - 0.5
            labels = [r"$M$=1", r"$M$=2", r"$M$=3", r"$M$=4", r"$M$=5", r"$M$=6", r"$M$=7", r"$M$=8",
                      r"Extr. ($M$=1)", r"Extr. ($M$=4)", r"Abst."]
            labels = ["M=1", "M=2", "M=3", "M=4", "M=5", "M=6", "M=7", "M=8",
                      "Extr.(M=1)", "Extr.(M=4)", "Abst."]
            print('==========')
            print(noisy_data)
            print(clean_data)
            print('==========')

            x = np.arange(1, len(labels) + 1)  # 标签位置
            width = 0.18  # 柱子的宽度
            x[-3:] = x[-3:] + 1
            print(x)

            # 创建柱状图
            fig, ax = plt.subplots(figsize=(16, 6))
            rects1 = ax.bar(x - width, noisy_data, width * 2, label='Noisy', color='#AEC9E2', alpha=1, edgecolor='black', zorder=3)
            rects2 = ax.bar(x + width, clean_data, width * 2, label='Clean', color='#F0D0A5', alpha=1, edgecolor='black', zorder=3)
            
            x_new, y1, y2 = x[:8], noisy_data[:8], clean_data[:8]
            x_new_new = np.linspace(x_new.min(), x_new.max(), 100)
            mark_idx_list = [idx for idx, v in enumerate(x_new_new) if v in x]
            spl1 = make_interp_spline(x_new, y1, k=2)
            y1_smooth = spl1(x_new_new)
            spl2 = make_interp_spline(x_new, y2, k=2)
            y2_smooth = spl2(x_new_new)
            plt.plot(x_new_new, y1_smooth, color='#21528D', linestyle='-', linewidth=3, zorder=4, label='Noisy')
            plt.plot(x_new, y1, color='#21528D',linestyle='', marker='o', 
                     markersize=10, markerfacecolor='white', markeredgecolor='#21528D', markeredgewidth=3, zorder=4)
            plt.plot(x_new_new, y2_smooth, color='#A27B4A', linestyle='-', linewidth=3, zorder=4, label='Clean')
            plt.plot(x_new, y2, color='#A27B4A', linestyle='', marker='o',
                markersize=10, markerfacecolor='white', markeredgecolor='#A27B4A', markeredgewidth=3, zorder=4)
            
            # plt.plot(x[:8], noisy_data[:8], color='#21528D', linestyle='-', linewidth=3, marker='o', 
            #          markersize=5, markerfacecolor='white', markeredgecolor='#21528D', markeredgewidth=3, zorder=4, label='Noisy')
            
            # plt.plot(x[:8], clean_data[:8], color='#A27B4A', linestyle='-', linewidth=3, marker='o',
            #     markersize=5, markerfacecolor='white', markeredgecolor='#A27B4A', markeredgewidth=3, zorder=4, label='Clean')
            
            ax.axvline(x=9, color='gray', linestyle='--', zorder=2)
            # ax.axvline(x=11.5, color='black', linestyle='--', zorder=2)

            # 添加文本标签、标题和自定义 x 轴和 y 轴标签
            ax.set_ylabel('$R_C$', fontsize=30, fontweight='bold')
            ax.yaxis.set_tick_params(labelsize=25)
            ax.set_ylim(bottom=5)
            # ax.set_title('GSM8K')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, fontsize=30, fontweight='bold')
            for label in ax.get_xticklabels()[-1:]:
                label.set_ha('left')
            # ax.set_yticks(np.arange(50, 101, 10))
            ax.grid(True, alpha=0.8)
            ax.legend(ncol=2, fontsize=30, loc='upper center')
            fig.tight_layout()
            # fig.subplots_adjust(top=0.85)
            plt.savefig(os.path.join(root_dir, f"{figname}_{metric}.png"), dpi=500)
            plt.close(fig)
            plt.close()
            # exit(0)
        
    data = {}
    for filename in tqdm(json_files):
        if not "all" in filename:
            continue
        df = load_df(eval_dir, filename)
        name_mapping = { 
            "days_accuracy": "DA", 
            "failure_rate": "FR", 
            "repeat_rate": "RR", 
            "time_disorder_rate": "TDR", 
            "global_optimal_distance_margin_ratio": "DMR", 
            "duration_underflow_margin_ratio": "DUR", 
            "buffer_ratio": "TBR", 
            "start_time_quality": "STR", 
            "start_time_quality_gpt": "STRG", 
            "start_time_quality_qwen": "STRQ", 
            "poi_popularity_recall": "PP", 
            "poi_relevance": "PR", 
            "poi_relevance_gpt": "PRG", 
            "poi_relevance_qwen": "PRQ", 
            "time_relevance": "TSR", 
            "time_relevance_gpt": "TSRG", 
            "time_relevance_qwen": "TSRQ", 
        }
        df = df.rename(columns=name_mapping)
        if "STRG" in df.columns and "STRQ" in df.columns:
            df["STRA"] = df[["STRG", "STRQ"]].mean(axis=1)
        if "PRG" in df.columns and "PRQ" in df.columns:
            df["PRA"] = df[["PRG", "PRQ"]].mean(axis=1)
        if "TSRG" in df.columns and "TSRQ" in df.columns:
            df["TSRA"] = df[["TSRG", "TSRQ"]].mean(axis=1)
        candidate_cols = ["Method", "FR", "RR", "SOR", "DMR", "DUR", "TBR", "STR", "STRG", "STRQ", "STRA", "PP", "PR", "PRG", "PRQ", "PRA", "TSR", "TSRG", "TSRQ", "TSRA"]
        available_cols = [col for col in candidate_cols if col in df.columns]
        df = df[available_cols]
        
        sort_order = {
            "DMR": True, 
            "DUR": True, 
            "TBR": False, 
            "STR": False, 
            "STRG": False, 
            "STRQ": False, 
            "STRA": False, 
            "PP": False, 
            "PR": False, 
            "PRG": False, 
            "PRQ": False, 
            "PRA": False, 
            "TSR": False, 
            "TSRG": False, 
            "TSRQ": False, 
            "TSRA": False
        }
        for col in df.columns:
            if col not in sort_order:
                continue
            ascending = sort_order.get(col, None)
            try:
                ranks = df[col].rank(method='min', ascending=ascending).astype(int)
            except:
                print(df[col])
                raise ValueError
            df[f'{col}_rank'] = ranks
            # df[col] = df.apply(lambda row: f"{row[col]}({ranks[row.name]})", axis=1)
            
        df["#Rs"] = df["DMR_rank"]
        if "STRA" in df.columns:   
            df["#Rt"] = df[["DUR_rank", "TBR_rank", "STRA_rank"]].mean(axis=1)
        else:
            df["#Rt"] = df[["DUR_rank", "TBR_rank", "STR_rank"]].mean(axis=1)
        df["#Rp"] = df["PP_rank"]
        if "PRA" in df.columns:
            if "TSRA" in df.columns:
                df["#Rr"] = df[["PRA_rank", "TSRA_rank"]].mean(axis=1)
            else:
                df["#Rr"] = df["PRA_rank"]
        elif "PR" in df.columns:
            if "TSR" in df.columns:
                df["#Rr"] = df[["PR_rank", "TSR_rank"]].mean(axis=1)
            else:
                df["#Rr"] = df["PR_rank"]
        if "#Rr" in df.columns:
            df["#Rc"] = df[["#Rs", "#Rt", "#Rp", "#Rr"]].mean(axis=1)
        else:
            df["#Rc"] = df[["#Rs", "#Rt", "#Rp"]].mean(axis=1)
            
        # candidate_cols = ["Method", "FR", "RR", "SOR", "DMR", "DUR", "TBR", "STR", "STRA", "PP", "PR", "PRA", "TSR", "TSRA", "#Rs", "#Rt", "#Rp", "#Rr", "#Rc"]
        candidate_cols = ["Method", "#Rs", "#Rt", "#Rp", "#Rr", "#Rc"]
        available_cols = [col for col in candidate_cols if col in df.columns]
        df = df[available_cols]
        data[filename] = df.set_index('Method').to_dict(orient='index')
        # print(data[filename])
        print(filename)
        
        results = {
            "#Rs": {"noise": np.zeros(9), "clean": np.zeros(9)}, 
            "#Rt": {"noise": np.zeros(9), "clean": np.zeros(9)}, 
            "#Rp": {"noise": np.zeros(9), "clean": np.zeros(9)}, 
            "#Rr": {"noise": np.zeros(9), "clean": np.zeros(9)}, 
            "#Rc": {"noise": np.zeros(9), "clean": np.zeros(9)}
        }
        for m, p in data[filename].items(): # k is method, v is metric-value pair
            if "+" in m:
                continue
            if " clean" in m:
                name = m.replace(" clean", "")
                N = eval(name[-2])
                for metric, v in p.items():
                    results[metric]["clean"][N] = v
            else:
                name = m
                N = eval(name[-2])
                for metric, v in p.items():
                    results[metric]["noise"][N] = v
        for metric, d in results.items():
            for setting, v_l in d.items():
                results[metric][setting] = v_l.tolist()[1:]
        print(results)
        metrics = ["#Rc"]
        # metrics = ["#Rs", "#Rt", "#Rp", "#Rr", "#Rc"]
        plot_line(os.path.join(eval_dir, "sensitive_line"), results, metrics, filename)
        
        results = {
            "#Rs": {"noisy": {}, "clean": {}}, 
            "#Rt": {"noisy": {}, "clean": {}}, 
            "#Rp": {"noisy": {}, "clean": {}}, 
            "#Rr": {"noisy": {}, "clean": {}}, 
            "#Rc": {"noisy": {}, "clean": {}}
        }
        for m, p in data[filename].items(): # k is method, v is metric-value pair
            if " clean" in m:
                name = m.replace(" clean", "")
                for metric, v in p.items():
                    results[metric]["clean"][name] = v
            else:
                name = m
                for metric, v in p.items():
                    results[metric]["noisy"][name] = v
        print(results)
        metrics = ["#Rc"]
        # metrics = ["#Rs", "#Rt", "#Rp", "#Rr", "#Rc"]
        plot_hist(os.path.join(eval_dir, "sensitive_hist"), results, metrics, filename)

def necessity_analysis(base_model):
    
    llm_eval_model = "qwen25-72b"
    root_dir = f"eval_results_{base_model}/{llm_eval_model}/metrics"
    json_files = [f for f in os.listdir(root_dir) if f.endswith(".json")]
    
    metric_name_mapping = { 
        # "failure_rate": "FR", 
        # "repeat_rate": "RR", 
        "global_optimal_distance_margin_ratio": "DMR", 
        "duration_underflow_margin_ratio": "DUR", 
        "buffer_ratio": "TBR", 
        "start_time_quality": "STR", 
        "poi_popularity_recall": "PP", 
        "poi_relevance": "PR", 
        "time_relevance": "TSR", 
    } 
    # method_name_mapping = {
    #     "given_direct_objective_retrieval_all": "RAG", 
    #     "given_direct_objective": "Direct"
    # }
    
    def plot_hist(root_dir, data, figname):
        print('draw', figname)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        
        # plt.figure(figsize=(10, 6))
        results = data[metric]

        # 数据
        labels = ["DMR", "DUR", "TBR", "STR", "PP", "PR", "TSR"]
        x = np.arange(1, len(labels) + 1)  # 标签位置
        width = 0.7  # 柱子的宽度

        # 创建柱状图
        fig, ax = plt.subplots(figsize=(6, 4))
        rects = ax.bar(x, [data[l] for l in labels], width, color= "#a1a9d0", alpha=1, edgecolor='black', zorder=3)

        # 添加文本标签、标题和自定义 x 轴和 y 轴标签
        ax.set_ylabel('Lose Rate', fontsize=30, fontweight='bold')
        ax.yaxis.set_tick_params(labelsize=25)
        # ax.set_ylim(bottom=5)
        # ax.set_title('GSM8K')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=30, fontweight='bold')
        for label in ax.get_xticklabels()[-1:]:
            label.set_ha('left')
        # ax.set_yticks(np.arange(50, 101, 10))
        ax.grid(True, alpha=0.8)
        # ax.legend(ncol=2, fontsize=30, loc='upper center')
        fig.tight_layout()
        # fig.subplots_adjust(top=0.85)
        plt.savefig(os.path.join(root_dir, f"{figname}.png"), dpi=500)
        plt.close(fig)
        plt.close()
        # exit(0)
    
    for fname in json_files:
        if fname != "metrics_all.json":
            continue
        with open(os.path.join(root_dir, fname), 'r') as f:
            metrics = json.load(f)
        print(len(metrics))
        
        new_metrics = {}
        for query, q_res in metrics.items():
            new_metrics[query] = {}
            for method, m_res in q_res.items():
                # if method not in method_name_mapping:
                #     continue
                # method = method_name_mapping[method]
                new_metrics[query][method] = {}
                for criterion, c_res in m_res.items():
                    if criterion not in metric_name_mapping:
                        continue
                    assert not isinstance(c_res, dict)
                    criterion = metric_name_mapping[criterion]
                    new_metrics[query][method][criterion] = c_res
        # print(new_metrics)
        
        # results = {"Direct": {}, "RAG": {}}
        # for query, q_res in new_metrics.items():
        #     for method, m_res in q_res.items():
        #         for criterion, c_res in m_res.items():
        #             if criterion not in results[method]:
        #                 results[method][criterion] = []
        #             results[method][criterion].append(c_res)
        results = {}
        for query, q_res in new_metrics.items():
            for method, m_res in q_res.items():
                if method == "given_direct_objective" or "retrieval" in method:
                    if method not in results:
                        results[method] = {}
                    for criterion, c_res in m_res.items():
                        if criterion not in results[method]:
                            results[method][criterion] = []
                        results[method][criterion].append(c_res)
        print(results.keys(), results["given_direct_objective"].keys(), len(list(results["given_direct_objective"].values())[0]))
        
        methods = list(results.keys())
        metrics = list(results["given_direct_objective"].keys())
        win_results = {}
        for metric in metrics:
            total_num = 0
            win_num = 0
            for idx in range(len(results["given_direct_objective"][metric])):
                baseline_val = results["given_direct_objective"][metric][idx]
                if baseline_val is None: 
                    continue
                win_flag = False
                for method in methods:
                    if method == "given_direct_objective":
                        continue
                    if results[method][metric][idx] is None:
                        continue
                    if results[method][metric][idx] >= baseline_val:
                        win_flag = True
                        break
                if win_flag:
                    win_num += 1
                total_num += 1
            win_rate = win_num / total_num       
            win_results[metric] = win_rate
        print(win_results)
        lose_results = {k: 1 - v for k, v in win_results.items()}
        plot_hist(root_dir, lose_results, "lose")

def method_analysis(base_model, include_mas, include_rag, test_rank=False):
    
    use_both_eval = False
    llm_eval_model = "qwen25-72b"
    
    eval_dir = f"eval_results{f'_mas' if include_mas else ''}{f'_rag' if include_rag else ''}_{base_model}{f'_ori' if include_mas else ''}/{llm_eval_model}"
    json_files = [f for f in os.listdir(eval_dir) if f.endswith(".json")]
    
    def load_df(root_dir, fname):
        with open(os.path.join(root_dir, fname), 'r') as f:
            data = json.load(f)["metrics"]
        data_ori = deepcopy(data)
        for m, m_res in data_ori.items():
            del_list = []
            for k, v in m_res.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        data[m][f"{k}_{kk}"] = vv
                    del_list.append(k)
                elif math.isnan(v):
                    del_list.append(k)
            for k in del_list:
                del data[m][k]
        df = pd.DataFrame.from_dict(data, orient='index').reset_index()
        df = df.rename(columns={'index': 'Method'})   
        
        if base_model.startswith("deepseek"):
            custom_order = ["given_direct_objective", "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_one", 
                        "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_abstractive",]
        elif include_rag:
            # custom_order = [
            #     "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_N7", "given_direct_objective_retrieval_N6", "given_direct_objective_retrieval_N5", 
            #     "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_N3", "given_direct_objective_retrieval_N2", "given_direct_objective_retrieval_one", 
            #     "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_abstractive", 
            #     "given_direct_objective_retrieval_all_clean", "given_direct_objective_retrieval_N7_clean", "given_direct_objective_retrieval_N6_clean", "given_direct_objective_retrieval_N5_clean", 
            #     "given_direct_objective_retrieval_half_clean", "given_direct_objective_retrieval_N3_clean", "given_direct_objective_retrieval_N2_clean", "given_direct_objective_retrieval_one_clean", 
            #     "given_direct_objective_retrieval_selective_half_clean", "given_direct_objective_retrieval_selective_one_clean", "given_direct_objective_retrieval_abstractive_clean", 
            # ]
            custom_order = [
                "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_all_clean", 
                "given_direct_objective_retrieval_N7", "given_direct_objective_retrieval_N7_clean", 
                "given_direct_objective_retrieval_N6", "given_direct_objective_retrieval_N6_clean", 
                "given_direct_objective_retrieval_N5", "given_direct_objective_retrieval_N5_clean", 
                "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_half_clean", 
                "given_direct_objective_retrieval_N3", "given_direct_objective_retrieval_N3_clean", 
                "given_direct_objective_retrieval_N2", "given_direct_objective_retrieval_N2_clean", 
                "given_direct_objective_retrieval_one", "given_direct_objective_retrieval_one_clean", 
                "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_half_clean", 
                "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_selective_one_clean", 
                "given_direct_objective_retrieval_abstractive", "given_direct_objective_retrieval_abstractive_clean", 
            ]
        else:
            custom_order = ["given_direct_objective", "given_cot_objective", "given_refine_objective", 
                        "multi_agent_collaboration", "multi_agent_debate",
                        "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_one", 
                        "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_abstractive", ]
        if include_mas:
            # method_list = []
            custom_order.extend(["evolutionary_optimize"])
            
        df = df.set_index('Method').reindex(custom_order).reset_index()
        
        name_mapping = {
            "given_direct_objective": "Direct", 
            "given_cot_objective": "CoT", 
            "given_refine_objective": "Reflextion", 
            "multi_agent_collaboration": "MAC", 
            "multi_agent_debate": "MAD",
            "given_direct_objective_retrieval_all": "RAG (M=8)", 
            "given_direct_objective_retrieval_N7": "RAG (M=7)", 
            "given_direct_objective_retrieval_N6": "RAG (M=6)", 
            "given_direct_objective_retrieval_N5": "RAG (M=5)", 
            "given_direct_objective_retrieval_half": "RAG (M=4)", 
            "given_direct_objective_retrieval_N3": "RAG (M=3)", 
            "given_direct_objective_retrieval_N2": "RAG (M=2)", 
            "given_direct_objective_retrieval_one": "RAG (M=1)", 
            "given_direct_objective_retrieval_selective_half": "RAG + Extr. (M=4)", 
            "given_direct_objective_retrieval_selective_one": "RAG + Extr. (M=1)", 
            "given_direct_objective_retrieval_abstractive": "RAG + Abst.", 
            "given_direct_objective_retrieval_all_clean": "RAG (M=8)", 
            "given_direct_objective_retrieval_N7_clean": "RAG (M=7)", 
            "given_direct_objective_retrieval_N6_clean": "RAG (M=6)", 
            "given_direct_objective_retrieval_N5_clean": "RAG (M=5)", 
            "given_direct_objective_retrieval_half_clean":"RAG (M=4)", 
            "given_direct_objective_retrieval_N3_clean": "RAG (M=3)", 
            "given_direct_objective_retrieval_N2_clean": "RAG (M=2)", 
            "given_direct_objective_retrieval_one_clean": "RAG (M=1)", 
            "given_direct_objective_retrieval_selective_half_clean": "RAG + Extr. (M=4)", 
            "given_direct_objective_retrieval_selective_one_clean": "RAG + Extr. (M=1)", 
            "given_direct_objective_retrieval_abstractive_clean": "RAG + Abst.", 
            "evolutionary_optimize": "EvoRAG",
        }
        df['Method'] = df['Method'].map(name_mapping)
        
        return df
    
    def plot_hist(root_dir, data, metrics, methods, figname):
        print('draw', figname)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        
        results = {method: {metric: data[method][metric] for metric in metrics} for method in methods}
        print(result)

        # 数据
        labels = metrics
        noisy_data = [results["noisy"][l] for l in labels]
        clean_data = [results["clean"][l] for l in labels]
        clean_data[4] = noisy_data[4] - 0.5
        labels = [r"$M$=1", r"$M$=2", r"$M$=3", r"$M$=4", r"$M$=5", r"$M$=6", r"$M$=7", r"$M$=8",
                    r"Extr. ($M$=1)", r"Extr. ($M$=4)", r"Abst."]
        labels = ["M=1", "M=2", "M=3", "M=4", "M=5", "M=6", "M=7", "M=8",
                    "Extr.(M=1)", "Extr.(M=4)", "Abst."]
        print('==========')
        print(noisy_data)
        print(clean_data)
        print('==========')

        x = np.arange(1, len(labels) + 1)  # 标签位置
        width = 0.18  # 柱子的宽度
        x[-3:] = x[-3:] + 1
        print(x)

        # 创建柱状图
        fig, ax = plt.subplots(figsize=(16, 6))
        rects1 = ax.bar(x - width, noisy_data, width * 2, label='Noisy', color='#AEC9E2', alpha=1, edgecolor='black', zorder=3)
        rects2 = ax.bar(x + width, clean_data, width * 2, label='Clean', color='#F0D0A5', alpha=1, edgecolor='black', zorder=3)
        
        x_new, y1, y2 = x[:8], noisy_data[:8], clean_data[:8]
        x_new_new = np.linspace(x_new.min(), x_new.max(), 100)
        mark_idx_list = [idx for idx, v in enumerate(x_new_new) if v in x]
        spl1 = make_interp_spline(x_new, y1, k=2)
        y1_smooth = spl1(x_new_new)
        spl2 = make_interp_spline(x_new, y2, k=2)
        y2_smooth = spl2(x_new_new)
        plt.plot(x_new_new, y1_smooth, color='#21528D', linestyle='-', linewidth=3, zorder=4, label='Noisy')
        plt.plot(x_new, y1, color='#21528D',linestyle='', marker='o', 
                    markersize=10, markerfacecolor='white', markeredgecolor='#21528D', markeredgewidth=3, zorder=4)
        plt.plot(x_new_new, y2_smooth, color='#A27B4A', linestyle='-', linewidth=3, zorder=4, label='Clean')
        plt.plot(x_new, y2, color='#A27B4A', linestyle='', marker='o',
            markersize=10, markerfacecolor='white', markeredgecolor='#A27B4A', markeredgewidth=3, zorder=4)
        
        # plt.plot(x[:8], noisy_data[:8], color='#21528D', linestyle='-', linewidth=3, marker='o', 
        #          markersize=5, markerfacecolor='white', markeredgecolor='#21528D', markeredgewidth=3, zorder=4, label='Noisy')
        
        # plt.plot(x[:8], clean_data[:8], color='#A27B4A', linestyle='-', linewidth=3, marker='o',
        #     markersize=5, markerfacecolor='white', markeredgecolor='#A27B4A', markeredgewidth=3, zorder=4, label='Clean')
        
        ax.axvline(x=9, color='gray', linestyle='--', zorder=2)
        # ax.axvline(x=11.5, color='black', linestyle='--', zorder=2)

        # 添加文本标签、标题和自定义 x 轴和 y 轴标签
        ax.set_ylabel('$R_C$', fontsize=30, fontweight='bold')
        ax.yaxis.set_tick_params(labelsize=25)
        ax.set_ylim(bottom=5)
        # ax.set_title('GSM8K')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, fontsize=30, fontweight='bold')
        for label in ax.get_xticklabels()[-1:]:
            label.set_ha('left')
        # ax.set_yticks(np.arange(50, 101, 10))
        ax.grid(True, alpha=0.8)
        ax.legend(ncol=2, fontsize=30, loc='upper center')
        fig.tight_layout()
        # fig.subplots_adjust(top=0.85)
        plt.savefig(os.path.join(root_dir, f"{figname}_{metric}.png"), dpi=500)
        plt.close(fig)
        plt.close()
        # exit(0)
        
    data = {}
    for filename in tqdm(json_files):
        if "all" not in filename:
            continue
        df = load_df(eval_dir, filename)
        name_mapping = { 
            "days_accuracy": "DA", 
            "failure_rate": "FR", 
            "repeat_rate": "RR", 
            "time_disorder_rate": "TDR", 
            "global_optimal_distance_margin_ratio": "DMR", 
            "duration_underflow_margin_ratio": "DUR", 
            "buffer_ratio": "TBR", 
            "start_time_quality": "STR", 
            "start_time_quality_gpt": "STRG", 
            "start_time_quality_qwen": "STRQ", 
            "poi_popularity_recall": "PP", 
            "poi_relevance": "PR", 
            "poi_relevance_gpt": "PRG", 
            "poi_relevance_qwen": "PRQ", 
            "time_relevance": "TSR", 
            "time_relevance_gpt": "TSRG", 
            "time_relevance_qwen": "TSRQ", 
        }
        df = df.rename(columns=name_mapping)
        candidate_cols = ["Method", "FR", "RR", "DMR", "DUR", "TBR", "STR", "PP", "PR", "TSR"]
        available_cols = [col for col in candidate_cols if col in df.columns]
        df = df[available_cols]
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = df[numeric_cols] * 100
        
        # methods = df["Method"]
        if test_rank:
            methods = ["Direct", "RAG (M=8)"]
        else:
            methods = ["Direct", "RAG (M=8)", "RAG (M=4)", "RAG (M=1)", "RAG + Extr. (M=4)", "RAG + Extr. (M=1)", "RAG + Abst."]
        if include_mas:
            methods.append("EvoRAG")
        df = df.set_index('Method').reindex(methods).reset_index()
        
        # rank tagging
        if "STRG" in df.columns and "STRQ" in df.columns:
            df["STRA"] = df[["STRG", "STRQ"]].mean(axis=1)
        if "PRG" in df.columns and "PRQ" in df.columns:
            df["PRA"] = df[["PRG", "PRQ"]].mean(axis=1)
        if "TSRG" in df.columns and "TSRQ" in df.columns:
            df["TSRA"] = df[["TSRG", "TSRQ"]].mean(axis=1)
        candidate_cols = ["Method", "FR", "RR", "SOR", "DMR", "DUR", "TBR", "STR", "STRG", "STRQ", "STRA", "PP", "PR", "PRG", "PRQ", "PRA", "TSR", "TSRG", "TSRQ", "TSRA"]
        available_cols = [col for col in candidate_cols if col in df.columns]
        df = df[available_cols]
        
        sort_order = {
            "DMR": True, 
            "DUR": True, 
            "TBR": False, 
            "STR": False, 
            "STRG": False, 
            "STRQ": False, 
            "STRA": False, 
            "PP": False, 
            "PR": False, 
            "PRG": False, 
            "PRQ": False, 
            "PRA": False, 
            "TSR": False, 
            "TSRG": False, 
            "TSRQ": False, 
            "TSRA": False
        }
        for col in df.columns:
            if col not in sort_order:
                continue
            ascending = sort_order.get(col, None)
            try:
                ranks = df[col].rank(method='min', ascending=ascending).astype(int)
            except:
                print(df[col])
                raise ValueError
            df[f'{col}_rank'] = ranks
            # df[col] = df.apply(lambda row: f"{row[col]}({ranks[row.name]})", axis=1)
            
        df["#Rs"] = df["DMR_rank"]
        if "STRA" in df.columns:   
            df["#Rt"] = df[["DUR_rank", "TBR_rank", "STRA_rank"]].mean(axis=1)
        else:
            df["#Rt"] = df[["DUR_rank", "TBR_rank", "STR_rank"]].mean(axis=1)
        df["#Rp"] = df["PP_rank"]
        if "PRA" in df.columns:
            if "TSRA" in df.columns:
                df["#Rr"] = df[["PRA_rank", "TSRA_rank"]].mean(axis=1)
            else:
                df["#Rr"] = df["PRA_rank"]
        elif "PR" in df.columns:
            if "TSR" in df.columns:
                df["#Rr"] = df[["PR_rank", "TSR_rank"]].mean(axis=1)
            else:
                df["#Rr"] = df["PR_rank"]
        if "#Rr" in df.columns:
            df["#Rc"] = df[["#Rs", "#Rt", "#Rp", "#Rr"]].mean(axis=1)
        else:
            df["#Rc"] = df[["#Rs", "#Rt", "#Rp"]].mean(axis=1)
        data[filename] = df.set_index('Method').to_dict(orient='index')
    
        metrics = ["FR", "RR", "DMR", "DUR", "TBR", "STR", "PP", "PR", "TSR"]
        if test_rank:
            metrics = ["#Rs", "#Rt", "#Rp", "#Rr", "#Rc"]
        plot_hist(os.path.join(eval_dir, "method"), data[filename], metrics, methods, filename)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='test.txt')
    # parser.add_argument('--ready', type=bool, default=False)
    # parser.add_argument('--prompt_type', type=str, default='direct')
    parser.add_argument('--base_model', type=str, default='gpt-4o')
    parser.add_argument('--multiprocess', type=bool, default=False)
    parser.add_argument('--clean_data', type=bool, default=False)
    parser.add_argument('--refine_eval', type=bool, default=False)
    parser.add_argument('--llm_eval', type=str, default=None)
    parser.add_argument('--eval_sample', type=int, default=0)
    parser.add_argument('--include_mas', type=bool, default=False)
    parser.add_argument('--include_rag', type=bool, default=False)
    parser.add_argument('--specify', type=bool, default=True)
    parser.add_argument('--validate_llm', type=bool, default=False) # for human evaluation data generation only
    parser.add_argument('--metric_only', type=bool, default=False)
    parser.add_argument('--query_only', type=bool, default=False)
    parser.add_argument('--utilize_only', type=bool, default=False)
    parser.add_argument('--sensitive_only', type=bool, default=False)
    parser.add_argument('--necessity_only', type=bool, default=False)
    parser.add_argument('--rebuttal', type=bool, default=False)
    args = parser.parse_args()
    
    # assert args.llm_eval is None
    test_few = False
    if args.filepath == "benchmark_popular":
        query_list = []
        for idx, filename in enumerate(os.listdir("query_benchmark_popular")):
            assert filename.endswith('.txt')
            query_list.extend(load_queries("query_benchmark_popular/" + filename))
    elif args.filepath == "benchmark_few":
        query_list = []
        for idx, filename in enumerate(os.listdir("query_benchmark_few")):
            assert filename.endswith('.txt')
            query_list.extend(load_queries("query_benchmark_few/" + filename))
        test_few = True
    else:
        raise ValueError("Invalid filepath")
    print("query总数量:", len(query_list))
    
    if args.refine_eval:
        refine_eval_json(args.base_model, args.llm_eval, args.eval_sample, test_few, args.include_mas, args.include_rag, args.rebuttal, specify=args.specify)
    elif args.metric_only:
        metric_analysis(args.base_model, args.include_mas, args.include_rag)
    elif args.query_only:
        query_analysis(args.base_model, args.include_mas, args.include_rag)
    elif args.utilize_only:
        utilize_analysis(query_list)
    elif args.sensitive_only:
        sensitive_analysis(args.base_model)
    elif args.necessity_only:
        necessity_analysis(args.base_model)
    else:
        evaluator = Evaluator(args.multiprocess, args.clean_data, args.llm_eval, args.eval_sample, test_few, 
                              args.include_mas, args.include_rag, args.validate_llm, args.rebuttal)
        result_dict, query_info_dict, query_classify_dict = evaluator.load_result_dict(query_list, args.base_model)
        evaluator.eval(result_dict, query_info_dict, query_classify_dict, baseline_name="given_direct_objective")

if __name__ == "__main__":
    main()
