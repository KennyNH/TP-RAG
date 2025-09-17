"""
模块的总运行脚本，用于生成agentpedia。
1:首先配置环境，确保脚本能够访问项目的其他部分，然后解析命令行提供的查询字符串
2:使用`GenerateArticle`类生成相应的文章。
"""
import os
import argparse
import json
import codecs
import concurrent.futures
import time
import pprint
import pickle
import threading
import numpy as np
from copy import deepcopy
from geopy.distance import geodesic
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing

from agentpedia.config import Config
from agentpedia.context.article import ArticleContext
from agentpedia.utils.request_llm import RequestLLM
# from agentpedia.utils.request_map import RequestMap
from agentpedia.utils.cache import Cache
from agentpedia.logger.logger_config import get_logger
from agentpedia.utils.format import Format
from agentpedia.utils.spider_sug import SpiderSug

class BatchStrategyDriver:
    """
    批量生产流程
    """

    def __init__(self, multiprocess, base_model, prompt_only=False, base_url=None, num_threads=5, val_comp=False):
        """
            Initializes the class with default values for number of threads and directory names.
        
        Args:
            num_threads (int, optional): Number of threads to use for parallel processing. Defaults to 10.
        
        Returns:
            None
        """
        self.config = Config()
        self.config.need_full_log = False
        self.num_threads = num_threads
        self.dir_name = "article_data"
        self.retriever_name = "retriever_data"
        self.sug_cq_name = "sug_cq_data"
        self.md_name = "md_data"
        self.spider_sug = SpiderSug(self.config)
        self.local = threading.local()
        # self._prepare_directories()

        self.multiprocess = multiprocess
        self.base_model = base_model
        self.prompt_only = prompt_only
        if prompt_only:
            self.prompt_dict = {}
        self.base_url = base_url
        
        self.val_comp = val_comp

    def _prepare_directories(self):
        """
            创建所需的目录，如果它们不存在。
        返回: None
        """
        for dir_name in [self.dir_name, self.retriever_name, self.sug_cq_name, self.md_name]:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                
    def generate(self, query, res_query_task_sug_dict, change_querys_dict):
        """
            根据输入的查询语句，生成相应的结果。包括结构体的检索、改变查询语句、结果建议等操作。
        然后对获取到的结果进行标题提取、内容请求和合并操作，最后将结果保存到文件中。
        
        Args:
            query (str): 用户输入的查询语句。
            res_query_task_sug_dict (dict): 字典格式，key为查询语句，value为结果建议列表。
            change_querys_dict (dict): 字典格式，key为查询语句，value为改变查询语句列表。
        
        Returns:
            None.
        
        Raises:
            None.
        """
        # try:
        if 2 > 1:
            _MODEL = self.base_model
            plan_dir = f"plan_data_{_MODEL}"
            if not os.path.exists(plan_dir):
                os.makedirs(plan_dir)
            plan_path = os.path.join(plan_dir, f"{query}-{self.prompt_type}.pkl")
            if os.path.exists(plan_path):
                plan = pickle.load(open(plan_path, "rb"))
                # print(plan)
                # return None, query
                if plan is not None and isinstance(plan, dict) and not self.val_comp:
                    return None, query

            # prepare data
            construct_data = pickle.load(open(f"construct_data_baseline/{query}-data_construct.pkl", 'rb'))
            attraction_extract_result = construct_data["poi_list_no_cheat"]
            retrieved_results_raw = construct_data["plan_extract_result_list"]
            if len(retrieved_results_raw) < 8:
                # print(len(retrieved_results_raw), 8)
                return None, query
            retrieved_results_raw = retrieved_results_raw[:8]
            # print(len(attraction_extract_result), attraction_extract_result[0])
            # print(len(retrieved_results_raw), retrieved_results_raw[0])
            if self.prompt_type == "reference_denoise":
                if "plan_extract_result_list_clean" in construct_data:
                    assert len(construct_data["plan_extract_result_list_clean"]) == 8
                    # print(len(construct_data["plan_extract_result_list_clean"]), len(retrieved_results_raw))
                    return None, query
            retrieved_results_raw_clean = construct_data["plan_extract_result_list_clean"]
            print(self.prompt_type, query)

            logger = get_logger(query)
            format_util = Format(query)
            context = ArticleContext(query)
            
            retriever_data = dict()
            retriever_data["1_query"] = query
            logger.info(f"query: {query} prompt_type: {self.prompt_type}")
            change_query_list = change_querys_dict.get(query, [])
            retriever_data["1_change_query_list"] = change_query_list

            self.local.config = Config()  # 为每个线程使用独立的配置实例
            self.local.config.need_full_log = False

            # 加载旅游攻略配置文件
            self.local.config.final_config.update(
                self.local.config._load_file_config('agentpedia/config/category_config/travel_plan_baseline_config.yaml'))

            self.local.config.model = _MODEL
            if self.base_url is not None:
                self.local.config.base_url = self.base_url
            # if _MODEL.startswith("deepseek"):
            #     self.local.config.temperature = 0.7
            self.local.cache = Cache("cache_data", query + "_baseline")
            self.local.request_llm = RequestLLM(self.local.config)
            # self.local.request_map = RequestMap(self.local.config, self.multiprocess)
            # self.local.dqa_mrc = DqaMrc(self.local.config)
            # generate_article = ArticleGenerator(query, self.local.config, context)
            res_sug_list = res_query_task_sug_dict.get(query, [])
            retriever_data["1_res_sug_list"] = res_sug_list

            # 大纲标题生成检索数量配比
            struct_quato = self.local.config.struct_quato
            struct_site = self.local.config.struct_site
            logger.info(f"struct_quato (general_count, biji_count, video_count, undisplay_count, google_url_count):\
                    {struct_quato} struct_quato: {struct_site}")

            CACHE_FLAG = True

            # 景点规划生成
            start_time = time.time()

            # planning from scratch
            if self.prompt_type == "given_direct":
                prompt = str(self.local.config.given_direct_prompt) % (query, query, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            elif self.prompt_type == "given_cot":
                prompt = str(self.local.config.given_cot_prompt) % (query, query, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)["规划"]
            elif self.prompt_type == "given_refine":
                num_iters = 1
                prompt = str(self.local.config.given_direct_prompt) % (query, query, attraction_extract_result)
                logger.info("init prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result 0: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
                for index in range(1, num_iters + 1):
                    prompt = str(self.local.config.given_feedback_prompt) % (query, attraction_extract_result, a_merge_result)
                    logger.info(f"feedback prompt {index}: \n%s\n", prompt)
                    feedback = self.local.request_llm.get_llm_result(prompt)
                    logger.info(f"feedback {index}: \n%s\n", feedback)
                    prompt = str(self.local.config.given_refine_prompt) % (query, query, attraction_extract_result, a_merge_result, feedback)
                    logger.info(f"refine prompt {index}: \n%s\n", prompt)
                    a_merge_result = self.local.request_llm.get_llm_result(prompt)
                    logger.info(f"attraction plan result {index}: \n%s\n", a_merge_result)
                    a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            elif self.prompt_type == "given_direct_objective":
                prompt = str(self.local.config.given_direct_objective_prompt) % (query, query, query, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            elif self.prompt_type == "given_cot_objective":
                prompt = str(self.local.config.given_cot_objective_prompt) % (query, query, query,  attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)["规划"]
            elif self.prompt_type == "given_refine_objective":
                num_iters = 1
                prompt = str(self.local.config.given_direct_objective_prompt) % (query, query, query, attraction_extract_result)
                logger.info("init prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result 0: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
                for index in range(1, num_iters + 1):
                    prompt = str(self.local.config.given_feedback_objective_prompt) % (query, query, attraction_extract_result, a_merge_result)
                    logger.info(f"feedback prompt {index}: \n%s\n", prompt)
                    feedback = self.local.request_llm.get_llm_result(prompt)
                    logger.info(f"feedback {index}: \n%s\n", feedback)
                    prompt = str(self.local.config.given_refine_objective_prompt) % (query, query, query, attraction_extract_result, a_merge_result, feedback)
                    logger.info(f"refine prompt {index}: \n%s\n", prompt)
                    a_merge_result = self.local.request_llm.get_llm_result(prompt)
                    logger.info(f"attraction plan result {index}: \n%s\n", a_merge_result)
                    a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            # multi-agent
            elif self.prompt_type == "multi_agent_debate":
                num_iters = 1
                prompt = str(self.local.config.given_direct_objective_prompt) % (query, query, query, attraction_extract_result)
                logger.info("init prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result 0: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
                for index in range(1, num_iters + 1):
                    # debate
                    prompt_list = []
                    order_list = ["空间", "时间", "语义", "相关性"]
                    prompt_list.append(str(self.local.config.given_mad_spatial_feedback_prompt) % (query, attraction_extract_result, a_merge_result))
                    prompt_list.append(str(self.local.config.given_mad_temporal_feedback_prompt) % (query, attraction_extract_result, a_merge_result))
                    prompt_list.append(str(self.local.config.given_mad_semantic_feedback_prompt) % (query, attraction_extract_result, a_merge_result))
                    prompt_list.append(str(self.local.config.given_mad_relevance_feedback_prompt) % (query, query, attraction_extract_result, a_merge_result))
                    feedback_list = self._fetch_list_results(prompt_list)
                    feedback_text = {}
                    for idx, feedback in enumerate(feedback_list):
                        prompt = prompt_list[idx]
                        profile = order_list[idx]
                        logger.info(f"feedback prompt {profile} {index}: \n%s\n", prompt)
                        logger.info(f"feedback {profile} {index}: \n%s\n", feedback)
                        feedback_text[profile] = feedback
                    feedback_text = "\n".join([f"{k}:\n{v}" for k, v in feedback_text.items()])
                    prompt = str(self.local.config.given_refine_objective_prompt) % (query, query, query, attraction_extract_result, a_merge_result, feedback_text)
                    logger.info(f"refine prompt {index}: \n%s\n", prompt)
                    a_merge_result = self.local.request_llm.get_llm_result(prompt)
                    logger.info(f"attraction plan result {index}: \n%s\n", a_merge_result)
                    a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            elif self.prompt_type == "multi_agent_collaboration":
                prompt = str(self.local.config.given_mac_manager_decompose_prompt) % (query, query, query)
                logger.info("decompose prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                sub_task_info = self.local.request_llm.get_llm_result(prompt)
                logger.info("decompose result 0: \n%s\n", sub_task_info)
                sub_task_info = self.local.request_llm.parse_json_response(sub_task_info, logger)
                
                pre_sub_task_results = {}
                for task_name, sub_task in sub_task_info.items():
                    pre_sub_task_text = "\n".join([f"{k}:{v['子问题描述']}\n{v['子问题回答']}" for k, v in pre_sub_task_results.items()])
                    prompt = str(self.local.config.given_mac_executor_prompt) % (query, attraction_extract_result, sub_task, pre_sub_task_text)
                    logger.info(f"sub task {task_name} solve prompt: \n%s\n", prompt)
                    result = self.local.request_llm.get_llm_result(prompt)
                    logger.info(f"sub task {task_name} solve result: \n%s\n", result)
                    pre_sub_task_results[task_name] = {"子问题描述": sub_task["子问题描述"], "子问题回答": result}
                    
                pre_sub_task_text = "\n".join([f"{k}:{v['子问题描述']}\n{v['子问题回答']}" for k, v in pre_sub_task_results.items()]) 
                prompt = str(self.local.config.given_mac_manager_compose_prompt) % (query, query, query, attraction_extract_result, pre_sub_task_text)
                logger.info(f"compose prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info(f"attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            # retrieval-augmented planning
            elif self.prompt_type == "reference" or self.prompt_type == "reference_best" or self.prompt_type == "reference_denoise":
                attraction_dict = {p["名称"]: p for p in attraction_extract_result}
                if self.prompt_type == "reference_denoise":
                    if "plan_extract_result_list_clean" in construct_data:
                        return None, query
                    assert _MODEL == "qwen25-72b"
                    retrieved_results_clean = []
                    for idx, r in enumerate(retrieved_results_raw):
                        cur_plan = r
                        retry = 0
                        while True and retry < 10:
                            prompt = str(self.local.config.reference_denoise_prompt) % (list(attraction_dict.keys()), cur_plan)
                            logger.info(f"reference denoise {idx} prompt: \n%s\n", prompt)
                            result = self.local.request_llm.get_llm_result(prompt)
                            logger.info(f"reference denoise {idx} result: \n%s\n", result)
                            cur_plan = self.local.request_llm.parse_json_response(result, logger)
                            flag = True
                            for _, p_l in cur_plan.items():
                                for p in p_l:
                                    if p["名称"] not in list(attraction_dict.keys()):
                                        flag = False
                                        break
                            if flag:
                                break
                            retry += 1
                        retrieved_results_clean.append(cur_plan)
                    construct_data["plan_extract_result_list_clean"] = retrieved_results_clean
                    pickle.dump(construct_data, open(f"construct_data_baseline/{query}-data_construct.pkl", 'wb'))
                else:
                    assert "plan_extract_result_list_clean" in construct_data
                    retrieved_results_clean = construct_data["plan_extract_result_list"]
                if self.prompt_type == "reference_denoise":
                    logger.info(f"elapsed time: {time.time() - start_time}s")
                    print(f"denoise完成: {time.time() - start_time}s")
                    return None, query
                # print(retrieved_results_clean)
                # print("-------\n")
                if self.prompt_type == "reference_best":
                    retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_clean)])
                    num_select = 1
                    retrieval_selective_prompt = str(self.local.config.retrieval_selective_objective_prompt) % (
                        query, num_select, query, attraction_extract_result, retrieved_results, num_select)
                    logger.info("retrieval selective prompt: \n%s\n", retrieval_selective_prompt)
                    selected_indices = self.local.request_llm.get_llm_result(retrieval_selective_prompt)
                    logger.info("retrieval selective result: \n%s\n", selected_indices)
                    selected_indices = self.local.request_llm.parse_json_response(selected_indices, logger)["筛选索引"]
                    if isinstance(selected_indices, list):
                        selected_indices = [i if isinstance(i, int) else eval(i) for i in selected_indices]
                    else:
                        if isinstance(selected_indices, str):
                            selected_indices = eval(selected_indices)
                        assert isinstance(selected_indices, int)
                        selected_indices = [selected_indices]
                    retrieved_results = [retrieved_results_clean[idx] for idx in selected_indices]
                    assert len(retrieved_results) == num_select
                    if isinstance(retrieved_results, list):
                        reference = retrieved_results[0]
                else:
                    reference = retrieved_results_clean[0]
                # print(reference)
                # reference = {k: [{"名称": p["名称"]} for p in v] for k, v in reference.items()}
                name_list = [p["名称"] for _, p_l in reference.items() for p in p_l]
                info_dict = {k: v for k, v in attraction_dict.items() if k in name_list}
                prompt = str(self.local.config.reference_prompt) % (info_dict, reference)
                logger.info("reference prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            elif self.prompt_type == "given_direct_retrieval_all":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw)])

                prompt = str(self.local.config.given_direct_retrieval_prompt) % (query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            elif self.prompt_type == "given_direct_retrieval_half":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw[:len(retrieved_results_raw)//2])])

                prompt = str(self.local.config.given_direct_retrieval_prompt) % (query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            elif self.prompt_type == "given_direct_retrieval_one":
                retrieved_results = retrieved_results_raw[0]

                prompt = str(self.local.config.given_direct_retrieval_prompt) % (query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            elif self.prompt_type == "given_direct_retrieval_selective_half":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw)])

                num_select = len(retrieved_results_raw) // 2
                retrieval_selective_prompt = str(self.local.config.retrieval_selective_prompt) % (query, num_select, retrieved_results, num_select)
                logger.info("retrieval selective prompt: \n%s\n", retrieval_selective_prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = retrieval_selective_prompt
                    return None, query
                selected_indices = self.local.request_llm.get_llm_result(retrieval_selective_prompt)
                logger.info("retrieval selective result: \n%s\n", selected_indices)
                selected_indices = self.local.request_llm.parse_json_response(selected_indices, logger)["筛选索引"]
                selected_indices = [i if isinstance(i, int) else eval(i) for i in selected_indices]
                retrieved_results = [retrieved_results_raw[idx] for idx in selected_indices]
                assert len(retrieved_results) == num_select

                prompt = str(self.local.config.given_direct_retrieval_prompt) % (query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger) 
            elif self.prompt_type == "given_direct_retrieval_selective_one":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw)])

                num_select = 1
                retrieval_selective_prompt = str(self.local.config.retrieval_selective_prompt) % (query, num_select, retrieved_results, num_select)
                logger.info("retrieval selective prompt: \n%s\n", retrieval_selective_prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = retrieval_selective_prompt
                    return None, query
                selected_indices = self.local.request_llm.get_llm_result(retrieval_selective_prompt)
                logger.info("retrieval selective result: \n%s\n", selected_indices)
                selected_indices = self.local.request_llm.parse_json_response(selected_indices, logger)["筛选索引"]
                selected_indices = [i if isinstance(i, int) else eval(i) for i in selected_indices]
                retrieved_results = [retrieved_results_raw[idx] for idx in selected_indices]
                assert len(retrieved_results) == num_select

                prompt = str(self.local.config.given_direct_retrieval_prompt) % (query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger) 
            elif self.prompt_type == "given_direct_retrieval_abstractive":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw)])

                retrieval_abstractive_prompt = str(self.local.config.retrieval_abstractive_prompt) % (query, retrieved_results)
                logger.info("retrieval abstractive prompt: \n%s\n", retrieval_abstractive_prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = retrieval_abstractive_prompt
                    return None, query
                retrieved_results = self.local.request_llm.get_llm_result(retrieval_abstractive_prompt)
                logger.info("retrieval abstractive result: \n%s\n", retrieved_results)
                retrieved_results = self.local.request_llm.parse_json_response(retrieved_results, logger)["总结结果"]

                prompt = str(self.local.config.given_direct_retrieval_prompt) % (query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger) 
            elif self.prompt_type == "given_direct_objective_retrieval_all":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw)])

                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            elif self.prompt_type == "given_direct_objective_retrieval_half":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw[:len(retrieved_results_raw)//2])])

                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            elif self.prompt_type == "given_direct_objective_retrieval_one":
                retrieved_results = retrieved_results_raw[0]

                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            elif self.prompt_type == "given_direct_objective_retrieval_selective_half":
                if self.val_comp:
                    val_comp_dir = "validate_compression"
                    if not os.path.exists(val_comp_dir):
                        os.makedirs(val_comp_dir)
                    retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw)])
                    num_select = len(retrieved_results_raw) // 2
                    retrieval_selective_prompt = str(self.local.config.retrieval_selective_objective_prompt) % (
                        query, num_select, query, attraction_extract_result, retrieved_results, num_select)
                    logger.info("retrieval selective prompt: \n%s\n", retrieval_selective_prompt)
                    selected_indices = self.local.request_llm.get_llm_result(retrieval_selective_prompt)
                    logger.info("retrieval selective result: \n%s\n", selected_indices)
                    selected_indices = self.local.request_llm.parse_json_response(selected_indices, logger)["筛选索引"]
                    selected_indices = [i if isinstance(i, int) else eval(i) for i in selected_indices]
                    if len(selected_indices) > num_select:
                        selected_indices = selected_indices[:num_select]
                    else:
                        selected_indices.extend([idx for idx in list(range(num_select)) if idx not in selected_indices][:num_select - len(selected_indices)])
                    retrieved_results = [retrieved_results_raw[idx] for idx in selected_indices]
                    assert len(retrieved_results) == num_select
                    prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (
                    query, query, query, retrieved_results, attraction_extract_result)
                    logger.info("prompt: \n%s\n", prompt)
                    a_merge_result = self.local.request_llm.get_llm_result(prompt)
                    logger.info("attraction plan result: \n%s\n", a_merge_result)
                    a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger) 
                    with open(os.path.join(val_comp_dir, f"{query}_{self.prompt_type}.json"), 'w', encoding='utf-8') as f:
                        json.dump({
                                "compression": selected_indices, 
                                "result": a_merge_result
                            }, f, ensure_ascii=False, indent=4)
                    return None, query

                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw)])

                num_select = len(retrieved_results_raw) // 2
                retrieval_selective_prompt = str(self.local.config.retrieval_selective_objective_prompt) % (
                    query, num_select, query, attraction_extract_result, retrieved_results, num_select)
                logger.info("retrieval selective prompt: \n%s\n", retrieval_selective_prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = retrieval_selective_prompt
                    return None, query
                selected_indices = self.local.request_llm.get_llm_result(retrieval_selective_prompt)
                logger.info("retrieval selective result: \n%s\n", selected_indices)
                selected_indices = self.local.request_llm.parse_json_response(selected_indices, logger)["筛选索引"]
                selected_indices = [i if isinstance(i, int) else eval(i) for i in selected_indices]
                if len(selected_indices) > num_select:
                    selected_indices = selected_indices[:num_select]
                else:
                    selected_indices.extend([idx for idx in list(range(num_select)) if idx not in selected_indices][:num_select - len(selected_indices)])
                retrieved_results = [retrieved_results_raw[idx] for idx in selected_indices]
                assert len(retrieved_results) == num_select

                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (
                    query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger) 
            elif self.prompt_type == "given_direct_objective_retrieval_selective_one":
                if self.val_comp:
                    val_comp_dir = "validate_compression"
                    if not os.path.exists(val_comp_dir):
                        os.makedirs(val_comp_dir)
                    retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw)])
                    num_select = 1
                    retrieval_selective_prompt = str(self.local.config.retrieval_selective_objective_prompt) % (
                        query, num_select, query, attraction_extract_result, retrieved_results, num_select)
                    logger.info("retrieval selective prompt: \n%s\n", retrieval_selective_prompt)
                    selected_indices = self.local.request_llm.get_llm_result(retrieval_selective_prompt)
                    logger.info("retrieval selective result: \n%s\n", selected_indices)
                    selected_indices = self.local.request_llm.parse_json_response(selected_indices, logger)["筛选索引"]
                    if isinstance(selected_indices, list):
                        selected_indices = [i if isinstance(i, int) else eval(i) for i in selected_indices]
                    else:
                        if isinstance(selected_indices, str):
                            selected_indices = eval(selected_indices)
                        assert isinstance(selected_indices, int)
                        selected_indices = [selected_indices]
                    if len(selected_indices) > num_select:
                        selected_indices = selected_indices[:num_select]
                    else:
                        selected_indices.extend([idx for idx in list(range(num_select)) if idx not in selected_indices][:num_select - len(selected_indices)])
                    retrieved_results = [retrieved_results_raw[idx] for idx in selected_indices]
                    assert len(retrieved_results) == num_select
                    prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (
                        query, query, query, retrieved_results, attraction_extract_result)
                    logger.info("prompt: \n%s\n", prompt)
                    a_merge_result = self.local.request_llm.get_llm_result(prompt)
                    logger.info("attraction plan result: \n%s\n", a_merge_result)
                    a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger) 
                    with open(os.path.join(val_comp_dir, f"{query}_{self.prompt_type}.json"), 'w', encoding='utf-8') as f:
                        json.dump({
                                "compression": selected_indices, 
                                "result": a_merge_result
                            }, f, ensure_ascii=False, indent=4)
                    return None, query
                
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw)])

                num_select = 1
                retrieval_selective_prompt = str(self.local.config.retrieval_selective_objective_prompt) % (
                    query, num_select, query, attraction_extract_result, retrieved_results, num_select)
                logger.info("retrieval selective prompt: \n%s\n", retrieval_selective_prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = retrieval_selective_prompt
                    return None, query
                selected_indices = self.local.request_llm.get_llm_result(retrieval_selective_prompt)
                logger.info("retrieval selective result: \n%s\n", selected_indices)
                selected_indices = self.local.request_llm.parse_json_response(selected_indices, logger)["筛选索引"]
                if isinstance(selected_indices, list):
                    selected_indices = [i if isinstance(i, int) else eval(i) for i in selected_indices]
                else:
                    if isinstance(selected_indices, str):
                        selected_indices = eval(selected_indices)
                    assert isinstance(selected_indices, int)
                    selected_indices = [selected_indices]
                if len(selected_indices) > num_select:
                    selected_indices = selected_indices[:num_select]
                else:
                    selected_indices.extend([idx for idx in list(range(num_select)) if idx not in selected_indices][:num_select - len(selected_indices)])
                retrieved_results = [retrieved_results_raw[idx] for idx in selected_indices]
                assert len(retrieved_results) == num_select

                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (
                    query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger) 
            elif self.prompt_type == "given_direct_objective_retrieval_abstractive":
                if self.val_comp:
                    val_comp_dir = "validate_compression"
                    if not os.path.exists(val_comp_dir):
                        os.makedirs(val_comp_dir)
                    retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw)])
                    retrieval_abstractive_prompt = str(self.local.config.retrieval_abstractive_objective_validate_prompt) % (
                        query, query, attraction_extract_result, retrieved_results)
                    logger.info("retrieval abstractive prompt: \n%s\n", retrieval_abstractive_prompt)
                    retrieved_results = self.local.request_llm.get_llm_result(retrieval_abstractive_prompt)
                    logger.info("retrieval abstractive result: \n%s\n", retrieved_results)
                    retrieved_results = self.local.request_llm.parse_json_response(retrieved_results, logger)["总结结果"]
                    prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (
                        query, query, query, retrieved_results, attraction_extract_result)
                    logger.info("prompt: \n%s\n", prompt)
                    a_merge_result = self.local.request_llm.get_llm_result(prompt)
                    logger.info("attraction plan result: \n%s\n", a_merge_result)
                    a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)                     
                    with open(os.path.join(val_comp_dir, f"{query}_{self.prompt_type}.json"), 'w', encoding='utf-8') as f:
                        json.dump({
                                "compression": retrieved_results, 
                                "result": a_merge_result
                            }, f, ensure_ascii=False, indent=4)
                    return None, query
                    
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw)])

                retrieval_abstractive_prompt = str(self.local.config.retrieval_abstractive_objective_prompt) % (
                    query, query, attraction_extract_result, retrieved_results)
                logger.info("retrieval abstractive prompt: \n%s\n", retrieval_abstractive_prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = retrieval_abstractive_prompt
                    return None, query
                retrieved_results = self.local.request_llm.get_llm_result(retrieval_abstractive_prompt)
                logger.info("retrieval abstractive result: \n%s\n", retrieved_results)
                retrieved_results = self.local.request_llm.parse_json_response(retrieved_results, logger)["总结结果"]

                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (
                    query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger) 
            
            elif self.prompt_type == "given_direct_objective_retrieval_N2":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw[:2])])
                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)    
            elif self.prompt_type == "given_direct_objective_retrieval_N3":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw[:3])])
                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)       
            elif self.prompt_type == "given_direct_objective_retrieval_N5":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw[:5])])
                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)   
            elif self.prompt_type == "given_direct_objective_retrieval_N6":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw[:6])])
                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)  
            elif self.prompt_type == "given_direct_objective_retrieval_N7":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw[:7])])
                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)    
            
            elif self.prompt_type == "given_direct_objective_retrieval_N2_clean":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw_clean[:2])])
                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)    
            elif self.prompt_type == "given_direct_objective_retrieval_N3_clean":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw_clean[:3])])
                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)       
            elif self.prompt_type == "given_direct_objective_retrieval_N5_clean":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw_clean[:5])])
                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)   
            elif self.prompt_type == "given_direct_objective_retrieval_N6_clean":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw_clean[:6])])
                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)  
            elif self.prompt_type == "given_direct_objective_retrieval_N7_clean":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw_clean[:7])])
                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)  
            elif self.prompt_type == "given_direct_objective_retrieval_all_clean":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw_clean)])

                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            elif self.prompt_type == "given_direct_objective_retrieval_half_clean":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw_clean[:len(retrieved_results_raw_clean)//2])])

                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            elif self.prompt_type == "given_direct_objective_retrieval_one_clean":
                retrieved_results = retrieved_results_raw_clean[0]

                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = prompt
                    return None, query
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)
            elif self.prompt_type == "given_direct_objective_retrieval_selective_half_clean":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw_clean)])

                num_select = len(retrieved_results_raw_clean) // 2
                retrieval_selective_prompt = str(self.local.config.retrieval_selective_objective_prompt) % (
                    query, num_select, query, attraction_extract_result, retrieved_results, num_select)
                logger.info("retrieval selective prompt: \n%s\n", retrieval_selective_prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = retrieval_selective_prompt
                    return None, query
                selected_indices = self.local.request_llm.get_llm_result(retrieval_selective_prompt)
                logger.info("retrieval selective result: \n%s\n", selected_indices)
                selected_indices = self.local.request_llm.parse_json_response(selected_indices, logger)["筛选索引"]
                selected_indices = [i if isinstance(i, int) else eval(i) for i in selected_indices]
                if len(selected_indices) > num_select:
                    selected_indices = selected_indices[:num_select]
                else:
                    selected_indices.extend([idx for idx in list(range(num_select)) if idx not in selected_indices][:num_select - len(selected_indices)])
                retrieved_results = [retrieved_results_raw_clean[idx] for idx in selected_indices]
                assert len(retrieved_results) == num_select

                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (
                    query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger) 
            elif self.prompt_type == "given_direct_objective_retrieval_selective_one_clean":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw_clean)])

                num_select = 1
                retrieval_selective_prompt = str(self.local.config.retrieval_selective_objective_prompt) % (
                    query, num_select, query, attraction_extract_result, retrieved_results, num_select)
                logger.info("retrieval selective prompt: \n%s\n", retrieval_selective_prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = retrieval_selective_prompt
                    return None, query
                selected_indices = self.local.request_llm.get_llm_result(retrieval_selective_prompt)
                logger.info("retrieval selective result: \n%s\n", selected_indices)
                selected_indices = self.local.request_llm.parse_json_response(selected_indices, logger)["筛选索引"]
                if isinstance(selected_indices, list):
                    selected_indices = [i if isinstance(i, int) else eval(i) for i in selected_indices]
                else:
                    if isinstance(selected_indices, str):
                        selected_indices = eval(selected_indices)
                    assert isinstance(selected_indices, int)
                    selected_indices = [selected_indices]
                if len(selected_indices) > num_select:
                    selected_indices = selected_indices[:num_select]
                else:
                    selected_indices.extend([idx for idx in list(range(num_select)) if idx not in selected_indices][:num_select - len(selected_indices)])
                retrieved_results = [retrieved_results_raw_clean[idx] for idx in selected_indices]
                assert len(retrieved_results) == num_select

                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (
                    query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger) 
            elif self.prompt_type == "given_direct_objective_retrieval_abstractive_clean":
                retrieved_results = "\n".join([f"[{i}] {r}" for i, r in enumerate(retrieved_results_raw_clean)])

                retrieval_abstractive_prompt = str(self.local.config.retrieval_abstractive_objective_prompt) % (
                    query, query, attraction_extract_result, retrieved_results)
                logger.info("retrieval abstractive prompt: \n%s\n", retrieval_abstractive_prompt)
                if self.prompt_only:
                    if query not in self.prompt_dict:
                        self.prompt_dict[query] = {}
                    if self.prompt_type not in self.prompt_dict[query]:
                        self.prompt_dict[query][self.prompt_type] = {}
                    self.prompt_dict[query][self.prompt_type]["prompt"] = retrieval_abstractive_prompt
                    return None, query
                retrieved_results = self.local.request_llm.get_llm_result(retrieval_abstractive_prompt)
                logger.info("retrieval abstractive result: \n%s\n", retrieved_results)
                retrieved_results = self.local.request_llm.parse_json_response(retrieved_results, logger)["总结结果"]

                prompt = str(self.local.config.given_direct_objective_retrieval_prompt) % (
                    query, query, query, retrieved_results, attraction_extract_result)
                logger.info("prompt: \n%s\n", prompt)
                a_merge_result = self.local.request_llm.get_llm_result(prompt)
                logger.info("attraction plan result: \n%s\n", a_merge_result)
                a_merge_result = self.local.request_llm.parse_json_response(a_merge_result, logger)             
            else:
                raise ValueError(f"Unknown prompt type: {self.prompt_type}")
            logger.info(f"elapsed time: {time.time() - start_time}s")
            print(f"景点规划生成: {time.time() - start_time}s")

            if not isinstance(a_merge_result, dict):
                a_merge_result = None
            if a_merge_result is None:
                return None, query
            assert not self.val_comp
            
            plan_dir = f"../rebuttal/plan_data_{self.local.config.model}"
            if not os.path.exists(plan_dir):
                os.makedirs(plan_dir)
            plan_path = os.path.join(plan_dir, f"{query}-{self.prompt_type}.pkl")
            pickle.dump(a_merge_result, open(plan_path, 'wb'))
            # self.local.request_map.collect_dump()
            return None, query
        # except:
        #     return "fail", query

    def _fetch_list_results(self, prompt_list):

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.local.request_llm.fetch_content_result, prompt, index)
                       for index, prompt in enumerate(prompt_list)]
            result_list = [future.result() for future in concurrent.futures.as_completed(futures)]

        result_list = sorted(result_list, key=lambda x: x[1])
        return [result[0] for result in result_list]

    def _fetch_dict_results(self, prompt_dict):

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.local.request_llm.fetch_content_result, prompt, k)
                       for k, prompt in prompt_dict.items()]
            result_list = [future.result() for future in concurrent.futures.as_completed(futures)]

        result_dict = {}
        for r in result_list:
            result_dict[r[1]] = r[0]
        return result_dict

    def _fetch_dict_list_results(self, prompt_dict_list):

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(
                self.local.request_llm.fetch_content_dict_result, prompt, index, key)
                for index, prompt_dict in enumerate(prompt_dict_list) 
                for key, prompt in prompt_dict.items()]
            result_list = [future.result() for future in concurrent.futures.as_completed(futures)]

        result_list = sorted(result_list, key=lambda x: x[1])
        res_dict_list = []
        pos = 0
        for i in range(len(prompt_dict_list)):
            res_dict = {}
            for j in range(len(prompt_dict_list[i])):
                res_dict[result_list[pos + j][2]] = result_list[pos + j][0]
            res_dict_list.append(res_dict)
            pos += len(prompt_dict_list[i])

        return res_dict_list

    def _fetch_dict_dict_results(self, prompt_dict_dict):

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(
                self.local.request_llm.fetch_content_dict_result, prompt, key_1, key_2)
                for key_1, prompt_dict in prompt_dict_dict.items()
                for key_2, prompt in prompt_dict.items()]
            result_list = [future.result() for future in concurrent.futures.as_completed(futures)]

        res_dict_dict = {}
        for r in result_list:
            if r[1] not in res_dict_dict:
                res_dict_dict[r[1]] = {}
            res_dict_dict[r[1]][r[2]] = r[0]

        return res_dict_dict

    def _save_output(self, query, save_md, retriever_data, res_md, urls_list):
        """
            保存输出，包括markdown文件和json文件。
        如果输入的urls_list不为空，则会将图片下载到本地并添加到markdown中。
        
        Args:
            query (str): 查询语句。
            save_md (str): markdown格式的结果字符串。
            retriever_data (dict): 返回给retriever的数据字典，包括'title', 'content', 'url'三个键值对。
            res_md (str): 用来返回json的md。
            urls_list (List[str]): 图片url列表，可以为空。
        
        Returns:
            None: 无返回值，直接在函数内部将结果写入文件。
        """
        format_util = Format(query)
        query_path = query.replace('/', '').replace('\\', '')
        with open(f"{self.md_name}/{query_path}_{self.prompt_type}.md", "w", encoding="utf-8") as f:
            f.write(save_md)
        return_json = format_util.format_return_json(res_md, urls_list)
        with open(f"{self.dir_name}/{query_path}_{self.prompt_type}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(return_json, ensure_ascii=False))
        with open(f"{self.retriever_name}/{query_path}_{self.prompt_type}.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(retriever_data, ensure_ascii=False))

    @staticmethod
    def strip(query_list):
        """
        去除字符串列表中所有元素中的斜杠（/）、点号（.）和反斜杠（\\）。
        
        Args:
            query_list (list[str]): 包含字符串的列表。
        
        Returns:
            list[str]: 去除斜杠、点号和反斜杠后的字符串列表。
        
        """
        tmp_query_list = []
        for q in query_list:
            q = q.replace('/', '').replace('\\', '')
            tmp_query_list.append(q)
        return tmp_query_list

    def run(self, query_list, ready, prompt_type):
        """
            运行生成文章的主函数。
        Args:
            filepath (str, optional): 包含sug和change query的文件路径（默认为None）。
            ready (bool, optional): 是否使用已经准备好的数据（默认为False）。
        
        Returns:
            None.
        """
        start_time = time.time()
        self.prompt_type = prompt_type
        # if ready:
        #     print("sug and change query ready!!!")
        #     query_list, res_query_task_sug_dict, change_querys_dict = self._load_ready_data()
        # else:
        #     print("sug and change query not ready!!!")
        #     query_list, res_query_task_sug_dict, change_querys_dict = self._prepare_data(filepath)
        # # query_list = self.strip(query_list)
        # # 记录获取sug和change query花费时间
        # end_time = time.time()
        # print(f"sug and change query time: {end_time - start_time}")
        res_query_task_sug_dict, change_querys_dict = {}, {k: [] for k in query_list}
        self._process_queries(query_list, res_query_task_sug_dict, change_querys_dict)
        # 记录生成文章花费时间
        end_time = time.time()
        print(f"total time: {end_time - start_time}")

    def _load_ready_data(self):
        """
            加载已经运行完成的数据，包括查询列表、结果和换Q列表。
        返回值：三个元组（query_list, res_query_task_sug_dict, change_querys_dict），分别为：
            1. query_list (list of str) - 查询列表，每个元素是一个字符串；
            2. res_query_task_sug_dict (dict) - 结果字典，key为查询任务ID，value为一个字典，包含查询结果和相关信息；
            3. change_querys_dict (dict) - 换Q列表，key为查询任务ID，value为一个字典，包含换Q的原始查询和换Q后的查询。
        
        Args:
            None
        
        Returns:
            tuple (list of str, dict, dict):
                query_list (list of str) - 查询列表，每个元素是一个字符串；
                res_query_task_sug_dict (dict) - 结果字典，key为查询任务ID，value为一个字典，包含查询结果和相关信息；
                change_querys_dict (dict) - 换Q列表，key为查询任务ID，value为一个字典，包含换Q的原始查询和换Q后的查询。
        """
        query_list = []
        res_query_task_sug_dict = {}
        change_querys_dict = {}
        # 获取跑好的sug数据
        if self.config.need_sug:
            with codecs.open(f"{self.sug_cq_name}/sug.txt", "rb", "utf-8") as fin:
                for line in fin:
                    word = line.rstrip().split("\t")
                    if len(word) == 2:
                        res_query_task_sug_dict[word[0]] = json.loads(word[1])
        # 获取跑好的换q数据
        with codecs.open(f"{self.sug_cq_name}/change_query.txt", "rb", "utf-8") as fin:
            for line in fin:
                word = line.rstrip().split("\t")
                if len(word) == 2:
                    query_list.append(word[0])
                    change_querys_dict[word[0]] = json.loads(word[1])
        return query_list, res_query_task_sug_dict, change_querys_dict

    def _prepare_data(self, filepath):
        """
            将文件中的每一行数据转换为列表，并返回三个元组：
        1. 原始查询列表（query_list）
        2. 对应每个查询生成的任务和建议字典（res_query_task_sug_dict）
        3. 需要更改的查询列表（change_querys_dict）
        
        Args:
            filepath (str): 待处理的文件路径
        
        Returns:
            tuple(list, dict, dict):
                - query_list (list[str]): 原始查询列表
                - res_query_task_sug_dict (dict): 对应每个查询生成的任务和建议字典
                    key (str): 查询语句
                    value (tuple(list[str], list[str])): 任务列表和建议列表
                - change_querys_dict (dict): 需要更改的查询列表
                    key (str): 查询语句
                    value (bool): True表示需要更改，False表示不需要更改
        """
        query_list = []
        with codecs.open(filepath, "rb", "utf-8") as fin:
            for line in fin:
                query = line.strip()
                query_list.append(query)
        # res_query_task_sug_dict, change_querys_dict = self.spider_sug.get_sug_change_q_batch(query_list)
        # print(res_query_task_sug_dict, change_querys_dict)
        res_query_task_sug_dict, change_querys_dict = {}, {k: [] for k in query_list}
        return query_list, res_query_task_sug_dict, change_querys_dict

    def _process_queries(self, query_list, res_query_task_sug_dict, change_querys_dict):
        """
            并发处理查询语句，将每个查询语句转换为对应的结果。
        如果查询语句在change_querys_dict中存在，则使用该字典中的值进行替换。
        参数：
            query_list (List[str]): 需要处理的查询语句列表。
            res_query_task_sug_dict (Dict[str, Any]): 包含任务、建议和任务ID等信息的字典，键是任务ID，值是一个包含任务、建议和任务ID等信息的字典。
            change_querys_dict (Dict[str, str]): 包含需要替换的查询语句及其替换后的查询语句的字典，键是需要被替换的查询语句，值是替换后的查询语句。
        返回值：
            无返回值，直接修改res_query_task_sug_dict和change_querys_dict。
        """
        tmp_query_list = []
        for query in query_list:
            plan_path = os.path.join(f"plan_data_{self.base_model}", f"{query}-{self.prompt_type}.pkl")
            if os.path.exists(plan_path):
                plan = pickle.load(open(plan_path, "rb"))
                if plan is not None and isinstance(plan, dict) and not self.val_comp:
                    continue
            tmp_query_list.append(query)
        query_list = tmp_query_list
        print("# Queries: ", len(query_list))
        
        failure_list = []
        if self.multiprocess:
            idx = 0
            print(self.num_threads)
            while idx < len(query_list):
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    futures = [executor.submit(self.generate, query, res_query_task_sug_dict, change_querys_dict)
                            for query in query_list[idx: idx + self.num_threads]]
                    result_list = [future.result() for future in concurrent.futures.as_completed(futures)]
                for res in result_list:
                    if res[0] == "fail":
                        print(self.prompt_type, res[1], "failure")
                        failure_list.append(f"{self.prompt_type, res[1]}-{self.prompt_type, res[1]}")
                idx += self.num_threads
                
        else:
            for query in query_list:
                # print(self.prompt_type, query)
                res = self.generate(query, res_query_task_sug_dict, change_querys_dict)
                if res[0] == "fail":
                    print(self.prompt_type, res[1], "failure")
                    failure_list.append(f"{self.prompt_type, res[1]}-{self.prompt_type, res[1]}")
                # try:
                #     self.generate(query, res_query_task_sug_dict, change_querys_dict)
                # except:
                #     failure_list.append(query)
                # break
                exit(-1)
        print("==== failure list ===")
        print(failure_list, len(failure_list))

def load_queries(filepath):

    query_list = []
    with codecs.open(filepath, "rb", "utf-8") as fin:
        for line in fin:
            query = line.strip()
            query_list.append(query)

    return query_list

def main():
    """
    主函数，用于执行程序的主要逻辑。
    
    Returns:
        None
    
    Raises:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='test.txt')
    parser.add_argument('--ready', type=bool, default=False)
    parser.add_argument('--prompt_type', type=str, default='direct')
    parser.add_argument('--multiprocess', type=bool, default=False)
    parser.add_argument('--base_model', type=str, default='gpt-4o')
    parser.add_argument('--generate_prompts', type=bool, default=False)
    parser.add_argument('--num_threads', type=int, default=5)
    parser.add_argument('--base_url', type=str, default=None)
    parser.add_argument('--validate_compression', type=bool, default=False)
    args = parser.parse_args()

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
    else:
        raise ValueError("Invalid filepath")
    print("query总数量:", len(query_list))
    if args.base_model == "deepseek-r1" and args.generate_prompts:
        if os.path.exists("deepseek_r1_prompts.pkl"):
            prompt_dict = pickle.load(open("deepseek_r1_prompts.pkl", "rb"))
            print(len(prompt_dict), len(prompt_dict[list(prompt_dict.keys())[0]]))
            # with open('deepseek_r1_prompts.json', 'w', encoding='utf-8') as f:
            #     json.dump(prompt_dict, f, ensure_ascii=False, indent=4)
            prompt_list = [
                {
                    "prompt": vv["prompt"], 
                    "query": q, 
                    "method": m
                }
                for q, v in prompt_dict.items() for m, vv in v.items()
            ]
            with open('deepseek_r1_prompts_list.json', 'w', encoding='utf-8') as f:
                json.dump(prompt_list, f, ensure_ascii=False, indent=4)
        else:
            batch_strategy_driver = BatchStrategyDriver(args.multiprocess, args.base_model, prompt_only=True, num_threads=args.num_threads)
            method_list = ["given_direct_objective", 
                        "given_direct_objective_retrieval_all", "given_direct_objective_retrieval_half", "given_direct_objective_retrieval_one", 
                        "given_direct_objective_retrieval_selective_half", "given_direct_objective_retrieval_selective_one", "given_direct_objective_retrieval_abstractive"]
            for prompt_type in method_list:
                batch_strategy_driver.run(query_list, args.ready, prompt_type)
            pickle.dump(batch_strategy_driver.prompt_dict, open("deepseek_r1_prompts.pkl", "wb"))
    else:
        batch_strategy_driver = BatchStrategyDriver(args.multiprocess, args.base_model, base_url=args.base_url, num_threads=args.num_threads, val_comp=args.validate_compression)
        batch_strategy_driver.run(query_list, args.ready, args.prompt_type)

if __name__ == "__main__":
    main()
