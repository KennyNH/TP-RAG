import os
import argparse
import json
import codecs
import concurrent.futures
import time
import random
import pprint
import threading
from copy import deepcopy
import numpy as np
from geopy.distance import geodesic
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from community import community_louvain
import networkx as nx
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing
# from vllm import LLM, SamplingParams
# from modelscope import snapshot_download

from agentpedia.utils.request_llm import RequestLLM
from agentpedia.logger.logger_config import get_logger
from agentpedia.prompt.plan_evaluator import Evaluator

random.seed(42)

class MultiAgentPlanner:

    def __init__(self, query, config, context, cache, num_threads):

        self.query = query
        self.config = config
        self.context = context
        self.cache = cache
        self.num_threads = num_threads
        self.logger = get_logger(query)
        
        self.request_llm = RequestLLM(config)
        self.cache_flag = False
        
        self.evaluator = Evaluator(query, llm_eval_model=self.config.model)
    
    def run_mas(self, poi_list, retrieved_plan_list, construct_data):

        planner_list = [{"retrieval": page, "idx": 0} for page in retrieved_plan_list]
        planner_list.append({"retrieval": "无参考，请自行规划", "idx": 0})

        global_poi_list = poi_list
        global_poi_dict = {p['名称']: p for p in global_poi_list}

        history_plan_records = {}
        global_memory = "暂无"
        
        # query analysis
        start_time = time.time()
        query_analysis_prompt = str(self.config.query_analysis_prompt) % (self.query)
        if self.cache_flag and self.cache.verify_data('query_analysis_result'):
            query_analysis_result = self.cache.load_data('query_analysis_result')
        else:
            query_analysis_result = self.request_llm.get_llm_result(query_analysis_prompt)
            self.cache.dump_data({'query_analysis_result': query_analysis_result})
        self.logger.info(f"query analysis prompt: \n{query_analysis_prompt}")
        self.logger.info(f"query analysis result: \n{query_analysis_result}")
        query_analysis_result = self.request_llm.parse_json_response(query_analysis_result, self.logger)
        assert query_analysis_result["query类型"] in ["通用", "个性化"] and "query要求" in query_analysis_result
        print(f"query分析: {time.time() - start_time}s")
        
        if query_analysis_result["query类型"] == "通用":
            plan_init_template = str(self.config.generic_plan_init_prompt)
        elif query_analysis_result["query类型"] == "个性化":
            plan_init_template = str(self.config.personal_plan_init_prompt)
        else:
            raise ValueError
        self.evaluator.personal_flag = query_analysis_result["query类型"] == "个性化"
        
        # initial planning
        start_time = time.time()
        if query_analysis_result["query类型"] == "通用":
            init_plan_prompt_list = [plan_init_template % (self.query, global_poi_dict, p["retrieval"]) for p in planner_list]
        elif query_analysis_result["query类型"] == "个性化":
            init_plan_prompt_list = [plan_init_template % (self.query, global_poi_dict, p["retrieval"], query_analysis_result["query要求"]) for p in planner_list]
        else:
            raise ValueError
        if self.cache_flag and self.cache.verify_data('init_plan_result_list'):
            init_plan_result_list = self.cache.load_data('init_plan_result_list')
        else:
            init_plan_result_list = self._fetch_list_results(init_plan_prompt_list)
            self.cache.dump_data({'init_plan_result_list': init_plan_result_list})
        for idx in range(len(init_plan_result_list)):
            self.logger.info(f"init prompt index:{idx}: \n{init_plan_prompt_list[idx]}")
            self.logger.info(f"init result index:{idx}: \n{init_plan_result_list[idx]}")            
            planner_list[idx]["plan"] = self.request_llm.parse_json_response(init_plan_result_list[idx], self.logger)
        print(f"初始规划: {time.time() - start_time}s")
        # self.cache_flag = False
        
        # self.request_llm.config.temperature = 0.4

        for round_idx in range(self.config.num_rounds):
            
            self.request_llm.config.temperature = min(0.7, self.request_llm.config.temperature * 1.1)
            
            start_time = time.time()
            # evaluate & rank
            if self.cache_flag and self.cache.verify_data(f"evaluation {round_idx}"):
                sorted_planner_list, sorted_eval_text_list, evaluation_protocol = self.cache.load_data(f"evaluation {round_idx}")
            else:
                sorted_planner_list, sorted_eval_text_list, evaluation_protocol = self.evaluator.eval(planner_list, construct_data)
                self.cache.dump_data({
                    f"evaluation {round_idx}": (sorted_planner_list, sorted_eval_text_list, evaluation_protocol)
                })
            sorted_plan_eval_results = '\n'.join([f"({idx + 1}) {planner['plan']}\n---\n{sorted_eval_text_list[idx]}" for idx, planner in enumerate(sorted_planner_list)])
            
            # reflection
            reflection_prompt = str(self.config.plan_evaluation_reflect_prompt) % (self.query, global_poi_dict, evaluation_protocol, sorted_plan_eval_results, global_memory)
            self.logger.info(f"reflect prompt index:{round_idx}: \n{reflection_prompt}")
            if self.cache_flag and self.cache.verify_data(f'global_memory {round_idx}'):
                global_memory = self.cache.load_data(f'global_memory {round_idx}')
            else:
                global_memory = self.request_llm.get_llm_result(reflection_prompt)
                self.cache.dump_data({f'global_memory {round_idx}': global_memory})
            self.logger.info(f"reflect result index:{round_idx}: \n{global_memory}")
            
            history_plan_records[round_idx] = {
                "plan_list": sorted_planner_list, 
                "reflection": global_memory, 
            }
            print(f"评估&反思{round_idx + 1}: {time.time() - start_time}s")
            
            # select & crossover & mutation
            start_time = time.time()
            ## mutation only
            num_only_mutation = int(len(sorted_planner_list) * 0.5)
            only_mutation_planner_list = sorted_planner_list[:num_only_mutation - 1]
            if "无参考，请自行规划" in [planner["retrieval"] for planner in only_mutation_planner_list]:
                only_mutation_planner_list.append(sorted_planner_list[num_only_mutation - 1])
            else:
                for planner in sorted_planner_list[num_only_mutation - 1:]:
                    if planner["retrieval"] == "无参考，请自行规划":
                        only_mutation_planner_list.append(planner)
                        break
            assert len(only_mutation_planner_list) == num_only_mutation
            only_mutation_sorted_planner_list, only_mutation_sorted_eval_text_list = self.evaluator.rank(only_mutation_planner_list)
            only_mutation_plan_eval_results = '\n'.join([f"({idx + 1}) {planner['plan']}\n---\n{only_mutation_sorted_eval_text_list[idx]}" 
                                                         for idx, planner in enumerate(only_mutation_sorted_planner_list)])
            mutation_only_prompt = str(self.config.plan_mutation_only_prompt) % (self.query, global_poi_dict, evaluation_protocol, 
                                                                               only_mutation_plan_eval_results, global_memory, num_only_mutation, num_only_mutation)
            self.logger.info(f"mutation only prompt index:{round_idx}: \n{mutation_only_prompt}")
            if self.cache_flag and self.cache.verify_data(f'mutation_only_result {round_idx}'):
                mutation_only_result = self.cache.load_data(f'mutation_only_result {round_idx}')
            else:
                mutation_only_result = self.request_llm.get_llm_result(mutation_only_prompt)
                self.cache.dump_data({f'mutation_only_result {round_idx}': mutation_only_result})
            self.logger.info(f"mutation only result index:{round_idx}: \n{mutation_only_result}")
            mutation_only_result = self.request_llm.parse_json_response(mutation_only_result, self.logger, parse_list=True)
            assert len(mutation_only_result) == num_only_mutation
            mutation_only_result = [
                {"retrieval": "无参考，请自行规划" if only_mutation_planner_list[idx]["retrieval"] == "无参考，请自行规划" else None, 
                 "plan": plan, "idx": round_idx + 1} for idx, plan in enumerate(mutation_only_result)]
            ## crossover & mutation
            num_crossover_mutation = len(sorted_planner_list) - num_only_mutation
            crossover_mutation_prompt = str(self.config.plan_crossover_mutation_prompt) % (self.query, global_poi_dict, evaluation_protocol, 
                                                                               sorted_plan_eval_results, global_memory, num_crossover_mutation, num_crossover_mutation, 
                                                                               num_crossover_mutation)
            self.logger.info(f"crossover mutation prompt index:{round_idx}: \n{crossover_mutation_prompt}")
            if self.cache_flag and self.cache.verify_data(f'crossover_mutation_result {round_idx}'):
                crossover_mutation_result = self.cache.load_data(f'crossover_mutation_result {round_idx}')
            else:
                crossover_mutation_result = self.request_llm.get_llm_result(crossover_mutation_prompt)
                self.cache.dump_data({f'crossover_mutation_result {round_idx}': crossover_mutation_result})
            self.logger.info(f"crossover mutation result index:{round_idx}: \n{crossover_mutation_result}")
            crossover_mutation_result = self.request_llm.parse_json_response(crossover_mutation_result, self.logger, parse_list=True)
            assert len(crossover_mutation_result) == num_crossover_mutation
            crossover_mutation_result = [{"retrieval": None, "plan": plan, "idx": round_idx + 1} for plan in crossover_mutation_result]
            planner_list = mutation_only_result + crossover_mutation_result
            print(f"规划完善{round_idx + 1}: {time.time() - start_time}s")
        
        start_time = time.time()
        # evaluate & rank
        if self.cache_flag and self.cache.verify_data(f"evaluation {self.config.num_rounds}"):
            sorted_planner_list, sorted_eval_text_list, evaluation_protocol = self.cache.load_data(f"evaluation {self.config.num_rounds}")
        else:
            sorted_planner_list, sorted_eval_text_list, evaluation_protocol = self.evaluator.eval(planner_list, construct_data)
            self.cache.dump_data({
                f"evaluation {self.config.num_rounds}": (sorted_planner_list, sorted_eval_text_list, evaluation_protocol)
            })
        sorted_plan_eval_results = '\n'.join([f"({idx + 1}) {planner['plan']}\n---\n{sorted_eval_text_list[idx]}" for idx, planner in enumerate(sorted_planner_list)])
        
        # reflection
        reflection_prompt = str(self.config.plan_evaluation_reflect_prompt) % (self.query, global_poi_dict, evaluation_protocol, sorted_plan_eval_results, global_memory)
        self.logger.info(f"reflect prompt index:{self.config.num_rounds}: \n{reflection_prompt}")
        if self.cache_flag and self.cache.verify_data(f'global_memory {self.config.num_rounds}'):
            global_memory = self.cache.load_data(f'global_memory {self.config.num_rounds}')
        else:
            global_memory = self.request_llm.get_llm_result(reflection_prompt)
            self.cache.dump_data({f'global_memory {self.config.num_rounds}': global_memory})
        self.logger.info(f"reflect result index:{self.config.num_rounds}: \n{global_memory}")
        
        history_plan_records[self.config.num_rounds] = {
            "plan_list": sorted_planner_list, 
            "reflection": global_memory, 
        }
        
        # find the optima
        sorted_planner_list, sorted_eval_text_list = self.evaluator.rank([v["plan_list"][0] for k, v in history_plan_records.items()])
        optimal_plan = sorted_planner_list[0]
        print(f"最终评估&选择: {time.time() - start_time}s")
        # print(optimal_plan["idx"], optimal_plan["retrieval"], sorted_planner_list[1]["retrieval"])
        # print(sorted_eval_text_list)
        print(optimal_plan["idx"], optimal_plan["retrieval"])
            
        return {
            "optimal_plan": optimal_plan, 
            "optimal_plan_list_ranking": sorted_planner_list, 
            "history_plan": history_plan_records,
        }

    def _fetch_list_results(self, prompt_list):
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.request_llm.fetch_content_result, prompt, index)
                    for index, prompt in enumerate(prompt_list)]
            result_list = [future.result() for future in concurrent.futures.as_completed(futures)]
        result_list = sorted(result_list, key=lambda x: x[1])
        
        return [result[0] for result in result_list]
            
    def calculate_distance(self, p_1, p_2):
        return geodesic((p_1["纬度"], p_1["经度"]), 
                        (p_2["纬度"], p_2["经度"])).m

    def calculate_optimal_distance(self, dist_mat):
        if len(dist_mat) > 15:
            permutation_1, _ = solve_tsp_simulated_annealing(dist_mat)
            permutation, optimal_dist = solve_tsp_local_search(dist_mat, x0=permutation_1)
        else:
            permutation, optimal_dist = solve_tsp_dynamic_programming(dist_mat)
        
        return permutation, optimal_dist

    def optimal_spatial_calculate(self, result, poi_gt_dict):
        
        # start_time = time.time()
        
        poi_list = [p for sub_title, p_l in result.items() for p in p_l]
        poi_detail_list = [poi_gt_dict[p["名称"]] if p["名称"] in poi_gt_dict else None for p in poi_list]
        num_total = len(poi_detail_list)
        poi_detail_list = [p for p in poi_detail_list if p is not None]
        if len(poi_detail_list) <= 1:
            return result, '无', -1, -1
        
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
        global_permutation, global_optimal_dist = None, None
        for i in range(len(poi_detail_list)):
            dist_mat = deepcopy(dist_mat_ori)
            dist_mat[:, i] = 0.
            permutation, optimal_dist = self.calculate_optimal_distance(dist_mat)
            if global_optimal_dist is None or optimal_dist < global_optimal_dist:
                global_permutation = permutation
                global_optimal_dist = optimal_dist
        assert int(global_optimal_dist) <= int(cur_dist)   
        optimal_poi_list = [poi_detail_list[i]["名称"] for i in global_permutation] 
        # print("optimal route calculate time cost:", time.time() - start_time)

        return result, optimal_poi_list, cur_dist, global_optimal_dist

    def make_spatial_inputs(self, global_poi_dict):
        
        poi_info = {p: {
            "地址": global_poi_dict[p]["地址"] if "地址" in global_poi_dict[p] else None,
            "纬度": global_poi_dict[p]["纬度"] if "纬度" in global_poi_dict[p] else None,
            "经度": global_poi_dict[p]["经度"] if "经度" in global_poi_dict[p] else None,
        } for p in global_poi_dict}

        return poi_info        

    def make_temporal_inputs(self, global_poi_dict):

        poi_info = {p: {
            "推荐游玩开始时间": global_poi_dict[p]["推荐游玩开始时间"] if "推荐游玩开始时间" in global_poi_dict[p] else None,
            "预计游玩时长": global_poi_dict[p]["预计游玩时长"] if "预计游玩时长" in global_poi_dict[p] else None,
            "开放时间": global_poi_dict[p]["开放时间"] if "开放时间" in global_poi_dict[p] else None,
        } for p in global_poi_dict}

        return poi_info

    def make_poi_inputs(self, global_poi_dict):

        poi_info = {p: {
            "描述": global_poi_dict[p]["描述"] if "描述" in global_poi_dict[p] else None,
        } for p in global_poi_dict}

        return poi_info
    
    def make_itinerary_inputs(self, global_poi_dict):

        poi_info = [p for p in global_poi_dict]

        return poi_info

    def make_refine_inputs(self, feedback):

        comments = ""
        text = {
            "spatial": "空间合理性", 
            "temporal": "时间合理性", 
            "poi": "POI质量", 
            "itinerary": "行程总质量",
        }
        for k, v in feedback.items():
            tmp = '\n'.join(v)
            comments += f"{text[k]}:\n {tmp}\n"

        return comments

    def jaccard_similarity(self, A, B):

        A = set([p["名称"] for _, p_l in A.items() for p in p_l])
        B = set([p["名称"] for _, p_l in B.items() for p in p_l])

        M_11 = len(A & B)
        M_10 = len(A.difference(B))
        M_01 = len(B.difference(A))

        if len(A) + len(B) == 0:
            s = 0.
        s = M_11 / (M_11 + M_10 + M_01)

        return s

    def get_k_min_edges(self, sim_mat, K):

        triu_indices = np.triu_indices_from(sim_mat, k=1)
        edge_values = sim_mat[triu_indices]
        edge_indices = list(zip(triu_indices[0], triu_indices[1]))
        sorted_edges = sorted(zip(edge_values, edge_indices), key=lambda x: x[0])
        k_min_edges = sorted_edges[:K]

        return [([edge[1][0], edge[1][1]], edge[0]) for edge in k_min_edges]
