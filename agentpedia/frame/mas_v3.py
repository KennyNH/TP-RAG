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
from agentpedia.prompt.attraction_planner_v3 import MultiAgentPlanner
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

    def __init__(self, multiprocess, base_model, num_threads):
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
        try:
        # if 2 > 1:
            _MODEL = self.base_model
            plan_dir = f"plan_data_mas_{_MODEL}"
            if not os.path.exists(plan_dir):
                os.makedirs(plan_dir)
            plan_path = os.path.join(plan_dir, f"{query}-{self.prompt_type}.pkl")
            if os.path.exists(plan_path):
                plan = pickle.load(open(plan_path, "rb"))
                # print(plan)
                # return
                if plan is not None and isinstance(plan, dict):
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
                self.local.config._load_file_config('agentpedia/config/category_config/travel_plan_mas_config_v3.yaml'))

            self.local.config.model = _MODEL
            self.local.cache = Cache(f"cache_data_{_MODEL}", query + self.prompt_type)
            # self.local.request_llm = RequestLLM(self.local.config)
            # self.local.request_map = RequestMap(self.local.config, self.multiprocess)
            # self.local.dqa_mrc = DqaMrc(self.local.config)
            # generate_article = ArticleGenerator(query, self.local.config, context)
            self.local.multi_agent_attraction_plan = MultiAgentPlanner(query, self.local.config, context, 
                                                    self.local.cache, self.num_threads)
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
            if self.prompt_type == "evolutionary_optimize":
                a_merge_result = self.local.multi_agent_attraction_plan.run_mas(attraction_extract_result, retrieved_results_raw, construct_data)
            else:
                raise ValueError(f"Unknown prompt type: {self.prompt_type}")
            logger.info(f"elapsed time: {time.time() - start_time}s")
            print(f"景点规划生成: {time.time() - start_time}s")

            if not isinstance(a_merge_result, dict):
                a_merge_result = None

            if a_merge_result is None:
                return None, query

            plan_path = os.path.join(plan_dir, f"{query}-{self.prompt_type}.pkl")
            pickle.dump(a_merge_result, open(plan_path, 'wb'))
            # self.local.request_map.collect_dump()
            return None, query
        except:
            return "fail", query

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
            plan_path = os.path.join(f"plan_data_mas_{self.base_model}", f"{query}-{self.prompt_type}.pkl")
            if os.path.exists(plan_path):
                plan = pickle.load(open(plan_path, "rb"))
                if plan is not None and isinstance(plan, dict):
                    try:
                        assert len(plan["history_plan"]) == 4
                    except:
                        print(len(plan["history_plan"]))
                        raise ValueError
                    continue
            tmp_query_list.append(query)
        query_list = tmp_query_list
        print("Remaining queries num: ", len(query_list))
        # exit(-1)

        failure_list = []
        if self.multiprocess:
            idx = 0
            batch_size = 5
            while idx < len(query_list):
                with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                    futures = [executor.submit(self.generate, query, res_query_task_sug_dict, change_querys_dict)
                            for query in query_list[idx: idx + batch_size]]
                    result_list = [future.result() for future in concurrent.futures.as_completed(futures)]
                for res in result_list:
                    if res[0] == "fail":
                        print(self.prompt_type, res[1], "failure")
                        failure_list.append(f"{self.prompt_type, res[1]}-{self.prompt_type, res[1]}")
                idx += batch_size
        else:
            failure_list = []
            for query in query_list:
                # print(self.prompt_type, query)
                res = self.generate(query, res_query_task_sug_dict, change_querys_dict)
                if res == "fail":
                    failure_list.append(query)
                # break
        print("==== failure list ===")
        print(failure_list)

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
    parser.add_argument('--prompt_type', type=str, default='mas')
    parser.add_argument('--multiprocess', type=bool, default=False)
    parser.add_argument('--base_model', type=str, default='gpt-4o')
    parser.add_argument('--num_threads', type=int, default=25)
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
    batch_strategy_driver = BatchStrategyDriver(args.multiprocess, args.base_model, args.num_threads)
    batch_strategy_driver.run(query_list, args.ready, args.prompt_type)

if __name__ == "__main__":
    main()
