"""
上下文模块，用于记录全流程中各个变量
"""
import time


class ArticleStruct:
    """
    记录文章结构相关数据
    """

    def __init__(self, query):
        """
        __init__
        """
        self.query = query
        self.prompt = ""
        self.llm_result = ""
        self.url_list = []


class ArticleContent:
    """
    记录文章内容相关数据
    """

    def __init__(self, query, title):
        """
        __init__
        """
        self.query = query
        self.title = title
        self.prompt = ""
        self.llm_result = ""
        self.url_list = []


class ArticleQueryAnalysis:
    """
    记录query分析相关数据
    """

    def __init__(self, query):
        """
        __init__
        """
        self.query = query
        self.prompt = ""
        self.llm_result = ""


class ArticleContext:
    """
    记录全流程中各个变量
    """

    def __init__(self, query):
        """
        __init__
        """
        self.query = query
        self.start_time = time.time()
        self.max_done_step = 0

        # 过程中的数据
        # query分析相关数据
        self.query_analysis = ArticleQueryAnalysis(query)

        # 结构相关数据
        self.change_query_list = []
        self.sug_list = []
        self.structure = ArticleStruct(query)
        self.structure_prompt_done_time = time.time()
        self.structure_done_time = time.time()
        self.scene_name = ""

        # 标题
        self.title_list = []
        self.title_w_query_list = []  # query, title, summary_content

        # 内容相关数据
        self.content_data = []
        self.url_list = []
        # 保留一份未处理的aiapi结果
        self.original_url_list = []
        self.content_done_time = time.time()

        # A页总结
        self.summary_part_prompt = []
        self.summary_part_result = []

        # A页标题改写
        self.title_rewrite_prompt = ""
        self.title_rewrite_result = ""

        # 聚合相关数据
        # 聚合prompt
        self.merge_prompt = ""
        # 聚合结果
        self.merge_result = ""
        self.merge_done_time = time.time()
        # 表格生成prompt
        self.table_gen_prompt = ""
        # 表格生成结果
        self.table_gen_result = ""
        self.table_gen_done_time = time.time()
        # 摘要prompt
        self.summary_prompt = ""
        # 摘要结果
        self.summary_result = ""
        self.summary_done_time = time.time()
        # 摘要拼接md
        self.summary_and_merge_md = ''

        # 格式化后md
        self.format_md = ''

        # 保存的md与json
        self.md_data = ""
        self.json_data = {}
