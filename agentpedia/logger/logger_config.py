"""
日志记录
"""
import logging
import datetime
import os


def setup_logging(query, log_dir_flag=None):
    """
    初始化日志记录功能。

    Args:
        query (str): 用于生成日志文件名的查询字符串。

    Returns:
        logging.Logger: 返回一个日志记录器对象，可以用于记录日志信息。
    """
    # 获取当前时间并格式化
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # 设置日志文件路径
    log_dir = "logs" if log_dir_flag is None else log_dir_flag
    log_filename = os.path.join(log_dir, f"{current_time}_{query}.log") if log_dir_flag is None else os.path.join(log_dir, f"{query}.log")

    # 如果日志目录不存在，则创建
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 创建一个日志记录器
    logger = logging.getLogger(query)
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 创建一个文件处理器
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)  # 设置文件处理器日志级别

    # 创建格式化器并添加到处理器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)

    return logger


# 单例模式，初始化一个query作key的全局日志记录器
logger_cache = {}

def get_logger(query=None, log_dir=None):
    """
    获取全局的日志记录器，如果没有初始化则创建一个。

    Args:
        query (str, optional): 用于设置日志记录器级别的查询字符串，默认为None。如果未提供，将引发ValueError异常。

    Returns:
        logging.Logger: 全局日志记录器对象。

    Raises:
        ValueError: 如果未提供查询字符串并且日志记录器尚未初始化。
    """
    global logger_cache
    query = query.replace('/', '').replace('\\', '')
    if query not in logger_cache:
        logger_cache[query] = setup_logging(query, log_dir)
    return logger_cache[query]