"""
本模块提供了一个应用程序配置管理工具，用于通过YAML文件加载、访问和操作配置设置。
定义了`Config`类，该类允许从指定的YAML文件中轻松加载配置设置，并提供了一种接口，
以属性和字典键的形式访问这些设置。此外，它还允许在运行时动态设置和修改配置。
"""

import re
import os
import yaml


class Config:
    """
    用于管理从YAML文件加载的应用程序配置设置的类。

    此类封装了从指定YAML文件加载配置设置的功能，并提供了一个接口，以属性和字典键的形式访问这些设置。
    它还允许在运行时动态修改配置。

    属性:
        yaml_loader (yaml.FullLoader): 用于解析YAML文件的加载器。
        internal_config (dict): 存储初始加载配置的字典。
        final_config (dict): 存储运行时可能修改的当前配置状态的字典。
    """

    def __init__(self):
        """
            初始化函数，用于初始化YamlConfig类的对象。
        该函数会创建一个YamlLoader对象，并将其作为属性保存在YamlConfig类的对象中。
        同时，它还会获取内部配置信息，并将其作为属性保存在YamlConfig类的对象中。
        最后，它会设置额外的关键字，以便在后续处理中使用。
        
        Args:
            无参数，不需要传入任何参数。
        
        Returns:
            无返回值，只是初始化了YamlConfig类的对象。
        """
        self.yaml_loader = self._build_yaml_loader()
        self.internal_config = self._get_internal_config()
        self.final_config = self.internal_config
        self._set_additional_key()

    @staticmethod
    def _build_yaml_loader():
        """
        构建一个能够解析各种格式浮点数的YAML加载器。

        此方法修改了PyYAML包中的FullLoader，添加了对科学记数法和其他常见的浮点数表示法的解析支持。

        返回:
            yaml.FullLoader: 可用于加载YAML文件的加载器类。
        """
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\.(?:inf|Inf|INF)
                |\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        return loader

    def _get_internal_config(self):
        """
        从预定义的YAML文件中检索初始配置设置。

        该方法从与此脚本位于同一目录的'basic_config.yaml'加载配置，并以字典形式返回设置。

        返回:
            dict: 包含加载的配置设置的字典。
        """
        current_path = os.path.dirname(os.path.realpath(__file__))
        init_config_path = os.path.join(current_path, "basic_config.yaml")
        article_prompt_path = os.path.join(current_path, "article_prompt.yaml")
        switch_path = os.path.join(current_path, "switch.yaml")
        return self._load_configs([init_config_path, article_prompt_path, switch_path])

    def _load_configs(self, file_paths):
        """
        加载给定文件路径的配置文件，并将其合并到config字典中。

        :param file_paths: 要加载的配置文件的路径
        :type file_paths: str

        对于给定的每个文件路径，此方法都会打开文件，
        并使用_load_file_config方法将其内容解析为字典。

        然后，它将使用字典的update方法将新加载的配置合并到config字典中。
        注意，如果多个文件包含相同的键，那么最后加载的文件中的值将覆盖先前的值。
        """
        config = {}
        for file_path in file_paths:
            config.update(self._load_file_config(file_path))
        return config
            
    def _load_file_config(self, config_file_path):
        """
        从指定的文件路径加载YAML配置数据。

        使用此类定义的加载器读取YAML文件，并将内容作为字典返回。如果文件未找到或读取过程中出现任何错误，则返回空字典。

        参数:
            config_file_path (str): YAML配置文件的文件路径。

        返回:
            dict: 配置数据的字典。如果文件读取失败，则返回空字典。
        """
        with open(config_file_path, "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=self.yaml_loader) or {}

    def _set_additional_key(self):
        """
            设置额外的密钥，这个函数是空的，不做任何事情。
        
        Args:
            无参数需要传入。
        
        Returns:
            None, 该函数没有返回值。
        """
        # This function is intentionally left blank
        pass

    def __setitem__(self, key, value):
        """
        将配置值索引由`key`设置为`value`。

        此方法允许在运行时修改配置设置。如果`key`不是字符串，则会引发TypeError异常。

        参数:
            key (str): 存储值的配置键。
            value (any): 要存储的值。

        异常:
            TypeError: 如果键不是字符串。
        """
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config[key] = value

    def __getattr__(self, item):
        """
        以属性方式访问配置键时返回键对应的值。

        此方法提供对配置键的动态访问。如果键不存在，则引发AttributeError异常。

        参数:
            item (str): 要访问的配置键名。

        异常:
            AttributeError: 如果键不存在于配置中。
        """
        if "final_config" not in self.__dict__:
            raise AttributeError("'Config' object has no attribute 'final_config'")
        if item in self.final_config:
            return self.final_config[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __getitem__(self, item):
        """
        从配置字典中检索指定键的值。

        提供类似字典的访问配置设置的方式。如果`item`在配置中不存在，则返回None。

        参数:
            item (str): 要检索的配置键。

        返回:
            any: 与`item`关联的值，如果键不存在则返回None。
        """
        return self.final_config.get(item)

    def __contains__(self, key):
        """
        检查配置中是否存在给定的键。

        此方法允许检查配置字典中是否存在键，并返回True如果键存在且其值不为None。

        参数:
            key (str): 要检查的配置键。

        返回:
            bool: 如果`key`存在且值不为None，则返回True；否则返回False。

        异常:
            TypeError: 如果`key`不是字符串。
        """
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config

    def __repr__(self):
        """
        提供对象的字符串表示形式。

        此方法返回准确表示`final_config`字典当前状态的字符串。用于调试和记录。

        返回:
            str: 表示配置的字符串。
        """
        return str(self.final_config)
