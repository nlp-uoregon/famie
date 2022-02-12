'''
Date: Feb 11, 2022
Mofied from:https://github.com/dataqa/dataqa/blob/master/src/dataqa/config/config_reader.py
'''
import configparser
from famie.constants import HOME, ROOT_PATH
from pathlib import Path

APP_CONFIG_FILE = str(Path(ROOT_PATH, "config/common_config.ini"))


def read_config(platform_config_path=None):
    config = configparser.ConfigParser()
    config.optionxform = str
    config["DEFAULT"]["home"] = HOME
    config["DEFAULT"]["root"] = ROOT_PATH
    config.read(APP_CONFIG_FILE)
    if platform_config_path is not None:
        config.read(platform_config_path)
    return config
