import logging
import os, sys, json
import random
import numpy as np
import torch
import json_repair

from copy import deepcopy, copy

class Logger:
    def __init__(self, log_file='app.log'):
        # 创建日志记录器
        self.logger = logging.getLogger('ConversationAgentLogger')
        self.logger.setLevel(logging.DEBUG)
        self.log_file = log_file
        
        # 创建文件处理器
        if not os.path.exists(log_file):
            f = open(log_file, 'w')
            f.close()
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建格式器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加处理器到日志记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, msg):
        if isinstance(msg, list):
            msg = '\n'+'\n'.join([str(item) for item in msg])
        if isinstance(msg, dict):
            msg = '\n'+'\n'.join([f"{key}: {value}" for key, value in msg.items()])
        self.logger.info(msg)

    def info(self, msg):
        if isinstance(msg, list):
            msg = '\n'+'\n'.join([str(item) for item in msg])
        if isinstance(msg, dict):
            msg = '\n'+'\n'.join([f"{key}: {value}" for key, value in msg.items()])
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)
        
    def get_log_file(self):
        return self.log_file

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
import argparse
import yaml

def load_yaml_to_args(yaml_path, args):
    with open(yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Check for duplicate attributes
    duplicate_keys = set(yaml_data.keys()).intersection(vars(args).keys())
    if duplicate_keys:
        raise ValueError(f"Duplicate keys found: {duplicate_keys}")

    # Add YAML parameters to args
    for key, value in yaml_data.items():
        setattr(args, key, value)

def count_json_files(folder_path):
    try:
        # 列出文件夹中的所有文件
        files = os.listdir(folder_path)
        # 筛选出扩展名为 .json 的文件
        json_files = [f for f in files if f.endswith('.json')]
        # 返回 JSON 文件的数量
        return len(json_files)
    except FileNotFoundError:
        raise ValueError(f"Error: The folder '{folder_path}' does not exist.")
        return 0
    except Exception as e:
        raise ValueError(f"An unexpected error occurred: {e}")
        return 0
    
def set_determinitic_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.use_deterministic_algorithms(True)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	return


class dotdict(dict):
	def __getattr__(self, name):
		return self[name]


class hashabledict(dict):
	def __hash__(self):
		return hash(tuple(sorted(self.items())))


def get_formatted_conversation_history(history=None, role="System", roles=None, last_k=None):

    if roles is not None:
        sys_role = roles['System']
        user_role = roles['User']

    if len(history) == 0:
        return "[]"

    role_map = {
        "System": {sys_role: sys_role, user_role: 'User'},
        "User": {sys_role: 'User', user_role: user_role},
        "Critic": {sys_role: 'Alice', user_role: 'User'},
        "Reward": {sys_role: sys_role, user_role: user_role},
    }

    if role not in role_map:
        raise ValueError("Invalid role")

    transformed_data = []
    current_turn = {}

    if last_k is None:
        last_k = len(history)

    for item in history[-last_k:]:
        item_role = item['role']
        content = item['content']

        temp_role = role_map[role][item_role]
        current_turn[temp_role] = content

        if len(current_turn) == 2:
            transformed_data.append(current_turn)
            current_turn = {}

    # 处理剩余未完成的轮次
    if current_turn:
        transformed_data.append(current_turn)

    result = "[" + str(transformed_data) + "]"
    return result