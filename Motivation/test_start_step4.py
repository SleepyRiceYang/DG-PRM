import json
import requests
import os
import sys
import argparse
import datetime
import re
import math
import random
import pytz

from tqdm import tqdm
from json_repair import repair_json

# ==================== 路径与导入设置 ====================
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
grandparent_dir = os.path.dirname(os.path.dirname(parent_dir))
sys.path.insert(0, grandparent_dir)

# 假设这些工具函数依然可用
from utils.Prompt import get_user_messages, human_persuader_strategy_instruction_map, get_message_prompt_func, get_message_prompt_func_change
from utils.api_call import call_llm_chat_api_openai 

# API 地址
API_URL = "http://127.0.0.1:8092/inference"
API_URL_2 = "http://127.0.0.1:8091/inference"

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "1"):
        return True
    elif value.lower() in ("false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

# ==================== 日志工具 ====================
class PrintLogger:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'

    @staticmethod
    def print_info(message):
        print(f"{PrintLogger.GREEN}[INFO] {message}{PrintLogger.ENDC}")
        print("-" * 30)
    @staticmethod
    def print_step(title, content="-"*30):
        print(f"\n{PrintLogger.HEADER}【STEP: {title}】{PrintLogger.ENDC}")
        if isinstance(content, (dict, list)):
            print(json.dumps(content, ensure_ascii=False, indent=2))
        else:
            print(str(content).strip())
        print("-" * 60)

    @staticmethod
    def print_prompt(messages):
        print(f"{PrintLogger.BLUE}[Constructed Prompt]{PrintLogger.ENDC}")
        for msg in messages:
            role = msg['role'].upper()
            content = msg['content']
            print(f"[{role}]: {content}")
        print("-" * 30)

# ==================== 基础工具函数 ====================

def print_list(history):
    for item in history:
        print(f"{item['role']}: {item['content']}")

def call_api(payload: dict):
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        if 'response' in locals() and response is not None:
            print(f"🔴 服务器返回的详细错误信息: {response.text}")
        return None

def calculate_entropy_and_topk(logprobs_data):
    """处理 API 返回的 logprobs 数据"""
    token_metrics = []
    if not logprobs_data:
        return []

    for item in logprobs_data:
        top_k_dict = item.get('top_logprobs', {})
        if isinstance(top_k_dict, list):
            temp_dict = {k['token']: k['logprob'] for k in top_k_dict}
            top_k_dict = temp_dict

        entropy = 0.0
        top_5_list = []
        for token_str, logprob in top_k_dict.items():
            prob = math.exp(logprob)
            top_5_list.append({"token": token_str, "prob": prob})
            if prob > 0:
                entropy -= prob * logprob 

        top_5_list.sort(key=lambda x: x['prob'], reverse=True)
        top_5_list = top_5_list[:5]

        token_metrics.append({
            "token": item.get('token', ''),
            "entropy": entropy,
            "top_5": top_5_list
        })
    return token_metrics

def formatted_strateies(strategies):
    # 返回纯策略名列表供模型选择
    return '\n'.join([f"- {key}" for key in strategies.keys()])

def formatted_dialog_history(dialog_history):
    formatted_str = ""
    for turn in dialog_history:
        role = "You" if turn['role'] == "Persuader" else "Persuadee"
        if 'strategy' in turn and turn['role'] == "Persuader":
            content = f"[{turn['strategy']}] {turn['content']}"
        else:
            content = turn['content']
        formatted_str += f'"{role}": {content}\n'
    # PrintLogger.print_step("Dialog History", dialog_history)
    # PrintLogger.print_step("Formatted Dialog History", formatted_str)
    return formatted_str

def parse_strategy_content(text):
    """
    专门解析格式为 "[Strategy Name]:[Brief Reason]" 的字符串
    返回: (name, reason) 元组
    """
    if not text: 
        return "", ""
    
    text = text.strip()
    
    # 处理多种可能的格式
    # 格式1: [Strategy Name]: Reason
    pattern1 = r"^\s*(?:\[)?([^\]:：]+)(?:\])?\s*[:：]\s*(.+)$"
    match1 = re.match(pattern1, text, re.DOTALL)
    
    if match1:
        name = match1.group(1).strip()
        reason = match1.group(2).strip()
        return name, reason
    
    # 格式2: Strategy Name: Reason
    pattern2 = r"^\s*([^:：\n]+)\s*[:：]\s*(.+)$"
    match2 = re.match(pattern2, text, re.DOTALL)
    
    if match2:
        name = match2.group(1).strip()
        reason = match2.group(2).strip()
        # 清理可能的多余字符
        name = name.strip("[]\"'")
        return name, reason
    
    # 格式3: 只有策略名称
    if text and not re.search(r'[:：]', text):
        # 清理可能的标记字符
        clean_name = text.strip("[]\"'* \n\t")
        return clean_name, ""
    
    # 兜底处理：尝试分离名称和原因
    parts = re.split(r'[:：]', text, 1)
    if len(parts) == 2:
        name = parts[0].strip("[]\"'* \n\t")
        reason = parts[1].strip()
        return name, reason
    else:
        # 最后的兜底：整个文本作为策略名称
        print(f"{PrintLogger.YELLOW}[Parser Warning] Strategy format mismatch. Raw: {text}{PrintLogger.ENDC}")
        return text.strip(" []\"'"), ""

# ==================== 核心生成逻辑 ====================

def generate_system_response(env, chat_history, strategies_map, 
                sys_llm_model_name, use_local_api=True,
                order_change=False, temperature=0.0, enable_thinking=False,
                forbidden_strategies=None):
    """
    一步生成：State Analysis -> Strategy -> Response
    并精确解析 Strategy Name 和 Reason
    """
    strategy_set_str = formatted_strateies(strategies_map)
    history_str = formatted_dialog_history(chat_history)

    infos = {
        'strategy_set': strategy_set_str
    }

    if not order_change:
        messages = get_message_prompt_func(env, "Persuader", infos, conversation=history_str)
    else:
        messages = get_message_prompt_func_change(env, "Persuader", infos, conversation=history_str)

    if forbidden_strategies:
        forbidden_str = ", ".join([f"'{s}'" for s in forbidden_strategies])
        constraint_content = (
            f"\n\n[Constraint]: To explore different possibilities, you MUST NOT use the following strategies: {forbidden_str}. "
            f"Please choose a distinct valid strategy from the strategy set."
        )
        messages.append({"role": "system", "content": constraint_content})

    PrintLogger.print_step("Unified Generation Prompt", "Prompt Constructed")
    PrintLogger.print_prompt(messages)

    if use_local_api: # 默认使用本地模型
        generation_payload = {
            "task_type": "generate",
            "messages": messages,
            "temperature": temperature, 
            "n": 1,
            "max_tokens": 10240,
            "logprobs": True,
            "top_logprobs": 5,
            "enable_thinking": enable_thinking
        }

        print(f"{PrintLogger.BLUE}... Calling Local Model API  ...{PrintLogger.ENDC}")
        result = call_api(generation_payload)
    else:
        # 需要调用远程 API
        generation_payload = {
            "model_name": sys_llm_model_name,
            "messages": messages,
            "temperature": 0.0,
            "n": 1,
            "max_tokens": 10240
        }
        local_api_url = "http://localhost:8093/inference"
        generation_payload['use_local_api'] = use_local_api
        generation_payload['local_api_url'] = local_api_url
        result = call_llm_chat_api_openai(**generation_payload)


    if result and "generated_texts" in result:
        full_text = result["generated_texts"][0].strip()

        # 交换顺序后的输出最后末尾部分不会输出</state_analysis>标签，进行处理
        if order_change and not full_text.endswith("</state_analysis>"):
            full_text += "</state_analysis>"

        print(f"{PrintLogger.GREEN}[Raw Model Output]:\n{full_text}{PrintLogger.ENDC}\n")

        # 处理 logprobs
        raw_logprobs = []
        if 'details' in result and isinstance(result['details'], list):
            raw_logprobs = result['details']
        elif 'choices' in result:
            raw_logprobs = result['choices'][0]['logprobs']['content']
        token_metrics = calculate_entropy_and_topk(raw_logprobs)

    elif result and not use_local_api:
        full_text = result[0]
        token_metrics = None

    else:
        raise Exception("[Error] Failed to generate response.")

    def parse_response_output(full_text):
        # ============= 2. 解析输出 (增强的 XML Tags Parsing) =================
        # 查找各部分的位置
        state_start = full_text.find("<state_analysis>")
        state_end = full_text.find("</state_analysis>")
        strategy_start = full_text.find("<strategy>")
        strategy_end = full_text.find("</strategy>")
        response_start = full_text.find("<response>")
        response_end = full_text.find("</response>")
        
        # 初始化内容
        state_content = ""
        raw_strategy_content = ""
        response_content = ""
        
        # 提取 State Analysis 内容
        if state_start != -1 and state_end != -1:
            state_content = full_text[state_start + len("<state_analysis>"):state_end].strip()
        
        # 提取 Strategy 内容
        if strategy_start != -1 and strategy_end != -1:
            raw_strategy_content = full_text[strategy_start + len("<strategy>"):strategy_end].strip()
        
        # 提取 Response 内容（增强处理）
        if response_start != -1:
            response_start_pos = response_start + len("<response>")
            # 如果找到了结束标签，则提取两者之间的内容
            if response_end != -1:
                response_content = full_text[response_start_pos:response_end].strip()
            else:
                # 如果没有找到结束标签，提取从开始标签到文本末尾的所有内容
                response_content = full_text[response_start_pos:].strip()
                # 进一步清理，移除可能的后续标签
                next_tag_start = response_content.find("<")
                if next_tag_start != -1:
                    response_content = response_content[:next_tag_start].strip()
        else:
            # 如果完全没有 <response> 标签，尝试兜底方案
            # 查找最后出现的重要内容作为响应
            last_part_start = max(state_end, strategy_end)
            if last_part_start != -1:
                # 从最后一个结束标签之后开始查找可能的响应内容
                possible_response_start = last_part_start + len("</state_analysis>") if state_end > strategy_end else last_part_start + len("</strategy>")
                if possible_response_start < len(full_text):
                    possible_content = full_text[possible_response_start:].strip()
                    # 移除可能的标签
                    tag_start = possible_content.find("<")
                    if tag_start != -1:
                        possible_content = possible_content[:tag_start].strip()
                    if possible_content:
                        response_content = possible_content
        
        # 兜底：如果所有方法都失败了，使用启发式方法提取响应
        if not response_content:
            # 尝试找到最后一个有意义的文本块
            lines = full_text.split('\n')
            response_lines = []
            in_response = False
            
            for line in reversed(lines):
                line_stripped = line.strip()
                if line_stripped.startswith("</"):
                    break
                elif line_stripped.startswith("<response>"):
                    in_response = True
                    continue
                elif in_response or (line_stripped and not line_stripped.startswith("<")):
                    response_lines.insert(0, line_stripped)
            
            if response_lines:
                response_content = '\n'.join(response_lines).strip()
        
        # 最后的兜底：如果还是空的，就使用整个文本（移除标签）
        if not response_content:
            # 移除所有标签，只保留纯文本内容
            cleaned_text = re.sub(r'<[^>]+>', '', full_text).strip()
            # 分割成行并尝试找到最像响应的内容
            lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
            if lines:
                # 通常响应是最长的几行之一
                response_content = max(lines, key=len)

        # ================= 3. 解析 Strategy 具体字段 =================
        # 目标格式: [Strategy Name]:[Brief Reason]
        strategy_name, strategy_reason = parse_strategy_content(raw_strategy_content)
        return state_content, strategy_name, strategy_reason, response_content

    state_content, strategy_name, strategy_reason, response_content \
         = parse_response_output(full_text)
        
    PrintLogger.print_step("Parsed Result", {
        "State Analysis": state_content[:100] + "..." if len(state_content) > 100 else state_content,
        "Strategy Name": strategy_name,
        "Strategy Reason": strategy_reason,
        "Response": response_content[:100] + "..." if len(response_content) > 100 else response_content
    })

    return {
        "state_analysis": state_content,
        "strategy_name": strategy_name,
        "strategy_reason": strategy_reason,
        "response": response_content,
        "metrics": token_metrics
    }

# ==================== 3. 用户模拟与主循环 ====================
def get_user_response(env, chat_history, user_profile, sys_role, user_role, use_local_api=False):
    """用户模拟器 (保持不变)"""
    conversation = []
    for item in chat_history:
        role = 'assistant' if item['role'] == user_role else "user"
        content = item['content']
        conversation.append({"role": role, "content": content})
    # assert conversation[-1]['role'] == "user", "Last turn must be the System role"
    infos = {'user_profile': user_profile}
    user_messages = get_user_messages(env, user_role, infos, conversation=conversation)
    PrintLogger.print_step("User Simulator Prompt", user_messages)

    generation_payload = {
        "model_name": "gpt-4o",
        "messages": user_messages,
        "temperature": 0.0,
        "n": 1,
        "max_tokens": 1024
    }
    local_api_url = "http://localhost:8093/inference"
    generation_payload['use_local_api'] = use_local_api
    generation_payload['local_api_url'] = local_api_url
    
    try:
        results = call_llm_chat_api_openai(**generation_payload)
        return results[0]
    except Exception as e:
        print(f"User simulator error: {e}")
        return "I'm not sure."

def get_reward_response(env, chat_history, user_profile, sys_role, user_role, use_local_api=False):
    """奖励计算 (保持不变)"""
    conversation = [{"role": 'assistant' if item['role']==user_role else "user", "content": item['content']} for item in chat_history] # BUG!
    assert conversation[-1]['role'] == "assistant", "Last turn must be the User role"
    
    conversation.append({"role": "user", "content": "Would you be willing to donate to Save the Children right now? Please answer yes or no."})
    infos = {'user_profile': user_profile}
    user_messages = get_user_messages(env, user_role, infos, conversation=conversation)
    PrintLogger.print_step("Reward Calculation Prompt", user_messages)

    print(f"Reward Response: -------------")
    generation_payload = {
        "model_name": "gpt-4o",
        "messages": user_messages,
        "temperature": 1.1,
        "n": 10,
        "max_tokens": 25
    }
    local_api_url = "http://localhost:8093/inference"
    generation_payload['use_local_api'] = use_local_api
    generation_payload['local_api_url'] = local_api_url
    try:
        results = call_llm_chat_api_openai(**generation_payload)
        PrintLogger.print_step("Reward Response", results)
        reward = get_reward(results)
        return reward
    except:
        return 0
    
def get_reward(outputs):
    reward = 0
    rewards = []

    for output in outputs:
        if 'yes' in output.lower(): 
            rewards.append(1)
        else:
            rewards.append(-1)
    if -1 in rewards:
        reward = -1
    else:
        reward = 1
    print(f"Reward: {reward}")
    return reward

def run_exp_start_step(input_data_file, output_data_file, 
        use_local_api=True, order_change=False, 
        sys_llm_model_name="Qwen3-8B", sys_use_local_api=False,
        enable_thinking=False,
        exp_data_file=None, temperature=0.0):
    with open(input_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict):
        new_data = []
        for key, value in data.items():
            new_data.append({'infos': {"user_profile": value['description'], "user_name": key}})
        data = new_data
    
    # 采样配置
    random_seed = 42
    random.seed(random_seed)
    data = random.sample(data, min(50, len(data))) 
    # # data = data[4:6] # 测试模式
    # data = data[:1]
    # max_turn = 3

    max_turn = 10

    if exp_data_file is not None and os.path.exists(exp_data_file):
        print(f"Loading existing results from {exp_data_file}...")
        try:
            with open(exp_data_file, 'r', encoding='utf-8') as f:
                exp_data = json.load(f)
            processed_user_ids = {item['user_id'] for item in exp_data}
            data = [d for d in data if d['infos']['user_name'] not in processed_user_ids]
        except Exception as e:
            print(f"Error loading experiment data: {e}")
    else:
        exp_data = []
    
    env = "P4G"
    sys_role = "Persuader"
    user_role = "Persuadee"
    infos = {"strategies": human_persuader_strategy_instruction_map}

    result_data = exp_data

    print(f"\n{PrintLogger.HEADER}=== STARTING EXPERIMENT ==={PrintLogger.ENDC}\n")

    for idx, item in enumerate(tqdm(data, desc="Processing Users")): 
        print(f"\n{PrintLogger.YELLOW}>>> Processing User {idx+1}/{len(data)}: {item['infos']['user_name']} <<<{PrintLogger.ENDC}")

        user_profile = item['infos']['user_profile']
        chat_history = []
        detailed_trajectory = []
        simple_trajectory = []

        for turn in range(max_turn):
            print(f"\n{PrintLogger.HEADER}--- Turn {turn+1} ---{PrintLogger.ENDC}")
            
            if turn == 0:
                # 第一轮 Hardcode
                sys_utt = {
                    "strategy": "Greeting",
                    "content": "Hello! How are you today?",
                    "role": sys_role,
                    "round": turn + 1,
                    "state_analysis": "Initial state",
                    "strategy_reason": "Start conversation naturally." # 第一轮手动补充 reason
                }
                metrics = None
                print(f"{PrintLogger.GREEN}System (Opening): {sys_utt['content']}{PrintLogger.ENDC}")
            else:
                # 调用修改后的生成函数
                unified_result = generate_system_response(
                    env=env,
                    chat_history=chat_history,  
                    strategies_map=infos['strategies'],
                    order_change=order_change,
                    temperature=temperature,
                    use_local_api=sys_use_local_api,
                    sys_llm_model_name=sys_llm_model_name,
                    enable_thinking=enable_thinking
                )

                sys_utt = {
                    "content": unified_result['response'],
                    "strategy": unified_result['strategy_name'],
                    "strategy_reason": unified_result['strategy_reason'], # 收集 Reason
                    "state_analysis": unified_result['state_analysis'],   # 收集 State Analysis
                    "role": sys_role,
                    "round": turn + 1
                }
                metrics = unified_result['metrics']

            # 更新对话历史
            chat_history.append(sys_utt)

            # 保存详细轨迹（包含Name和Reason）
            detailed_trajectory.append({
                "role": sys_role,
                "content": sys_utt['content'],
                "strategy_name": sys_utt.get('strategy'),
                "strategy_reason": sys_utt.get('strategy_reason'), # 保存 Reason
                "state_analysis": sys_utt.get('state_analysis'),
                "metrics": metrics,
                "round": turn + 1
            })

            simple_trajectory.append({
                "role": sys_role,
                "strategy": sys_utt.get('strategy'),
                "content": sys_utt['content'],
                "round": turn + 1
            })

            # === User Turn ===
            user_response = get_user_response(env, chat_history, user_profile, sys_role, 
                                            user_role, use_local_api=use_local_api)
            
            user_utt = {
                "content": user_response,
                "role": user_role,
                "round": turn + 1
            }
            chat_history.append(user_utt)
            
            detailed_trajectory.append({"role": user_role, "content": user_response, "round": turn + 1 })
            simple_trajectory.append({"role": user_role, "content": user_response, "round": turn + 1 })
            print(f"User: {user_response}")

            # === Evaluation ===
            reward = get_reward_response(env, chat_history, user_profile, sys_role, user_role,
                                        use_local_api=use_local_api)
            if reward >= 1.0 or turn == max_turn - 1:
                item_result = {
                    "user_id": item['infos']['user_name'],
                    "user_profile": user_profile,
                    "success": (reward >= 1.0),
                    "reward": reward,
                    "turns": turn + 1,
                    "trajectory": simple_trajectory,
                    "detailed_history": detailed_trajectory # 包含完整信息
                }
                result_data.append(item_result)
                print(f"{PrintLogger.YELLOW}Conversation Ended. Success: {reward >= 1.0}{PrintLogger.ENDC}")
                with open(output_data_file, "w", encoding='utf-8') as f:
                    json.dump(result_data, f, indent=4, ensure_ascii=False)
                break

    print(f"All results saved to {output_data_file}")

import copy

def generate_counterfactual_states(original_state, history_summary, use_local_api=True):
    """
    使用 LLM 生成反事实的状态分析
    返回: {'negative': str, 'positive': str}
    """
    prompt = [
        {"role": "system", "content": "You are an expert data augmentor for dialogue systems."},
        {"role": "user", "content": f"""
        There is a dialogue history between a Persuader and a Persuadee in a donation persuasion scenario, along with the Persuader’s analysis of the current conversation progress, the Persuadee’s mental state, and a prediction of the Persuadee’s next action.
        
        Original State Analysis: "{original_state}"
        Dialogue History: "{history_summary}"
        
        Please generate two counterfactual state analyses that are plausible but semantically opposite to the original one:
        1. [Negative]: The user is extremely resistant, angry, or refuses to communicate.
        2. [Positive]: The user is extremely compliant, happy, and ready to donate immediately.
        
        Output strictly in JSON format:
        {{
            "negative": "...",
            "positive": "..."
        }}
        """}
    ]
    
    PrintLogger.print_step("Counterfactual State Generation Prompt", prompt)

    try:  
        generation_payload = { 
            "model_name": "gpt-4o",
            "messages": prompt,
            "temperature": 0.7,
            "max_tokens": 1024,
            "n": 1,
        }
        generation_payload['use_local_api'] = use_local_api
        result = call_llm_chat_api_openai(**generation_payload)
        print(result[0])
        return json.loads(repair_json(result[0]))
    
    except Exception as e:
        print(f"Error generating counterfactuals: {e}")
        return None

def generate_strategy_with_intervention(env, chat_history, injected_state, strategies_map):
    """
    注入强制的 State Analysis，让模型生成 Strategy
    关键点：在 Prompt 中模拟模型已经输出了 State Analysis 的状态
    """
    strategy_set_str = formatted_strateies(strategies_map)
    history_str = formatted_dialog_history(chat_history)
    
    # 1. 基础 Prompt (与 generate_system_resonse 类似)
    infos = {'strategy_set': strategy_set_str}
    messages = get_message_prompt_func(env, "Persuader", infos, conversation=history_str)
    
    # 2. 构造预填充内容 (Prefill)
    # 格式: <state_analysis>INJECTED_CONTENT</state_analysis>
    prefill_content = f"<state_analysis>{injected_state}</state_analysis>"
    messages.append({"role": "assistant", "content": prefill_content})
    
    # 调用模型
    generation_payload = {
        "task_type": "generate_with_prefill",
        "messages": messages,
        "temperature": 1.0, # 降低温度，测试确定性
        "n": 1, # 多次采样生成，计算改变的频率
        "max_tokens": 1024,
        "logprobs": True
    }
    
    result = call_api(generation_payload)
    if result and "generated_texts" in result:
        full_text = result["generated_texts"][0].strip()
        
        # 解析 Strategy
        strategy_start = full_text.find("<strategy>")
        strategy_end = full_text.find("</strategy>")
        if strategy_start != -1 and strategy_end != -1:
            raw_strategy = full_text[strategy_start + len("<strategy>"):strategy_end].strip()
            name, reason = parse_strategy_content(raw_strategy)
            return name
    return "Unknown"

def run_exp_CounterfactualIntervention(input_data_file, output_data_file, use_local_api=True, order_change=False):
    print(f"\n{PrintLogger.HEADER}=== STARTING COUNTERFACTUAL INTERVENTION EXPERIMENT ==={PrintLogger.ENDC}\n")
    with open(input_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = data[:10]

    env = "P4G"
    infos = {"strategies": human_persuader_strategy_instruction_map}
    
    intervention_results = []
    
    total_turns = 0
    changed_turns_neg = 0
    changed_turns_pos = 0
    
    def __formatted_dialog_history(dialog_history):
        print(dialog_history)

        formatted_history = []
        for item in dialog_history:
            if item['role'] == "Persuader":
                formatted_history.append(f"{item['role']}[{item['strategy_name']}]: {item['content']}")
            else:
                formatted_history.append(f"{item['role']}: {item['content']}")
        return '\n'.join(formatted_history)

    # 遍历每个用户的对话
    for user_idx, item in tqdm(enumerate(data), total=len(data), desc="Processing Users"):
        user_id = item['user_id']
        detailed_history = item['detailed_history']
        
        print(f"\nProcessing User: {user_id}")
        
        # 重建对话历史栈
        current_chat_history = []
        
        for turn_idx, turn_data in enumerate(detailed_history):
            role = turn_data['role']
            content = turn_data['content']
            
            if role == "Persuader" and turn_idx > 0:
                original_state = turn_data.get('state_analysis', "")
                original_strategy = turn_data.get('strategy_name', "")
                
                if not original_state or not original_strategy:
                    current_chat_history.append({"role": role, "content": content})
                    continue
                
                PrintLogger.print_info(f"  Turn {turn_data['round']}: Intervening...")
                
                # 1. 生成反事实状态 (Negative & Positive)
                assert current_chat_history[-1]['role'] == "Persuadee", "Last turn must be Persuadee"
                history_summary = __formatted_dialog_history(current_chat_history)
                counterfactuals = generate_counterfactual_states(original_state, history_summary, use_local_api)
                
                if not counterfactuals:
                    current_chat_history.append({"role": role, "content": content})
                    continue
                
                # 2. 执行干预 A (Negative)
                neg_strategy = generate_strategy_with_intervention(
                    env, current_chat_history, counterfactuals['negative'], infos['strategies']
                )
                
                # 3. 执行干预 B (Positive)
                pos_strategy = generate_strategy_with_intervention(
                    env, current_chat_history, counterfactuals['positive'], infos['strategies']
                )
                
                # 4. 记录结果
                is_changed_neg = (neg_strategy.lower() != original_strategy.lower())
                is_changed_pos = (pos_strategy.lower() != original_strategy.lower())
                
                if is_changed_neg: changed_turns_neg += 1
                if is_changed_pos: changed_turns_pos += 1
                total_turns += 1
                
                intervention_record = {
                    "user_id": user_id,
                    "turn": turn_data['round'],
                    "original_state": original_state,
                    "original_strategy": original_strategy,
                    "negative_intervention": {
                        "state": counterfactuals['negative'],
                        "strategy": neg_strategy,
                        "changed": is_changed_neg
                    },
                    "positive_intervention": {
                        "state": counterfactuals['positive'],
                        "strategy": pos_strategy,
                        "changed": is_changed_pos
                    }
                }
                intervention_results.append(intervention_record)
                
                print(f"    Org Strat: {original_strategy}")
                print(f"    Neg Strat: {neg_strategy} (Changed: {is_changed_neg})")
                print(f"    Pos Strat: {pos_strategy} (Changed: {is_changed_pos})")

            # 将当前轮的真实数据加入历史，准备下一轮
            # 注意：历史必须是真实的，不能是被干预过的，否则不仅是 Counterfactual State，连 History 都变了
            current_chat_history.append({
                "role": role, 
                "content": content,
                "strategy_name": turn_data.get('strategy_name') # 如果 history需要策略名
            })
            
    # 计算统计数据
    stats = {
        "total_turns": total_turns,
        "neg_change_rate": changed_turns_neg / total_turns if total_turns > 0 else 0,
        "pos_change_rate": changed_turns_pos / total_turns if total_turns > 0 else 0,
        "avg_change_rate": (changed_turns_neg + changed_turns_pos) / (2 * total_turns) if total_turns > 0 else 0
    }
    
    print(f"\n{PrintLogger.GREEN}=== Intervention Results ==={PrintLogger.ENDC}")
    print(json.dumps(stats, indent=2))
    
    # 保存结果
    final_output = {
        "stats": stats,
        "details": intervention_results
    }
    
    with open(output_data_file, "w", encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    print(f"Intervention results saved to {output_data_file}")
def main():
    parser = argparse.ArgumentParser(description='Run motivation experiment')
    parser.add_argument('--use_local_api', type=str2bool, default=True, help='Whether to use local API')
    parser.add_argument('--order_change', type=str2bool, default=False, help='Whether to use changed strategy')
    parser.add_argument('--REMARK', type=str, default=None, help='Input data file')
    parser.add_argument('--exp_data_file', type=str, default=None, help='Existing experiment data file to resume from')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for generation')
    parser.add_argument('--sys_llm_model', type=str, default="Qwen3-8B", choices=["Qwen3-8B", "gpt-4o-mini", "qwen2.5-7B-Instruct"], help='System LLM model name')
    parser.add_argument('--sys_local_api', type=str2bool, default=True, help='Whether system LLM uses local API')
    parser.add_argument('--enable_thinking', type=str2bool, default=False, help='Whether to enable thinking')
    args = parser.parse_args()
    print(args)
    beijing_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.datetime.now(beijing_tz).strftime("%Y-%m-%dT%H-%M-%S")

    input_data_file = r"/root/EvolvingAgent-master/EvolvingAgentTest_wym/user_personas.json"
    output_data_file = f"/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/" + \
    f"results_unified_{now}_{args.order_change}_{args.use_local_api}_t_{args.temperature}_sys_{args.sys_llm_model}_think_{args.enable_thinking}.json"
    if args.REMARK is not None:
        output_data_file = output_data_file.replace(".json", f"_{args.REMARK}.json")
    os.makedirs(os.path.dirname(output_data_file), exist_ok=True)

    if args.REMARK == "CI": # Counetfactual Intervention 实验
        input_data_file = r"/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0/results_unified_2025-12-22T15-55-18_False_False_t_0.0.json"
        run_exp_CounterfactualIntervention(input_data_file, output_data_file, use_local_api=args.use_local_api, order_change=args.order_change)
        return
    
    run_exp_start_step(input_data_file, output_data_file, 
                    use_local_api=args.use_local_api, order_change=args.order_change,
                    sys_llm_model_name=args.sys_llm_model, sys_use_local_api=args.sys_local_api,
                    enable_thinking=args.enable_thinking,
                    exp_data_file=args.exp_data_file, temperature=args.temperature)

if __name__ == "__main__":
    main()