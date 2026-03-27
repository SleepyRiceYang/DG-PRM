import json,requests,os,sys
import argparse, os, sys, datetime, glob, importlib, csv, logging, pytz
import time
import random

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
grandparent_dir = os.path.dirname(os.path.dirname(parent_dir))
print(grandparent_dir)
sys.path.insert(0, grandparent_dir)

from json_repair import repair_json
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.Prompt import get_user_messages, get_system_messages_v4, Persuader_First_Sentence, get_system_messages
from utils.Prompt import human_persuader_strategy_instruction_map
from utils.api_call import *
from utils.retry import *

API_URL = "http://127.0.0.1:8992/inference"

## 利用Transformer建立的API
def call_api(payload: dict):
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

def translate_to_lower_keys(input_dict):
    return {k.lower(): v for k, v in input_dict.items()}

def get_assistant_prefix(tokenizer, messages: list) -> str:
    if not messages or messages[-1]['role'] != 'user':
        raise ValueError("The last message in the history must be from the 'user' to get the assistant prefix.")
    prompt_no_assistant = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    prompt_with_assistant = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    if prompt_with_assistant.startswith(prompt_no_assistant):
        assistant_prefix = prompt_with_assistant[len(prompt_no_assistant):]
        return assistant_prefix
    else:
        # 这种情况很少见，但作为健壮性检查
        raise RuntimeError("Could not determine the assistant prefix. The chat template might be unusual.")

def get_last_assistant_json_template(assitant_content) -> str:
    strategy_marker = '"strategy"'
    index_start = assitant_content.find(strategy_marker)
    if index_start == -1:
        raise ValueError("The assistant content does not contain a 'strategy' field.")
    index_end = index_start + len(strategy_marker)
    json_template = assitant_content[:index_end]
    return json_template

def get_strategy_entropy_correct(conversation_history: list, tokenizer):
    history_prompt = tokenizer.apply_chat_template(
        conversation_history[:-1], # 不要最后一轮模型的回复
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    assistant_prefix = get_assistant_prefix(tokenizer, conversation_history[:-1])
    incomplete_json_template = get_last_assistant_json_template(assitant_content=conversation_history[-1]['content'])
    text_to_probe = history_prompt + assistant_prefix + incomplete_json_template
    return text_to_probe

def formatted_strateies(strategies):
    return '\n'.join([f"{i+1}. {key}: {value}" for i, (key, value) in enumerate(strategies.items())])

# 加入了历史交互轨迹
def formatted_dialog_history(dialog_history, sys_role, user_role):
    formatted_history = ""
    for turn in dialog_history:
        role = turn['role']
        content = turn['content']
        formatted_history += f"{role.capitalize()}: {content}\n"
    return formatted_history

def formatted_dialog_trajectory(dialog_history):
    results = []
    for idx in range(0, len(dialog_history) - 1, 2):
        sys_utt = dialog_history[idx]
        user_utt = dialog_history[idx+1]
        content = f"""
        Turn: {sys_utt['round']}
        {sys_utt['role']} response: {sys_utt['content']}
        {user_utt['role']} response: {user_utt['content']} 
        """
        # 删除模型的理由，模拟最真实的标注场景
        results.append(content.strip())
    return "\n---\n".join(results)

def AI_judge_prompt(dialog_history, task_info):
    dialog_trajcetory = formatted_dialog_trajectory(dialog_history)
    strategy_content = formatted_strateies(task_info['strategies'])
    messages = [
        {"role": "system", "content": "You are an expert AI judge specializing in evaluating and inferring strategies from dialogue trajectories."},

        {"role": "user", "content": f"""
            ## Role Definition
            You are a top-tier dialogue strategy analyst, skilled at dissecting conversation flow and identifying high-level plans. Your core task is to **infer the underlying dialogue strategies** used by the system in each turn, and **segment the conversation into key stages**.
            
            You will receive a complete dialogue trajectory where the system's previous strategies are NOT explicitly labeled, only the responses.
            
            ## Task Background:
            * Task Description: {task_info['description']}
            * System Role: {task_info['system_role']}
            * User Role: {task_info['user_role']}
            * **Known Strategy Set (Reference):** {strategy_content} own strategies.

            ## Dialogue Trajectory (Strategy NOT Labeled):
            {dialog_trajcetory}
            
            ## Thinking Process:
            1. Global Understanding & Stage Segmentation: First, read the entire dialogue. Segment the conversation into key stages (e.g., Opening, Value Building, Negotiation, etc.).
            2. **Strategy Inference & Attribution:** Analyze each system response sequentially. **For every turn, infer one or more strategies that were used to formulate the response.**
               * **Naming Convention:** Use the exact name from the **Known Strategy Set**. If the strategy is NOT in the set, summarize and name a **New Strategy** (Format: "New Strategy Name: Brief description").
               * **Multi-Strategy Labeling:** If a turn uses multiple strategies, list them sequentially.
            3. Output Generation: Generate your analysis strictly following the specified output format.

            ## Please strictly adhere to the following output format:
            ```
            <thinking_process>
            [Your detailed thought process on flow, user resistance, and key strategic choices.]
            <stage_segmentation_result>
            ```json
            [
                {{
                    "stage idx": 1,
                    "stage name": "XX", 
                    "turns": [X,X],
                    "reasoning": "Briefly explain what dialogue content or strategy shifts led you to define this stage."
                }}
            ]
            ```
            <strategy_inference_result>
            ```json
            [
                {{
                    "turn": 1, 
                    "inferred_strategies": [ // Array to hold one or more strategies
                        {{"strategy_name": "Strategy Name 1", "reasoning": "Basis for Strategy 1 inference."}},
                        {{"strategy_name": "New Strategy: Brief Description", "reasoning": "Basis for Strategy 2 inference."}}
                    ]
                }},
                // ... and so on for all turns
            ]
            ```
            ```
            Attention! You must infer strategies for ALL system turns. Do not miss any turns.
        """}
    ]
    return messages

def parse_llm_output_by_markers(result_text):
    # 定义三个标记
    think_marker = "<thinking_process>"
    segment_marker = "<stage_segmentation_result>"
    judgement_marker = "<strategy_inference_result>"

    # --- 1. 定位每个标记的起始位置 ---
    # .find() 如果找不到会返回 -1
    idx_think = result_text.find(think_marker)
    idx_segment = result_text.find(segment_marker)
    idx_judgement = result_text.find(judgement_marker)

    # --- 2. 检查标记是否存在 ---
    # 如果任何一个关键标记不存在，可能无法正确解析，可以选择返回错误或空字典
    if idx_think == -1 or idx_segment == -1 or idx_judgement == -1:
        print("错误：输出中缺少一个或多个关键标记。")
        raise ValueError(f"输出格式错误 {result_text}")

    # 提取 <thinking_process> 的内容
    # 内容开始于 think_marker 之后，结束于 segment_marker 之前
    start = idx_think + len(think_marker)
    end = idx_segment
    thinking_process_raw = result_text[start:end].strip()

    # 提取 <stage_segmentation_result> 的内容
    # 内容开始于 segment_marker 之后，结束于 judgement_marker 之前
    start = idx_segment + len(segment_marker)
    end = idx_judgement
    stage_segmentation_raw = result_text[start:end].strip()

    # 提取 <strategy_judgment_result> 的内容
    # 内容开始于 judgement_marker 之后，一直到字符串末尾
    start = idx_judgement + len(judgement_marker)
    strategy_judgment_raw = result_text[start:].strip()

    # 将提取出的原始字符串块传入辅助函数进行解析
    stage_segmentation_json = json.loads(repair_json(stage_segmentation_raw))
    strategy_judgment_json = json.loads(repair_json(strategy_judgment_raw))

    return {
        "thinking_process": thinking_process_raw,
        "stage_segmentation_raw": stage_segmentation_raw, # 保留原始字符串块
        "strategy_judgment_raw": strategy_judgment_raw,   # 保留原始字符串块
        "stage_segmentation_json": stage_segmentation_json, # 解析后的JSON对象
        "strategy_judgment_json": strategy_judgment_json  # 解析后的JSON对象
    }

# 这将捕获并重试所有继承自 Exception 的错误（几乎是所有常见的运行时错误）
@retry(max_attempts=5, delay=2, exceptions=(Exception,), backoff=2)
def get_judge_response(dialog_history, task_info):
    messages = AI_judge_prompt(dialog_history, task_info)
    results = call_llm_chat_api_openai(model_name="gpt-4o",messages=messages, temperature=0., n=1, max_tokens=10240)
    # 进行处理
    result = results[0]
    judge_result = parse_llm_output_by_markers(result_text=result)
    return judge_result

def get_reward_response(env, chat_history, user_profile, sys_role, user_role):
    conversation = [{"role": 'assistant' if item['role']==user_role else "user", "content": item['content']} for item in chat_history]
    conversation.append({
        "role": "user",
        "content": f"""
            Would you be willing to donate to Save the Children right now?
            """
    })
    infos = {}
    infos['user_profile'] = user_profile
    user_messages = get_user_messages(env, user_role, infos, conversation=conversation)
    print_list(user_messages)
    generation_payload = {
        "model_name": "gpt-4o",
        "messages": user_messages,
        "temperature": 1.1,
        "n":10,
        "max_tokens": 256
    }
    try:
        results = call_llm_chat_api_openai(
            model_name=generation_payload['model_name'],
            messages=generation_payload['messages'],
            temperature=generation_payload['temperature'],
            n=generation_payload['n'],
            max_tokens=generation_payload['max_tokens']
        )
        # print(results)
        reward = get_reward(results)
    except ValueError as e:
        print(f"捕获到错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")
    return reward

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

def print_list(history):
    for item in history:
        print(f"{item['role']}: {item['content']}")

def get_system_response_old_prompt(env, chat_history, infos, sys_role, user_role, tokenizer):
    conversation = []
    for message in chat_history:
        if message['role'] == sys_role:
            conversation.append({
                "role": "assistant", "content": message['content']
            })
        else:
            conversation.append({
                "role": "user", "content": message['content']
            })
    system_messages = get_system_messages(env, sys_role, infos, conversation, "", False)
    enable_thikning = True
    generation_payload = {
        "task_type": "generate",
        "messages": system_messages,
        "temperature": 0.0,
        "n": 1,
        "max_tokens": 10240,
        "enable_thinking": enable_thikning
    }
    result = call_api(generation_payload)
    print(f"调用本地API的结果: {result}")
    if result and "generated_texts" in result:
        response = result["generated_texts"][0].strip()
    else:
        raise ValueError("Failed to get a valid response from the API.")

    if enable_thikning:
        think_start = response.find("<think>")
        think_end = response.find("</think>")

        if think_start != -1 and think_end != -1:
            reason = response[think_start + len("<think>") : think_end].strip()
            response_text = response[think_end + len("</think>") :].strip()
        else:
            # 如果没有 think 标签，则返回原文
            reason = ""
            response_text = response

        strategy_result_json = {
            "response": response_text,
            "reason": reason
        }
    else:
        strategy_result_json = {
            "response": response,
            "reason": None
        }

    return strategy_result_json

def get_system_response(env, chat_history, infos, sys_role, user_role, tokenizer):
    chat_history_content = formatted_dialog_history(chat_history, sys_role, user_role)
    new_infos = {}
    new_infos['strategy_set'] = formatted_strateies(infos['strategies'])
    system_messages = get_system_messages_v4(env, sys_role, new_infos, chat_history_content)
    print_list(system_messages)

    ## 首先生成回复
    generation_payload = {
        "task_type": "generate",
        "messages": system_messages,
        "temperature": 0.0,
        "n": 1,
        "max_tokens": 10240,
        "enable_thinking": False
    }
    # print(generation_payload)
    result = call_api(generation_payload)
    print(f"调用本地API的结果: {result}")
    if result and "generated_texts" in result:
        response = result["generated_texts"][0].strip()
        strategy_result_json = translate_to_lower_keys(json.loads(repair_json(response)))
    else:
        raise ValueError("Failed to get a valid response from the API.")

    return strategy_result_json

def get_user_response(env, chat_history, user_profile, sys_role, user_role):
    conversation = [{"role": 'assistant' if item['role']==user_role else "user", "content": item['content']} for item in chat_history]
    infos = {}
    infos['user_profile'] = user_profile
    user_messages = get_user_messages(env, user_role, infos, conversation=conversation)
    print_list(user_messages)
    generation_payload = {
        "model_name": "gpt-4o",
        "messages": user_messages,
        "temperature": 0.,
        "n":1,
        "max_tokens": 1024
    }
    try:
        results = call_llm_chat_api_openai(
            model_name=generation_payload['model_name'],
            messages=generation_payload['messages'],
            temperature=generation_payload['temperature'],
            n=generation_payload['n'],
            max_tokens=generation_payload['max_tokens']
        )
    except Exception as e:
        print(f"发生未知错误: {e}")
    response = results[0]
    return response

def run_exp_motivation_0(input_data_file, output_data_file, tokenizer):
    with open(input_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict):
        new_data = []
        for key, value in data.items():
            item = {}
            item['infos'] = {
                "user_profile": value['description'],
                "user_name": key
            }
            new_data.append(item)
        data = new_data
    random_seed = 42
    random.seed(random_seed)
    data = random.sample(data, 50)
    # data = data[:1]
    print(f"共有{len(data)}个用户数据")
    env = "P4G"
    sys_role = "Persuader"
    user_role = "Persuadee"
    infos = {
        "strategies": human_persuader_strategy_instruction_map
    }

    ## 进行每一个角色的对话
    result_data = []
    max_turn = 10
    for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing User Profiles"):
        item_result = {}

        user_profile = item['infos']['user_profile']
        chat_history = []
        for turn in range(max_turn):
            # 系统进行发言
            if turn == 0:
                sys_utt = {
                    "strategy": "greetings",
                    "content": Persuader_First_Sentence,
                    "role": sys_role,
                    "round": turn + 1
                }
            else:
                sys_result = get_system_response_old_prompt(
                    env=env,
                    chat_history=chat_history,
                    infos=infos,
                    sys_role=sys_role,
                    user_role=user_role, tokenizer=tokenizer
                )
                sys_utt = {
                    "content": sys_result['response'], # 加入到对话历史
                    "reason": sys_result['reason'],
                    "role": sys_role,
                    "round": turn+1,
                }

            chat_history.append(sys_utt)

            user_result = get_user_response(
                env=env,
                chat_history=chat_history,
                user_profile=user_profile,
                sys_role=sys_role,
                user_role=user_role)
            user_utt = {
                "content": user_result,
                "role": user_role,
                "round": turn + 1
            }
            chat_history.append(user_utt)

            ## 评估
            reward = get_reward_response(env, chat_history, user_profile, sys_role, user_role)

            if reward >= 1.0:
                success = True
                result = {
                    "success": success,
                    "turns": turn + 1,
                }
                break
            if turn == max_turn - 1:
                success = False
                result = {
                    "success": success,
                    "turns": max_turn,
                }
                break
        item_result = {
            "infos": {
                "user_id": item['infos']['user_name'],
                "user_profile": user_profile,
                "success": result['success'],
                "reward": reward,
                "turns": result['turns'],
            },
            "trajectory":chat_history
        }

        # 进行AI judge分析
        # task_info = {
        #     "description":"""
        #     Save the Children is head-quartered in London, and they work to help fight poverty around the world.
        #     Children need help in developing countries and war zones. Small donations like $1 or $2 go a long way to help.
        #     The persuader used strategic dialogue to persuade the persuadee to make a donation.
        #     """,
        #     "system_role": sys_role,
        #     "user_role": user_role,
        #     "strategies": human_persuader_strategy_instruction_map
        # }
        # judge_result = get_judge_response(chat_history, task_info)

        # item_result['judge_result'] = {
        #     "thinking_process": judge_result['thinking_process'],
        #     "stage_seg_result": judge_result['stage_segmentation_json'],
        #     "strategy_judge_result": judge_result['strategy_judgment_json']
        # }
        # 保存结果
        result_data.append(item_result)

    with open(output_data_file, "w", encoding='utf-8') as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)

def main():
    # input_data_file = r"/root/EvolvingAgent-master/EvolvingAgentTest/model/Motivation/selected_users.json"
    beijing_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.datetime.now(beijing_tz).strftime("%Y-%m-%dT%H-%M-%S")

    input_data_file = r"/root/EvolvingAgent-master/EvolvingAgentTest/user_personas.json"
    output_data_file = f"/root/EvolvingAgent-master/EvolvingAgentTest/model/Motivation/results_e0_v5_{now}.json"

    MODEL_PATH = "/data/pretrained_models/Qwen3-8B" # ！！重要：替换成你的模型路径
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"无法加载 Tokenizer: {e}")
        exit()
    run_exp_motivation_0(input_data_file, output_data_file, tokenizer)

if __name__ == "__main__":
    main()