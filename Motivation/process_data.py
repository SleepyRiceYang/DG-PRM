import json
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
grandparent_dir = os.path.dirname(os.path.dirname(parent_dir))
sys.path.insert(0, grandparent_dir)

from model.Motivation.analysis_add_strategy_reason.analysis_token import *

def str2bool(value):
    if isinstance(value, bool): return value
    if value.lower() in ("true", "t", "1"): return True
    elif value.lower() in ("false", "f", "0"): return False
    else: raise argparse.ArgumentTypeError("Boolean value expected.")

def calculate_mean_entropy(tokens):
    if not tokens: return 0.0
    return np.mean([t['entropy'] for t in tokens])

def plot_distribution(data, title, xlabel, save_path):
    """绘制分布图并保存"""
    if len(data) == 0: 
        raise ValueError("Data is empty.")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {title}')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    
    # 添加均值和中位数线
    mean_val = np.mean(data)
    median_val = np.median(data)
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='-', label=f'Median: {median_val:.2f}')
    plt.legend()
    
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def calculate_percentiles(data):
    """计算 0% - 100% 每隔 10% 的分位数"""
    # 修改点：使用 len(data) == 0 来判断 NumPy 数组是否为空
    if len(data) == 0: 
        raise ValueError("Data is empty.")
    
    percentiles = list(range(0, 101, 10))
    values = np.percentile(data, percentiles)
    return {f"p{p}": float(v) for p, v in zip(percentiles, values)}

def process_data(input_file, output_dir):
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    summary_records = []
    detailed_records = []

    all_hs_values = []
    all_ha_values = []
    all_z_hs_values = [] # 新增
    all_z_ha_values = [] # 新增

    try:
        user_persona = json.load(open(r"/root/EvolvingAgent-master/EvolvingAgentTest_wym/user_personas.json", 'r', encoding='utf-8'))
    except FileNotFoundError:
        print("Warning: user_personas.json not found, skipping persona info.")
        user_persona = {}

    print("Processing trajectories...")
    
    performance_stats = {
        "total": {"count": 0, "success": 0, "turns": []},
        "persona": {},  # Big-Five
        "decision": {}  # Decision-Making Style
    }
    def update_group_stats(group_dict, key, is_success, n_turns):
        key = key.strip().lower() # 统一转换为小写
        if key not in group_dict:
            group_dict[key] = {"count": 0, "success": 0, "turns": []}
        
        group_dict[key]["count"] += 1
        if is_success:
            group_dict[key]["success"] += 1
        group_dict[key]["turns"].append(n_turns)

    for episode in tqdm(data):
        user_id = episode.get('user_id', 'Unknown')
        detailed_history = episode.get('detailed_history', [])
        
        is_success = episode.get('success', False)
        # 注意：test_start_step4 保存的 turns 是 int，如果是 list 取长度
        n_turns = episode.get('turns', 0)
        if isinstance(n_turns, list): n_turns = len(n_turns)

        performance_stats["total"]["count"] += 1
        if is_success:
            performance_stats["total"]["success"] += 1
        performance_stats["total"]["turns"].append(n_turns)
        
        # 2. 人格统计 (Big-Five)
        p_type = user_persona.get(user_id, {}).get('Big-Five Personality', 'unknown')
        update_group_stats(performance_stats["persona"], p_type, is_success, n_turns)
        
        # 3. 决策风格统计 (Decision-Making Style)
        d_type = user_persona.get(user_id, {}).get('Decision-Making Style', 'unknown')
        update_group_stats(performance_stats["decision"], d_type, is_success, n_turns)

        episode_summary = {
            "user_id": user_id,
            "success": is_success,
            "turns": n_turns,
            "user_profile": episode.get('user_profile', "unknown"),
            "five-personality": p_type.lower(),
            "decision-making-style": d_type.lower(),
            "episode": [],
        }
        
        history_hs = []
        history_ha = []

        episode_detail = {
            "user_id": user_id,
            "success": is_success,
            "turns": n_turns,
            "episode": [],
        }

        for turn in detailed_history:
            if turn['role'] == 'Persuader':
                metrics = turn.get('metrics', [])
                if not metrics: 
                    turn_summary = {
                        "role": "Persuader",
                        "round": turn['round'],
                        "content": turn.get("content", ""),
                        "strategy_name": turn.get("strategy_name", "Greeting"), # 默认 Greeting
                        "hs": None, "ha": None, "hr": None, # 补零
                        "z_score_hs": None, "z_score_ha": None
                    }
                    episode_summary['episode'].append(turn_summary)
                    # 同时也需要加入到 detail 中，保持结构对齐
                    turn_detail = {
                        "role": "Persuader",
                        "round": turn['round'],
                        "content": turn.get("content", ""), # 关键：保留 content
                        "strategy_name": turn.get("strategy_name", "Greeting"),
                        "metrics": []
                    }
                    episode_detail['episode'].append(turn_detail)
                    continue
                state_tokens, strategy_tokens, response_tokens = segment_metrics_new_format(metrics)
                
                hs = calculate_mean_entropy(state_tokens)
                ha = calculate_mean_entropy(strategy_tokens)
                hr = calculate_mean_entropy(response_tokens)
                
                all_hs_values.append(hs)
                all_ha_values.append(ha)
                
                # Z-Score Calculation
                if len(history_hs) < 2:
                    z_score_hs = 0.0
                else:
                    mean_hist = np.mean(history_hs)
                    std_hist = np.std(history_hs) + 1e-6
                    z_score_hs = (hs - mean_hist) / std_hist
                
                if len(history_ha) < 2:
                    z_score_ha = 0.0
                else:
                    mean_hist_a = np.mean(history_ha)
                    std_hist_a = np.std(history_ha) + 1e-6
                    z_score_ha = (ha - mean_hist_a) / std_hist_a
                
                all_z_hs_values.append(z_score_hs)
                all_z_ha_values.append(z_score_ha)

                history_hs.append(hs)
                history_ha.append(ha)

                turn_summary = {
                    "role": "Persuader",
                    "round": turn['round'],
                    "content": turn.get("content", ""),
                    "hs": hs, 
                    "ha": ha, 
                    "hr": hr,
                    "z_score_hs": z_score_hs,
                    "z_score_ha": z_score_ha,
                    "strategy_name": turn.get("strategy_name", ""),
                    "state_analysis": turn.get("state_analysis", ""),
                }
                episode_summary['episode'].append(turn_summary)
                
                turn_detail = {
                    "role": "Persuader",
                    "round": turn['round'],
                    "metrics": turn.get('metrics', []),
                }
                episode_detail['episode'].append(turn_detail)
            else:
                # User Turn
                episode_summary['episode'].append({"role": "Persuadee", "content": turn.get("content", "")})
                episode_detail['episode'].append({"role": "Persuadee", "content": turn.get("content", "")})
        
        summary_records.append(episode_summary)
        detailed_records.append(episode_detail)

    # 4. 计算全局统计与分位数
    print("Calculating global statistics and visualizing...")
    
    # 转换为 numpy 数组
    hs_array = np.array([x for x in all_hs_values if x is not None])
    ha_array = np.array([x for x in all_ha_values if x is not None])
    z_hs_array = np.array([x for x in all_z_hs_values if x is not None])
    z_ha_array = np.array([x for x in all_z_ha_values if x is not None])
    
    print(f"Total Persuader Turns Processed: {len(hs_array)}, {len(ha_array)}, {len(z_hs_array)}, {len(z_ha_array)}")

    # 计算详细分位数 (0% - 100%)
    global_stats = {
        "hs_stats": {
            "mean": float(np.mean(hs_array)),
            "std": float(np.std(hs_array)),
            "percentiles": calculate_percentiles(hs_array)
        },
        "ha_stats": {
            "mean": float(np.mean(ha_array)),
            "std": float(np.std(ha_array)),
            "percentiles": calculate_percentiles(ha_array)
        },
        "z_hs_stats": {
            "mean": float(np.mean(z_hs_array)),
            "std": float(np.std(z_hs_array)),
            "percentiles": calculate_percentiles(z_hs_array)
        },
        "z_ha_stats": {
            "mean": float(np.mean(z_ha_array)),
            "std": float(np.std(z_ha_array)),
            "percentiles": calculate_percentiles(z_ha_array)
        }
    }
    
    # 添加用于 Config 的关键阈值 (为了向后兼容)
    global_stats.update({
        "hs_top_20": global_stats["hs_stats"]["percentiles"]["p80"],
        "hs_top_10": global_stats["hs_stats"]["percentiles"]["p90"],
        "ha_top_20": global_stats["ha_stats"]["percentiles"]["p80"],
        "ha_bottom_20": global_stats["ha_stats"]["percentiles"]["p20"],
    })
    
    def calc_metrics(stats_node):
        total = stats_node["count"]
        success_num = stats_node["success"]
        if total == 0: return {"sr": 0.0, "success_count": 0, "avg_turns": 0.0, "count": 0}
        return {
            "success_rate": success_num / total,
            "success_count": success_num,
            "avg_turns": float(np.mean(stats_node["turns"])),
            "count": total
        }
    performance_summary = {
        "overall": calc_metrics(performance_stats["total"]),
        "by_persona": {k: calc_metrics(v) for k, v in performance_stats["persona"].items()},
        "by_decision_style": {k: calc_metrics(v) for k, v in performance_stats["decision"].items()}
    }
    global_stats["performance"] = performance_summary

    print("\n=== Global Statistics Summary ===")
    print(json.dumps(global_stats, indent=2))

    # 5. 可视化分布
    os.makedirs(output_dir, exist_ok=True)
    plot_distribution(hs_array, "State Entropy (Hs)", "Entropy Value", os.path.join(output_dir, "dist_hs.png"))
    plot_distribution(ha_array, "Action Entropy (Ha)", "Entropy Value", os.path.join(output_dir, "dist_ha.png"))
    plot_distribution(z_hs_array, "Z-Score of State Entropy", "Z-Score", os.path.join(output_dir, "dist_z_hs.png"))
    plot_distribution(z_ha_array, "Z-Score of Action Entropy", "Z-Score", os.path.join(output_dir, "dist_z_ha.png"))

    # 6. 保存文件
    summary_path = os.path.join(output_dir, "summary_stats.json")
    detail_path = os.path.join(output_dir, "detailed_tokens.json")
    
    final_summary = {
        "global_stats": global_stats,
        "episodes": summary_records
    }
    
    print(f"\nSaving summary to {summary_path}...")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(final_summary, f, indent=2, ensure_ascii=False)
        
    print(f"Saving details to {detail_path}...")
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_records, f, indent=2, ensure_ascii=False)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Path to the unified result json file")
    parser.add_argument("--output_dir", default=None, help="Directory to save the processed files")
    
    # 只需要 input 即可推断其他参数
    args = parser.parse_args()

    if args.input and not args.output_dir:
        # 自动生成输出目录名
        parts = args.input.split("/")[-1].split("_")
        # 假设文件名格式: results_unified_TIME_BOOL_BOOL_t_FLOAT_sys_ModelName_think_BOOL.json # 最后两个 bool 代表order_change 和 use_local_api
        exp_time = parts[2]
        order_change = parts[3]
        use_local_api = parts[4]
        temperature = parts[6]
        sys_model_name = parts[8]
        thinking_flag = parts[10].replace(".json", "")

        dir_name = f"exp_{exp_time}_Order_{order_change}_User_{use_local_api}_t_{temperature}_sys_{sys_model_name}_think_{thinking_flag}"
        # 假设输出到 input 同级目录的 Experiments 文件夹下
        base_dir = os.path.dirname(args.input)
        args.output_dir = os.path.join(base_dir, dir_name)

    if args.input:
        print(f"Input: {args.input}")
        print(f"Output: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)
        process_data(args.input, args.output_dir)
    else:
        print("Please provide --input path.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", default=None, help="Path to the unified result json file")
#     parser.add_argument("--output_dir", default=None, help="Directory to save the processed files")
#     parser.add_argument('--mode', type=str, default="exp", choices=["exp", "random", "action_entropy", "state_entropy", "dissonance"])
#     parser.add_argument('--use_local_api', type=str2bool, default=True, help='Whether to use local API')
#     parser.add_argument('--order_change', type=str2bool, default=False, help='Whether to use changed strategy')
#     args = parser.parse_args()

#     args.mode = "exp"   
#     exp_time = args.input.split("/")[-1].split("_")[2]
#     args.order_change = args.input.split("/")[-1].split("_")[3]
#     args.use_local_api = args.input.split("/")[-1].split("_")[4]
#     temperature = args.input.split("/")[-1].split("_")[6].replace(".json", "")
#     args.output_dir = r"/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/" + f"{args.mode}_{exp_time}_Order_{args.order_change}_User_{args.use_local_api}_t_{temperature}"
#     print(args.output_dir)
#     os.makedirs(args.output_dir, exist_ok=True)
#     process_data(args.input, args.output_dir)