import json
import os, sys
import argparse
import numpy as np
from collections import defaultdict

# ==================== 路径与导入设置 ====================
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
grandparent_dir = os.path.dirname(os.path.dirname(parent_dir))
sys.path.insert(0, grandparent_dir)

import json
import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_turn_info(trajectory, target_round):
    """从轨迹中获取指定轮次的策略和内容"""
    for turn in trajectory.get('turns', []):
        # 找到 System 的由 target_round 标识的轮次
        if turn.get('role') == 'Persuader' and turn.get('round') == target_round:
            return {
                "strategy": turn.get('strategy_name') or turn.get('strategy', "Unknown"),
                "content": turn.get('content', "")[:50] + "..." # 截断内容用于显示
            }
    return {"strategy": "Unknown", "content": "N/A"}

def is_success(trajectory):
    """判断轨迹是否成功 (最后一轮有 reward >= 1.0)"""
    if not trajectory.get('turns'): return False
    last_turn = trajectory['turns'][-1]
    # 兼容两种标记方式：直接在 traj 标记或在 last turn 标记
    return last_turn.get('reward', 0.0) >= 1.0 or trajectory.get('success', False)

def get_length(trajectory):
    """获取轨迹长度"""
    if not trajectory.get('turns'): return 0
    # 返回最后一轮的 round 数
    return trajectory['turns'][-1].get('round', 0)

def analyze_detailed(input_file):
    data = load_data(input_file)
    if not data: return

    print(f"Analyzing File: {input_file}")
    
    # === 容器初始化 ===
    # 1. 详细日志 (用于生成 CSV/Excel)
    detail_rows = []
    
    # 2. 全局统计
    global_total_traj = 0
    global_success_traj = 0
    global_branch_success_lengths = []
    
    # 3. 用户级统计
    user_stats_map = {} # user_id -> stats

    for episode in data:
        user_id = episode.get('user_id', 'Unknown')
        trajectories = episode.get('trajectories', [])
        
        # 找到 Root
        root_traj = next((t for t in trajectories if t['id'] == 'root'), None)
        if not root_traj: continue
        
        # --- 用户级临时变量 ---
        u_total_traj = 0
        u_success_traj = 0
        u_branch_success_lens = []
        
        # 1. 处理 Root 轨迹
        root_succ = is_success(root_traj)
        root_len = get_length(root_traj)
        
        u_total_traj += 1
        if root_succ: u_success_traj += 1
        
        # 记录 Root 详情
        detail_rows.append({
            "User ID": user_id,
            "Type": "Root",
            "Traj ID": "root",
            "Branch Round": "-",
            "Old Strategy": "-",
            "New Strategy": "-", # Root 没有新策略
            "Success": "✅" if root_succ else "❌",
            "Turns": root_len
        })

        # 2. 处理分叉轨迹
        branch_trajs = [t for t in trajectories if t['id'] != 'root']
        
        for b_traj in branch_trajs:
            u_total_traj += 1
            
            b_succ = is_success(b_traj)
            b_len = get_length(b_traj)
            b_round = b_traj.get('branch_at_turn', -1)
            
            if b_succ:
                u_success_traj += 1
                u_branch_success_lens.append(b_len)
            
            # 提取策略对比
            # 旧策略：Root 在该轮的策略
            old_info = get_turn_info(root_traj, b_round)
            # 新策略：当前分叉轨迹在该轮的策略
            new_info = get_turn_info(b_traj, b_round)
            
            detail_rows.append({
                "User ID": user_id,
                "Type": "Branch",
                "Traj ID": b_traj['id'],
                "Branch Round": b_round,
                "Old Strategy": old_info['strategy'],
                "New Strategy": new_info['strategy'],
                "Success": "✅" if b_succ else "❌",
                "Turns": b_len
            })

        # --- 汇总用户级数据 ---
        u_success_rate = u_success_traj / u_total_traj if u_total_traj > 0 else 0
        u_avg_len = np.mean(u_branch_success_lens) if u_branch_success_lens else 0.0
        
        user_stats_map[user_id] = {
            "Total Trajectories": u_total_traj,
            "Success Trajectories": u_success_traj,
            "Success Rate": u_success_rate,
            "Avg Success Branch Len": u_avg_len
        }
        
        # --- 累加全局数据 ---
        global_total_traj += u_total_traj
        global_success_traj += u_success_traj
        global_branch_success_lengths.extend(u_branch_success_lens)

    # === 生成输出 ===
    
    # 1. 详细 DataFrame
    df_details = pd.DataFrame(detail_rows)
    
    # 2. 用户统计 DataFrame
    df_user_stats = pd.DataFrame.from_dict(user_stats_map, orient='index')
    df_user_stats.reset_index(inplace=True)
    df_user_stats.rename(columns={'index': 'User ID'}, inplace=True)
    
    # 3. 全局统计字典
    overall_stats = {
        "Global Trajectory Success Rate": global_success_traj / global_total_traj if global_total_traj > 0 else 0,
        "Total Trajectories Generated": global_total_traj,
        "Total Successful Trajectories": global_success_traj,
        "Global Avg Success Branch Length": float(np.mean(global_branch_success_lengths)) if global_branch_success_lengths else 0.0
    }

    # === 打印报告 ===
    print("\n" + "="*80)
    print(" >>> 1. Overall Statistics <<<")
    print("-" * 30)
    print(json.dumps(overall_stats, indent=4))
    
    print("\n" + "="*80)
    print(" >>> 2. Per-User Statistics (Top 5) <<<")
    print("-" * 30)
    # 设置显示格式
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_user_stats.head(5).to_string(index=False))

    print("\n" + "="*80)
    print(" >>> 3. Detailed Trajectory Comparison (Example) <<<")
    print("-" * 30)
    # 打印一个有分叉的用户例子
    example_user = df_details[df_details['Type'] == 'Branch'].iloc[0]['User ID'] if not df_details.empty else None
    if example_user:
        print(df_details[df_details['User ID'] == example_user].to_string(index=False))
    
    # === 保存文件 ===
    base_name = input_file.replace(".json", "")
    
    # 保存详细 CSV (Excel Friendly)
    csv_path = f"{base_name}_detailed_analysis.csv"
    df_details.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    # 保存统计 JSON
    stats_json_path = f"{base_name}_metrics_summary.json"
    output_json = {
        "overall": overall_stats,
        "user_stats": user_stats_map
    }
    with open(stats_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=4)
        
    print(f"\n✅ Analysis saved to:\n  - {csv_path}\n  - {stats_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    motivation_dir = r"/root/EvolvingAgent-master/EvolvingAgentTest_wym"
    exp_file = r"model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0/rollout_v1/results_dissonance_t_1.0.json"
    input_file = os.path.join(motivation_dir, exp_file)
    analyze_detailed(input_file=input_file)