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

import json
import os
import argparse
import numpy as np
from collections import defaultdict

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_strategy_at_turn(trajectory, turn_num):
    """获取指定轮次的策略名称"""
    if not trajectory or 'turns' not in trajectory:
        return "Unknown"
    
    for turn in trajectory['turns']:
        # 匹配 Persuader 的轮次
        if turn.get('role') == 'Persuader' and turn.get('round') == turn_num:
            # 优先取 strategy_name, 其次 strategy
            return turn.get('strategy_name') or turn.get('strategy') or "Unknown"
    return "N/A"

def is_traj_success(trajectory):
    # BUG: 还要再处理一下
    """判断单条轨迹是否成功"""
    if trajectory.get('success'):
        return trajectory['success']
    if not trajectory.get('turns'): return False
    # 检查最后一轮是否有 reward >= 1.0
    last_turn = trajectory['turns'][-1]
    return last_turn.get('reward', 0.0) >= 1.0

def get_traj_length(trajectory):
    """获取轨迹轮数"""
    if not trajectory.get('turns'): return 0
    return trajectory['turns'][-1].get('round', 0)

def analyze_and_update_eval(input_file):
    data = load_data(input_file)
    if not data: return

    print(f"Analyzing: {input_file}")
    
    # === 1. 原始的高层指标统计 (Episode Level) ===
    total_episodes = len(data)
    count_org_success = 0
    count_branched_oracle_success = 0
    count_org_fail = 0
    count_rescue = 0
    
    # === 2. 新增的轨迹级统计 (Trajectory Level) ===
    global_total_trajs = 0
    global_success_trajs = 0
    global_branch_success_lens = [] # 仅统计分叉成功的轨迹长度
    
    # === 3. 用户详情容器 ===
    user_details_list = []

    for episode in data:
        user_id = episode.get('user_id', 'Unknown')
        trajectories = episode.get('trajectories', [])
        
        # 找到 Root
        root_traj = next((t for t in trajectories if t['id'] == 'root'), None)
        if not root_traj: continue
        
        # --- Episode Level Stats ---
        root_is_success = is_traj_success(root_traj)
        
        # 检查是否有任意分叉成功
        any_branch_success = False
        for t in trajectories:
            if t['id'] != 'root' and is_traj_success(t):
                any_branch_success = True
                break
        
        if root_is_success:
            count_org_success += 1
            count_branched_oracle_success += 1 # Root 成功也算 Oracle 成功
        else:
            count_org_fail += 1
            if any_branch_success:
                count_rescue += 1
                count_branched_oracle_success += 1

        # --- Trajectory Level Stats (User Specific) ---
        u_total_trajs = 0
        u_success_trajs = 0
        u_success_lens = []
        u_branches_info = []
        
        # 1. Root Stats
        u_total_trajs += 1
        if root_is_success: u_success_trajs += 1
        
        # 2. Branch Stats
        branch_trajs = [t for t in trajectories if t['id'] != 'root']
        
        for b_traj in branch_trajs:
            u_total_trajs += 1
            b_succ = is_traj_success(b_traj)
            b_len = get_traj_length(b_traj)
            b_turn = b_traj.get('branch_at_turn', -1)
            
            if b_succ:
                u_success_trajs += 1
                u_success_lens.append(b_len)
                global_branch_success_lens.append(b_len)
            
            # 提取策略对比
            old_strat = get_strategy_at_turn(root_traj, b_turn)
            new_strat = get_strategy_at_turn(b_traj, b_turn)
            
            u_branches_info.append({
                "branch_turn": b_turn,
                "old_strategy": old_strat,
                "new_strategy": new_strat,
                "is_success": b_succ,
                "length": b_len
            })

        # 汇总用户数据
        global_total_trajs += u_total_trajs
        global_success_trajs += u_success_trajs
        
        user_details_list.append({
            "user_id": user_id,
            "root_success": root_is_success,
            "traj_stats": {
                "total": u_total_trajs,
                "success": u_success_trajs,
                "success_rate": u_success_trajs / u_total_trajs if u_total_trajs > 0 else 0.0,
                "avg_success_branch_len": float(np.mean(u_success_lens)) if u_success_lens else 0.0
            },
            "branches": u_branches_info
        })

    # === 4. 构建最终 JSON 结构 ===
    
    # 基础指标 (Episode Level)
    episode_metrics = {
        "total_episodes": total_episodes,
        "success_rate": {
            "original": count_org_success / total_episodes if total_episodes else 0,
            "branched_oracle": count_branched_oracle_success / total_episodes if total_episodes else 0,
            "improvement": (count_branched_oracle_success - count_org_success) / total_episodes if total_episodes else 0
        },
        "rescue_rate": count_rescue / count_org_fail if count_org_fail > 0 else 0.0
    }
    
    # 进阶指标 (Trajectory Level)
    trajectory_metrics = {
        "global_total_trajectories": global_total_trajs,
        "global_success_trajectories": global_success_trajs,
        "global_traj_success_rate": global_success_trajs / global_total_trajs if global_total_trajs > 0 else 0.0,
        "global_avg_branch_success_len": float(np.mean(global_branch_success_lens)) if global_branch_success_lens else 0.0
    }
    
    final_output = {
        "summary": {
            "episode_metrics": episode_metrics,
            "trajectory_metrics": trajectory_metrics
        },
        "user_details": user_details_list
    }

    # === 5. 输出与保存 ===
    output_file = input_file.replace(".json", "_eval.json")
    
    print("\n[Analysis Summary]")
    print(json.dumps(final_output["summary"], indent=2))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"\nDetailed evaluation saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    motivation_dir = r"/root/EvolvingAgent-master/EvolvingAgentTest_wym"

    # mode = [
    #     "dissonance",
    #     "state_entropy",
    #     "action_entropy",
    #     "random"
    # ]

    # for m in mode:
    #     exp_file = r"model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0/rollout_v2/results_" + m + "_t_1.0.json"
    #     input_file = os.path.join(motivation_dir, exp_file)
    #     analyze_and_update_eval(input_file=input_file)
    exp_file = r"model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0/rollout_v3_1.0/results_exp_t_1.0.json"
    input_file = os.path.join(motivation_dir, exp_file)
    analyze_and_update_eval(input_file=input_file)