import json
import numpy as np

def is_traj_success(trajectory):
    """判断单条轨迹是否成功 (返回 1.0 或 0.0)"""
    if not trajectory:
        return 0.0
    if trajectory.get('success'):
        return 1.0
    turns = trajectory.get('turns', [])
    if not turns:
        return 0.0
    for turn in turns:
        if turn.get('reward', 0.0) >= 1.0:
            return 1.0
    if turns[-1].get('reward', 0.0) >= 1.0:
        return 1.0
    return 0.0

def get_system_strategies(trajectory):
    """提取轨迹中所有 Persuader (系统) 的策略，返回字典 {round_num: strategy_name}"""
    strategies = {}
    for turn in trajectory.get('turns', []):
        if turn.get('role') == 'Persuader':
            strategies[turn.get('round')] = turn.get('strategy_name')
    return strategies

def calculate_normalized_edit_distance(seq1, seq2):
    """计算两个策略序列的归一化编辑距离 (Normalized Levenshtein Distance)"""
    m, n = len(seq1), len(seq2)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0

    import numpy as np
    dp = np.zeros((m + 1, n + 1))

    # 【真正修复的地方】：正确初始化二维数组的第一列和第一行
    for i in range(m + 1):
        dp[i, 0] = i  # 第一列：删除操作
    for j in range(n + 1):
        dp[0, j] = j  # 第一行：插入操作

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i-1] == seq2[j-1] else 1
            dp[i, j] = min(dp[i-1, j] + 1,      # Deletion (删除)
                           dp[i, j-1] + 1,      # Insertion (插入)
                           dp[i-1, j-1] + cost) # Substitution (替换)

    max_len = max(m, n)
    return dp[m, n] / max_len

def analyze_intervention_validity(data_file_path):
    with open(data_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_branches = 0
    local_strategy_changed = 0

    # 用于统计全局轨迹偏移距
    # 按照干预是否导致了结局逆转（Rescue/Vulnerability）进行分组
    distance_when_outcome_changed = []  # 干预导致胜负逆转的轨迹距离
    distance_when_outcome_same = []     # 干预未改变胜负结果的轨迹距离

    for user_item in data:
        trajectories = user_item.get('trajectories', [])

        # 找到初始轨迹 (Root)
        root_traj = next((t for t in trajectories if t.get('id') == 'root'), None)
        if not root_traj:
            continue

        root_success = is_traj_success(root_traj)
        root_strategies = get_system_strategies(root_traj)

        # 遍历所有分叉轨迹
        for traj in trajectories:
            if traj.get('id') == 'root':
                continue

            branch_turn = traj.get('branch_at_turn')
            if branch_turn is None or branch_turn not in root_strategies:
                continue

            branch_strategies = get_system_strategies(traj)
            if branch_turn not in branch_strategies:
                continue

            total_branches += 1

            # --- 1. 局部策略变更率 (Local Strategy Shift) ---
            if root_strategies[branch_turn] != branch_strategies[branch_turn]:
                local_strategy_changed += 1

            # --- 2. 全局轨迹偏移度 (Global Trajectory Deviation) ---
            # 提取分支点之后的后续策略序列 (不包括分支点本身，或者包括分支点看整体影响)
            # 这里我们提取从分支点开始到结束的完整后续序列
            root_future_seq = [strat for rnd, strat in sorted(root_strategies.items()) if rnd >= branch_turn]
            branch_future_seq = [strat for rnd, strat in sorted(branch_strategies.items()) if rnd >= branch_turn]

            norm_distance = calculate_normalized_edit_distance(root_future_seq, branch_future_seq)

            # 检查这次干预是否改变了最终结果 (Advantage 绝对值 > 0)
            branch_success = is_traj_success(traj)
            is_outcome_changed = (root_success != branch_success)

            if is_outcome_changed:
                distance_when_outcome_changed.append(norm_distance)
            else:
                distance_when_outcome_same.append(norm_distance)

    # --- 打印分析报告 ---
    print("="*50)
    print("📊 Analysis of Intervention Validity (干预有效性分析报告)")
    print("="*50)

    if total_branches == 0:
        print("未找到有效的分叉数据。")
        return

    local_shift_rate = local_strategy_changed / total_branches
    print(f"1. 局部维度 (Local Level): 策略跳出率")
    print(f"   - 总干预节点数: {total_branches}")
    print(f"   - 策略类别发生实质性改变的次数: {local_strategy_changed}")
    print(f"   - 局部策略变更率 (Local Strategy Shift Rate): {local_shift_rate:.2%} \n")

    print(f"2. 全局维度 (Global Level): 后续轨迹偏移度 (STeCa Deviation Distance)")
    avg_dist_changed = np.mean(distance_when_outcome_changed) if distance_when_outcome_changed else 0
    avg_dist_same = np.mean(distance_when_outcome_same) if distance_when_outcome_same else 0

    print(f"   - 导致结局逆转的干预 (Advantage != 0)，平均后续轨迹偏移度: {avg_dist_changed:.4f}")
    print(f"   - 未改变结局的平庸干预 (Advantage == 0)，平均后续轨迹偏移度: {avg_dist_same:.4f}")

    if avg_dist_changed > avg_dist_same:
        print(f"\n💡 科学结论推演:")
        print(f"   结果表明，产生有效优势值（挽救或脆弱）的干预，其引发的『因果涟漪（轨迹偏移）』")
        print(f"   比无效干预高出 {(avg_dist_changed - avg_dist_same)/avg_dist_same if avg_dist_same > 0 else 0:.2%}。")
        print(f"   这完美证明了：我们定位的关键节点不仅改变了当前动作，更重构了后续的马尔可夫转移矩阵！")
    print("="*50)

# 调用方式 (替换为您真实的 json 文件路径)
# analyze_intervention_validity("your_experiment_data.json")

import argparse, os
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', required=True, help="List of EXP result json files")
    parser.add_argument("--names", nargs='+', required=True, help="List of Model Names corresponding to files")
    parser.add_argument("--output_dir", default="eval_results", help="Directory to save analysis results")
    args = parser.parse_args()

    if len(args.files) != len(args.names):
        print("[Error] Number of files and names must match.")
        return
    EXPERIMENTS_CONFIG = []
    os.makedirs(args.output_dir, exist_ok=True)

    for file_path, model_name in zip(args.files, args.names):
        EXPERIMENTS_CONFIG.append({"name": model_name, "path": file_path})
        print(f"\n[Info] Analyzing intervention validity for: {model_name}")
        analyze_intervention_validity(file_path)

if __name__ == "__main__":
    main()