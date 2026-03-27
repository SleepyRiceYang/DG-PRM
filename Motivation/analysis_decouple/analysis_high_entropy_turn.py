import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re
import numpy as np

# === 配置区域 ===
INPUT_FILE = "results_decoupled_2025-11-26T18-46-54.json"  # 替换为你的输入文件
OUTPUT_DIR = "analysis_all_samples_vis_v2"     # 结果输出目录
# =============

# 绘图设置
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def segment_metrics(metrics):
    """切分 Thinking 和 Decision"""
    split_index = -1
    for i, item in enumerate(metrics):
        token = item['token']
        if "Strategy" in token or "strategy" in token:
            look_ahead = metrics[i:i+4]
            look_ahead_str = "".join([m['token'] for m in look_ahead])
            if ":" in look_ahead_str or "：" in look_ahead_str:
                current_idx = i + 1
                while current_idx < len(metrics):
                    next_tok = metrics[current_idx]['token']
                    if re.match(r'^[\s\*[:：]+$', next_tok):
                        current_idx += 1
                    else:
                        split_index = current_idx
                        break
                break
    
    if split_index != -1:
        return metrics[:split_index], metrics[split_index:], True
    else:
        return metrics, [], False

def plot_high_entropy_ratio_trend(data, output_dir):
    """
    绘制每个用户的高熵 Token 比例随轮次变化的折线图，
    并保存包含详细统计信息的 CSV 文件。
    """
    stats_data = []

    print("正在提取并计算详细的高熵统计数据...")

    for session_idx, session in enumerate(data):
        user_id = session.get('user_id', f'User_{session_idx}')
        history = session.get('detailed_history', [])
        
        for turn_idx, turn in enumerate(history):
            if turn['role'] == 'Persuader':
                metrics = turn.get('strategy_entropy_metrics')
                if not metrics: continue
                
                # 1. 只提取思考部分 (Reasoning Phase)
                reasoning_m, _, _ = segment_metrics(metrics)
                
                # 过滤过短的思考
                if len(reasoning_m) < 5:
                    continue
                
                entropies = [m['entropy'] for m in reasoning_m]
                
                # 2. 动态计算阈值
                mean_val = np.mean(entropies)
                std_val = np.std(entropies)
                threshold = max(mean_val + std_val*2, 0.8)
                threshold_formula = "max(Mean + 2*Std, 0.8)" # 记录公式字符串
                
                # 3. 统计高熵 Token 数量
                high_ent_count = sum(1 for e in entropies if e > threshold)
                total_count = len(entropies)
                ratio = high_ent_count / total_count if total_count > 0 else 0
                
                # 4. [修改] 记录更详细的数据
                stats_data.append({
                    "User": user_id,
                    "Turn": turn.get('round', turn_idx + 1),
                    "Strategy": turn.get('strategy', 'Unknown'),
                    
                    # 核心绘图数据
                    "High Entropy Ratio": ratio,
                    
                    # [新增] 详细统计数据
                    "Mean Entropy (Reasoning)": round(mean_val, 4), # 思考过程平均熵
                    "Threshold Value": round(threshold, 4),         # 判定高熵的阈值
                    "Threshold Formula": threshold_formula,         # 阈值计算公式
                    "High Entropy Count": high_ent_count,           # 高熵 Token 数量
                    "Total Token Count": total_count                # 总 Token 数量
                })

    if not stats_data:
        print("未提取到有效数据，无法绘图。")
        return

    # 转换为 DataFrame
    df = pd.DataFrame(stats_data)

    # === 1. 绘图 (保持不变) ===
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df, 
        x="Turn", 
        y="High Entropy Ratio", 
        hue="User", 
        style="User",
        markers=True, 
        dashes=False, 
        linewidth=2.5,
        palette="viridis",
        markersize=8
    )
    plt.title("Trend of High Entropy Token Ratio (Reasoning Phase)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Dialogue Turn", fontsize=12)
    plt.ylabel("High Entropy Ratio", fontsize=12)
    
    from matplotlib.ticker import PercentFormatter
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    max_turn = df["Turn"].max()
    plt.xticks(range(1, int(max_turn) + 1))
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="User ID")
    plt.tight_layout()
    
    img_save_path = os.path.join(output_dir, "trend_high_entropy_ratio_by_user__2std_0.8.png")
    plt.savefig(img_save_path, dpi=300)
    plt.close()
    print(f"✅ 比例变化趋势图已保存至: {img_save_path}")

    # === 2. [修改] 保存详细 CSV ===
    csv_path = os.path.join(output_dir, "high_entropy_detailed_stats_2std_0.8.csv")
    
    # 调整列顺序，让表格更好看
    columns_order = [
        "User", "Turn", "Strategy", 
        "High Entropy Ratio", 
        "High Entropy Count", "Total Token Count", 
        "Mean Entropy (Reasoning)", "Threshold Value", "Threshold Formula"
    ]
    # 确保只保存存在的列 (防止某些极端情况列缺失)
    existing_cols = [c for c in columns_order if c in df.columns]
    
    df[existing_cols].to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 详细数据表 CSV 已保存至: {csv_path}")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading Data...")
    data = load_data(INPUT_FILE)
    
    print("Generating Analysis...")
    plot_high_entropy_ratio_trend(data, OUTPUT_DIR)
    
    print("\nDone!")

if __name__ == "__main__":
    main()