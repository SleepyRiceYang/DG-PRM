import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import os
from collections import Counter

# 尝试导入 WordCloud
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# === Configuration ===
INPUT_FILE = "results_decoupled_2025-11-26T18-46-54.json"   # 替换为你的文件路径
OUTPUT_DIR = "analysis"      # 输出目录
# =====================

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def segment_metrics(metrics):
    """根据 Markdown 格式切分 Reasoning 和 Decision"""
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

# ==========================================
# [新增] 1. 专门用于收集所有 Token 熵值的函数
# ==========================================
def collect_entropy_values(data):
    """
    遍历所有数据，收集每一个 Token 的熵值。
    返回一个 DataFrame，包含 ['Entropy', 'Phase'] 列
    """
    all_records = []

    for session in data:
        history = session.get('detailed_history', [])
        for turn in history:
            if turn['role'] == 'Persuader':
                metrics = turn.get('strategy_entropy_metrics')
                if not metrics:
                    continue

                # 切分阶段
                reasoning_m, decision_m, _ = segment_metrics(metrics)

                # 收集 Reasoning 阶段的熵
                for m in reasoning_m:
                    all_records.append({
                        "Entropy": m['entropy'],
                        "Phase": "Reasoning"
                    })

                # 收集 Decision 阶段的熵
                for m in decision_m:
                    all_records.append({
                        "Entropy": m['entropy'],
                        "Phase": "Decision"
                    })

    return pd.DataFrame(all_records)

def analyze_entropy_details(data):
    # (此函数保持不变，用于生成 Turn 级别的统计)
    records = []
    for session_idx, session in enumerate(data):
        user_id = session.get('user_id', f'User_{session_idx}')
        history = session.get('detailed_history', [])
        for turn_idx, turn in enumerate(history):
            if turn['role'] == 'Persuader':
                metrics = turn.get('strategy_entropy_metrics')
                if not metrics: continue

                reasoning_m, decision_m, segmented = segment_metrics(metrics)

                all_entropies = [m['entropy'] for m in metrics]
                turn_mean = np.mean(all_entropies) if all_entropies else 0
                reasoning_mean = np.mean([m['entropy'] for m in reasoning_m]) if reasoning_m else 0
                decision_mean = np.mean([m['entropy'] for m in decision_m]) if decision_m else 0

                high_50_tokens = []
                high_70_tokens = []
                threshold_50 = turn_mean * 1.5
                threshold_70 = turn_mean * 1.7

                for m in metrics:
                    if m['entropy'] > threshold_50:
                        token_info = {
                            "token": m['token'],
                            "entropy": m['entropy'],
                            "prob": m.get('top_5', [{}])[0].get('prob', 0),
                            "is_decision_phase": (m in decision_m)
                        }
                        high_50_tokens.append(token_info)
                        if m['entropy'] > threshold_70:
                            high_70_tokens.append(token_info)

                records.append({
                    "user_id": user_id,
                    "turn": turn.get('round', turn_idx),
                    "strategy_name": turn.get('strategy', 'Unknown'),
                    "global_mean": turn_mean,
                    "reasoning_mean": reasoning_mean,
                    "decision_mean": decision_mean,
                    "segmented_success": segmented,
                    "high_50_tokens": high_50_tokens,
                    "high_70_tokens": high_70_tokens
                })
    return pd.DataFrame(records)

# ... (save_text_report 函数保持不变，省略) ...

def save_text_report(df, output_dir):
    # 为了代码完整性，简单保留核心逻辑
    file_path = os.path.join(output_dir, "detailed_token_analysis.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Dimension 2 Analysis Report\n")
        # ... (内容同上一个版本) ...

# ==========================================
# [新增] 2. 绘制全量熵值分布直方图
# ==========================================
def plot_entropy_distribution(entropy_df, output_dir):
    """
    绘制所有 Token 的熵值分布图 (Histogram + KDE)
    区分 Reasoning 和 Decision 阶段
    """
    if entropy_df.empty:
        return

    plt.figure(figsize=(12, 7))

    # 绘制直方图和密度曲线
    # common_norm=False 让两个类别的密度独立计算，方便对比形状
    sns.histplot(
        data=entropy_df,
        x="Entropy",
        hue="Phase",
        kde=True,
        element="step",
        stat="density",
        common_norm=False,
        palette="viridis",
        alpha=0.3
    )

    # 计算一些统计数据放在标题或图注里
    mean_res = entropy_df[entropy_df['Phase']=='Reasoning']['Entropy'].mean()
    mean_dec = entropy_df[entropy_df['Phase']=='Decision']['Entropy'].mean()

    plt.title(f"Distribution of Token Entropy Values (All Tokens)\nReasoning Mean: {mean_res:.3f} | Decision Mean: {mean_dec:.3f}", fontsize=14)
    plt.xlabel("Entropy Value (Higher = More Uncertain)", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    save_path = os.path.join(output_dir, "dist_all_token_entropy.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 全量熵值分布图已保存至: {save_path}")

def plot_basic_charts(df, output_dir):
    if df.empty: return
    # Boxplot
    plt.figure(figsize=(14, 8))
    df_melted = df.melt(id_vars=["turn", "strategy_name"], value_vars=["reasoning_mean", "decision_mean"], var_name="Phase", value_name="Entropy")
    df_melted['Phase'] = df_melted['Phase'].replace({'reasoning_mean': 'Reasoning Phase', 'decision_mean': 'Decision Phase'})
    sns.boxplot(data=df_melted, x="strategy_name", y="Entropy", hue="Phase", palette="Set2")
    plt.xticks(rotation=45, ha='right')
    plt.title("Entropy Distribution by Strategy", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot_reasoning_vs_decision.png"), dpi=300)
    plt.close()

    # Lineplot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="turn", y="global_mean", label="Global Mean Entropy", marker="o", errorbar=None)
    sns.lineplot(data=df, x="turn", y="decision_mean", label="Decision Phase Entropy", marker="x", linestyle="--", errorbar=None)
    plt.title("Entropy Trend over Dialogue Turns", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trend_entropy_over_turns.png"), dpi=300)
    plt.close()

def plot_token_visuals(df, output_dir):
    if df.empty: return
    all_tokens = []
    for tokens_list in df['high_50_tokens']:
        for item in tokens_list:
            raw_token = item['token'].strip()
            if len(raw_token) > 1 or raw_token.isalnum():
                all_tokens.append(raw_token)

    if not all_tokens: return
    token_counts = Counter(all_tokens)

    # Word Cloud
    if HAS_WORDCLOUD:
        wc = WordCloud(width=1600, height=800, background_color='white', colormap='magma').generate_from_frequencies(token_counts)
        plt.figure(figsize=(16, 8))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"High Entropy Token Cloud", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "wordcloud_high_entropy_tokens.png"), dpi=300)
        plt.close()

    # Bar Chart
    top_20 = token_counts.most_common(20)
    if top_20:
        words, counts = zip(*top_20)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=list(counts), y=list(words), palette="viridis")
        plt.title("Top 20 Frequent High-Entropy Tokens", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "barchart_top_entropy_tokens.png"), dpi=300)
        plt.close()

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        print("Loading data...")
        data = load_data(INPUT_FILE)

        # 1. 提取 Turn 级别的统计 (用于箱线图、趋势图、Token分析)
        df_turns = analyze_entropy_details(data)

        # 2. [新增] 提取全量 Token 熵值 (用于分布直方图)
        print("Collecting raw entropy values...")
        df_raw_tokens = collect_entropy_values(data)

        # --- Generating Outputs ---

        # A. 全量分布图 (新增功能)
        if not df_raw_tokens.empty:
            plot_entropy_distribution(df_raw_tokens, OUTPUT_DIR)

        if not df_turns.empty:
            # B. 文本报告
            save_text_report(df_turns, OUTPUT_DIR)
            # C. 基础统计图 (箱线图、趋势图)
            plot_basic_charts(df_turns, OUTPUT_DIR)
            # D. Token 词云与条形图
            plot_token_visuals(df_turns, OUTPUT_DIR)

        print(f"\nAll analysis saved to: {OUTPUT_DIR}")

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()