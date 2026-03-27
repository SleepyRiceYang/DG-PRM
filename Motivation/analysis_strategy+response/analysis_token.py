import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import os
from collections import Counter
import argparse  # <--- 新增导入

# 尝试导入 WordCloud
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

# === Configuration ===

OUTPUT_DIR = "analysis_token"      # 输出目录
# =====================

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def segment_metrics_new_format(metrics):
    """
    根据新格式切分 Reasoning 和 Response 部分
    格式: <reasoning>...</reasoning><strategy>...</strategy><response>...</response>
    """
    # 将所有token组合成文本，方便查找
    tokens_text = "".join([m['token'] for m in metrics])
    
    # 查找各部分的位置
    reasoning_start = tokens_text.find("<reasoning>")
    reasoning_end = tokens_text.find("</reasoning>")
    
    response_start = tokens_text.find("<response>")
    response_end = tokens_text.find("</response>")
    
    reasoning_part = []
    response_part = []
    
    if reasoning_start != -1 and reasoning_end != -1:
        # 计算 reasoning 部分在 metrics 中的索引范围
        start_index = 0
        current_pos = 0
        for i, item in enumerate(metrics):
            if current_pos <= reasoning_start + len("<reasoning>") < current_pos + len(item['token']):
                start_index = i + 1 if current_pos + len(item['token']) > reasoning_start + len("<reasoning>") else i
                break
            current_pos += len(item['token'])
        
        end_index = 0
        current_pos = 0
        for i, item in enumerate(metrics):
            if current_pos <= reasoning_end < current_pos + len(item['token']):
                end_index = i
                break
            current_pos += len(item['token'])
        
        if start_index < end_index:
            reasoning_part = metrics[start_index:end_index]
    
    if response_start != -1 and response_end != -1:
        # 计算 response 部分在 metrics 中的索引范围
        start_index = 0
        current_pos = 0
        for i, item in enumerate(metrics):
            if current_pos <= response_start + len("<response>") < current_pos + len(item['token']):
                start_index = i + 1 if current_pos + len(item['token']) > response_start + len("<response>") else i
                break
            current_pos += len(item['token'])
        
        end_index = 0
        current_pos = 0
        for i, item in enumerate(metrics):
            if current_pos <= response_end < current_pos + len(item['token']):
                end_index = i
                break
            current_pos += len(item['token'])
        
        if start_index < end_index:
            response_part = metrics[start_index:end_index]
    
    return reasoning_part, response_part

def collect_entropy_values_by_turn(data):
    """
    遍历所有数据，收集每个用户在每个轮次中 reasoning 和 response 阶段的平均熵值。
    返回一个 DataFrame，包含 ['User', 'Turn', 'Phase', 'Mean_Entropy'] 列
    """
    all_records = []

    for session in data:
        user_id = session.get('user_id', 'Unknown_User')
        history = session.get('detailed_history', [])
        
        for turn in history:
            if turn['role'] == 'Persuader':
                metrics = turn.get('metrics')  # 新格式使用 'metrics' 字段
                if not metrics:
                    continue

                # 切分阶段
                reasoning_m, response_m = segment_metrics_new_format(metrics)
                
                turn_num = turn.get('round', 0)

                # 计算 Reasoning 阶段的平均熵
                if reasoning_m:
                    reasoning_entropies = [m['entropy'] for m in reasoning_m]
                    if reasoning_entropies:
                        reasoning_mean = np.mean(reasoning_entropies)
                        all_records.append({
                            "User": user_id,
                            "Turn": turn_num,
                            "Phase": "Reasoning",
                            "Mean_Entropy": reasoning_mean
                        })

                # 计算 Response 阶段的平均熵
                if response_m:
                    response_entropies = [m['entropy'] for m in response_m]
                    if response_entropies:
                        response_mean = np.mean(response_entropies)
                        all_records.append({
                            "User": user_id,
                            "Turn": turn_num,
                            "Phase": "Response",
                            "Mean_Entropy": response_mean
                        })

    return pd.DataFrame(all_records)

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
                metrics = turn.get('metrics')  # 新格式使用 'metrics' 字段
                if not metrics:
                    continue

                # 切分阶段
                reasoning_m, response_m = segment_metrics_new_format(metrics)

                # 收集 Reasoning 阶段的熵
                for m in reasoning_m:
                    all_records.append({
                        "Entropy": m['entropy'],
                        "Phase": "Reasoning"
                    })

                # 收集 Response 阶段的熵
                for m in response_m:
                    all_records.append({
                        "Entropy": m['entropy'],
                        "Phase": "Response"
                    })

    return pd.DataFrame(all_records)

def plot_users_subplots_entropy_trend(df, output_dir,remark=""):
    """
    将四个用户的 reasoning 和 response 阶段平均熵随轮次变化的趋势绘制在2x2的子图中
    """
    if df.empty:
        print("警告：没有数据可用于绘制子图趋势图")
        return

    # 获取所有唯一用户
    users = df['User'].unique()
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 为每个用户分配颜色
    colors = plt.cm.Set1(np.linspace(0, 1, len(users)))
    
    # 为每个用户绘制子图
    for i, user in enumerate(users):
        if i >= 4:  # 最多绘制4个用户
            break
            
        ax = axes[i]
        user_data = df[df['User'] == user]
        
        # 绘制 reasoning 趋势
        reasoning_data = user_data[user_data['Phase'] == 'Reasoning']
        if not reasoning_data.empty:
            ax.plot(reasoning_data['Turn'], reasoning_data['Mean_Entropy'], 
                   marker='o', linewidth=2, color='#1f77b4', alpha=0.8,
                   label='Reasoning')
        
        # 绘制 response 趋势
        response_data = user_data[user_data['Phase'] == 'Response']
        if not response_data.empty:
            ax.plot(response_data['Turn'], response_data['Mean_Entropy'], 
                   marker='s', linewidth=2, color='#ff7f0e', alpha=0.8, linestyle='--',
                   label='Response')
        
        ax.set_title(f"User: {user}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Turn", fontsize=12)
        ax.set_ylabel("Mean Entropy", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 设置 x 轴为整数
        max_turn = max(user_data['Turn']) if not user_data.empty else 1
        ax.set_xticks(range(1, max_turn + 1))
    
    # 隐藏多余的子图（如果用户少于4个）
    for i in range(len(users), 4):
        axes[i].set_visible(False)
    
    plt.suptitle("Average Token Entropy Trend - All Users (2x2 Subplots)", fontsize=16, y=0.95)
    plt.tight_layout()
    
    suffix = f"_{remark}" if remark else ""
    save_path = os.path.join(output_dir, f"entropy_trend_users{suffix}.png") # <--- 修改文件名
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 2x2子图趋势图已保存至: {save_path}")

def plot_all_users_entropy_trend(df, output_dir,remark=""):
    """
    将所有用户的 reasoning 和 response 阶段平均熵随轮次变化的趋势绘制在同一张图上
    """
    if df.empty:
        print("警告：没有数据可用于绘制综合趋势图")
        return

    plt.figure(figsize=(15, 10))
    
    # 获取所有唯一用户
    users = df['User'].unique()
    
    # 为每个用户分配颜色
    colors = plt.cm.Set1(np.linspace(0, 1, len(users)))
    
    # 分别绘制每个用户的 reasoning 和 response 趋势
    for i, user in enumerate(users):
        user_data = df[df['User'] == user]
        color = colors[i]
        
        # 绘制 reasoning 趋势
        reasoning_data = user_data[user_data['Phase'] == 'Reasoning']
        if not reasoning_data.empty:
            plt.plot(reasoning_data['Turn'], reasoning_data['Mean_Entropy'], 
                    marker='o', linewidth=2, color=color, alpha=0.7,
                    label=f'{user} - Reasoning')
        
        # 绘制 response 趋势
        response_data = user_data[user_data['Phase'] == 'Response']
        if not response_data.empty:
            plt.plot(response_data['Turn'], response_data['Mean_Entropy'], 
                    marker='s', linewidth=2, color=color, alpha=0.7, linestyle='--',
                    label=f'{user} - Response')
    
    plt.title("Average Token Entropy Trend - All Users", fontsize=16)
    plt.xlabel("Turn", fontsize=14)
    plt.ylabel("Mean Entropy", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 设置 x 轴为整数
    max_turn = max(df['Turn']) if not df.empty else 1
    plt.xticks(range(1, max_turn + 1))
    
    plt.tight_layout()
    suffix = f"_{remark}" if remark else ""
    save_path = os.path.join(output_dir, f"entropy_trend_all_users{suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 所有用户的熵值趋势图已保存至: {save_path}")

def plot_reasoning_phase_trend(df, output_dir,remark=""):
    """
    绘制所有用户的 reasoning 阶段平均熵随轮次变化的趋势图
    """
    if df.empty:
        print("警告：没有数据可用于绘制 reasoning 阶段趋势图")
        return

    plt.figure(figsize=(12, 8))
    
    # 获取所有唯一用户
    users = df['User'].unique()
    
    # 为每个用户分配颜色
    colors = plt.cm.Set1(np.linspace(0, 1, len(users)))
    
    # 只绘制 reasoning 阶段的趋势
    reasoning_data = df[df['Phase'] == 'Reasoning']
    
    for i, user in enumerate(users):
        user_data = reasoning_data[reasoning_data['User'] == user]
        color = colors[i]
        
        if not user_data.empty:
            plt.plot(user_data['Turn'], user_data['Mean_Entropy'], 
                    marker='o', linewidth=2, color=color, alpha=0.7,
                    label=f'{user}')
    
    plt.title("Average Token Entropy Trend - Reasoning Phase", fontsize=16)
    plt.xlabel("Turn", fontsize=14)
    plt.ylabel("Mean Entropy", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 设置 x 轴为整数
    max_turn = max(reasoning_data['Turn']) if not reasoning_data.empty else 1
    plt.xticks(range(1, max_turn + 1))
    
    suffix = f"_{remark}" if remark else ""
    save_path = os.path.join(output_dir, f"entropy_trend_reasoning_phase{suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Reasoning 阶段熵值趋势图已保存至: {save_path}")

def plot_response_phase_trend(df, output_dir,remark=""):
    """
    绘制所有用户的 response 阶段平均熵随轮次变化的趋势图
    """
    if df.empty:
        print("警告：没有数据可用于绘制 response 阶段趋势图")
        return

    plt.figure(figsize=(12, 8))
    
    # 获取所有唯一用户
    users = df['User'].unique()
    
    # 为每个用户分配颜色
    colors = plt.cm.Set1(np.linspace(0, 1, len(users)))
    
    # 只绘制 response 阶段的趋势
    response_data = df[df['Phase'] == 'Response']
    
    for i, user in enumerate(users):
        user_data = response_data[response_data['User'] == user]
        color = colors[i]
        
        if not user_data.empty:
            plt.plot(user_data['Turn'], user_data['Mean_Entropy'], 
                    marker='s', linewidth=2, color=color, alpha=0.7,
                    label=f'{user}')
    
    plt.title("Average Token Entropy Trend - Response Phase", fontsize=16)
    plt.xlabel("Turn", fontsize=14)
    plt.ylabel("Mean Entropy", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 设置 x 轴为整数
    max_turn = max(response_data['Turn']) if not response_data.empty else 1
    plt.xticks(range(1, max_turn + 1))
    
    suffix = f"_{remark}" if remark else ""
    save_path = os.path.join(output_dir, f"entropy_trend_response_phase{suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Response 阶段熵值趋势图已保存至: {save_path}")

def plot_entropy_distribution(entropy_df, output_dir,remark=""):
    """
    绘制所有 Token 的熵值分布图 (Histogram + KDE)
    区分 Reasoning 和 Response 阶段
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
    mean_resp = entropy_df[entropy_df['Phase']=='Response']['Entropy'].mean()

    plt.title(f"Distribution of Token Entropy Values (All Tokens)\nReasoning Mean: {mean_res:.3f} | Response Mean: {mean_resp:.3f}", fontsize=14)
    plt.xlabel("Entropy Value (Higher = More Uncertain)", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    suffix = f"_{remark}" if remark else ""
    save_path = os.path.join(output_dir, f"dist_all_token_entropy{suffix}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 全量熵值分布图已保存至: {save_path}")

# === 重写 Main 函数 ===
def main():
    parser = argparse.ArgumentParser(description="Token Entropy Analysis")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--remark", default="", help="Remark suffix for filenames")
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output_dir
    remark = args.remark

    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        print("Loading data...")
        data = load_data(input_file)

        print("Collecting entropy values by turn...")
        df_turn_entropy = collect_entropy_values_by_turn(data)

        print("Collecting raw entropy values...")
        df_raw_tokens = collect_entropy_values(data)

        if not df_turn_entropy.empty:
            plot_users_subplots_entropy_trend(df_turn_entropy, output_dir, remark)
            plot_all_users_entropy_trend(df_turn_entropy, output_dir, remark)
            plot_reasoning_phase_trend(df_turn_entropy, output_dir, remark)
            plot_response_phase_trend(df_turn_entropy, output_dir, remark)

        if not df_raw_tokens.empty:
            plot_entropy_distribution(df_raw_tokens, output_dir, remark)

        print(f"\nAll analysis saved to: {output_dir}")

    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()