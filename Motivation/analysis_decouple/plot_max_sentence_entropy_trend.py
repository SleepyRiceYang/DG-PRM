import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re

# === 配置区域 ===
INPUT_FILE = "results_decoupled_2025-11-29T11-02-41.json"  # 替换为你的输入文件
OUTPUT_DIR = "analysis_sentence_flow"   # 结果输出目录
# =============

# 绘图设置
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def segment_metrics(metrics):
    """只提取思考过程"""
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
        return metrics[:split_index]
    else:
        return metrics

def split_into_sentences(metrics):
    """断句"""
    sentences = []
    current_tokens = []
    current_entropies = []
    current_text_buffer = ""
    
    for m in metrics:
        token = m['token']
        entropy = m['entropy']
        
        current_tokens.append(token)
        current_entropies.append(entropy)
        current_text_buffer += token
        
        is_end = False
        if '\n' in token:
            is_end = True
        elif '.' in token or '?' in token or '!' in token:
            is_end = True
            
        if is_end:
            if len(current_text_buffer.strip()) > 1:
                sentences.append({
                    "text": current_text_buffer.strip(),
                    "entropies": current_entropies,
                })
                current_tokens = []
                current_entropies = []
                current_text_buffer = ""
            elif len(current_text_buffer.strip()) == 0:
                current_tokens = []
                current_entropies = []
                current_text_buffer = ""

    if len(current_text_buffer.strip()) > 1:
        sentences.append({
            "text": current_text_buffer.strip(),
            "entropies": current_entropies,
        })
    return sentences

def extract_trend_data(data):
    """
    提取绘图所需数据：
    [User_ID, Turn, Peak_Entropy]
    """
    records = []
    
    print("正在处理数据...")
    for session_idx, session in enumerate(data):
        user_id = session.get('user_id', f'User_{session_idx}')
        history = session.get('detailed_history', [])
        
        for turn_idx, turn in enumerate(history):
            if turn['role'] == 'Persuader':
                metrics = turn.get('strategy_entropy_metrics')
                if not metrics: continue
                
                # 1. 提取思考部分
                reasoning_m = segment_metrics(metrics)
                if len(reasoning_m) < 5: continue
                
                # 2. 断句
                sentences = split_into_sentences(reasoning_m)
                
                # 3. 删除最后一句 (Refined Logic)
                if len(sentences) > 0:
                    sentences = sentences[:-1]
                
                # 如果删完没剩下了，跳过
                if not sentences:
                    continue
                
                # 4. 寻找本轮的最大均值熵
                # 计算每一句的平均熵
                means = [np.mean(s['entropies']) for s in sentences]
                if not means: continue
                
                # 取最大值
                max_mean_entropy = max(means)
                
                records.append({
                    "User": user_id,
                    "Turn": turn.get('round', turn_idx + 1),
                    "Max Sentence Entropy": max_mean_entropy
                })
    
    return pd.DataFrame(records)

def plot_trend_chart(df, output_dir):
    """绘制三条折线"""
    if df.empty:
        print("无数据可绘图")
        return

    plt.figure(figsize=(12, 6))
    
    # 绘制折线图
    # hue="User": 自动为不同 User 分配不同颜色，生成多条线
    # style="User": 自动为不同 User 分配不同标记点(Marker)，方便区分
    sns.lineplot(
        data=df, 
        x="Turn", 
        y="Max Sentence Entropy", 
        hue="User", 
        style="User",
        markers=True, 
        dashes=False,   # 实线
        linewidth=2.5,  # 线宽
        palette="viridis", # 配色方案
        markersize=9
    )

    # 标题与标签
    plt.title("Trend of Peak Thinking Difficulty per Turn\n(Max Average Entropy among sentences, excluding last sentence)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Dialogue Turn", fontsize=12)
    plt.ylabel("Max Sentence Entropy", fontsize=12)
    
    # 设置X轴为整数刻度
    max_turn = df["Turn"].max()
    plt.xticks(range(1, int(max_turn) + 1))
    
    # 网格线
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 图例放在外侧
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="User ID")
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "trend_max_sentence_entropy_29_1.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 趋势对比图已保存至: {save_path}")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading Data...")
    data = load_data(INPUT_FILE)
    
    print("Extracting Trend Data...")
    df = extract_trend_data(data)
    
    print(f"Extracted {len(df)} points. Plotting...")
    plot_trend_chart(df, OUTPUT_DIR)
    
    # 保存 CSV 备查
    csv_path = os.path.join(OUTPUT_DIR, "trend_max_sentence_entropy_data_29_1.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ 数据源 CSV 已保存至: {csv_path}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()