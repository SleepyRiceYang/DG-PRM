import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
    """只提取思考过程 (Reasoning Phase)"""
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
    """基于 Token 流的断句算法"""
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
        
        # 简单的断句逻辑
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

    # 处理缓冲区剩余内容
    if len(current_text_buffer.strip()) > 1:
        sentences.append({
            "text": current_text_buffer.strip(),
            "entropies": current_entropies,
        })
    return sentences

def process_data(data):
    """处理数据：断句 -> 删除最后一句 -> 统计"""
    detailed_logs = []  # 用于保存 CSV
    turn_max_means = [] # 用于保存每轮的最大均值 (Peak Difficulty)
    plot_data_list = [] # 用于画 PDF
    
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
                
                # === 3. 关键步骤：删除最后一句 ===
                original_count = len(sentences)
                if original_count > 0:
                    sentences = sentences[:-1] # 切片删除最后一个
                
                # 如果删除后没有句子了（原话太短），则跳过
                if not sentences:
                    continue
                
                # 4. 统计剩余句子
                turn_id = f"{user_id}_Turn{turn.get('round', turn_idx+1)}"
                strategy = turn.get('strategy', 'Unknown')
                
                current_turn_means = [] # 记录这一轮里每句话的均值
                sentences_info_for_plot = []
                
                for s_idx, sent in enumerate(sentences):
                    # 计算单句统计
                    mean_ent = np.mean(sent['entropies'])
                    max_ent = np.max(sent['entropies'])
                    min_ent = np.min(sent['entropies'])
                    std_ent = np.std(sent['entropies'])
                    
                    current_turn_means.append(mean_ent)
                    
                    # 收集详细日志
                    detailed_logs.append({
                        "Sample_ID": turn_id,
                        "Strategy": strategy,
                        "Sentence_Index": s_idx + 1, # 1-based index
                        "Mean_Entropy": mean_ent,
                        "Max_Entropy": max_ent,
                        "Std_Entropy": std_ent,
                        "Text_Content": sent['text'].replace('\n', ' ').strip()
                    })
                    
                    # 收集绘图数据
                    sentences_info_for_plot.append({
                        "idx": s_idx + 1,
                        "mean": mean_ent,
                        "std": std_ent,
                        "text": sent['text']
                    })
                
                # 5. 记录本轮的“峰值难度”（所有剩余句子中，平均熵最大的那个值）
                if current_turn_means:
                    peak_difficulty = max(current_turn_means)
                    turn_max_means.append({
                        "Sample_ID": turn_id,
                        "Strategy": strategy,
                        "Peak_Mean_Entropy": peak_difficulty
                    })
                
                plot_data_list.append({
                    "id": turn_id,
                    "strategy": strategy,
                    "sentences": sentences_info_for_plot
                })

    return pd.DataFrame(detailed_logs), pd.DataFrame(turn_max_means), plot_data_list

def save_detailed_csv(df, output_dir):
    """保存所有句子的详细信息"""
    path = os.path.join(output_dir, "detailed_sentence_log_29_1.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"✅ 详细句子日志已保存至: {path}")

def plot_max_entropy_distribution(df_max, output_dir):
    """绘制每轮最大句子熵值的分布图"""
    if df_max.empty: return
    
    plt.figure(figsize=(10, 6))
    
    # 直方图 + 核密度估计
    sns.histplot(data=df_max, x="Peak_Mean_Entropy", kde=True, color="#d62728", bins=15)
    
    mean_val = df_max["Peak_Mean_Entropy"].mean()
    plt.axvline(mean_val, color='blue', linestyle='--', label=f'Avg Peak: {mean_val:.2f}')
    
    plt.title("Distribution of Peak Sentence Entropy per Turn\n(Excluding the final sentence)", fontsize=14)
    plt.xlabel("Max Mean Entropy of a Sentence", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    path = os.path.join(output_dir, "peak_entropy_distribution_29_1.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ 峰值分布图已保存至: {path}")




import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator  # 新增导入：用于控制坐标轴刻度为整数


def save_flow_pdf(plot_data, output_dir):
    """绘制流图 PDF，并统一Y轴上限为1，X轴只显示整数"""
    if not plot_data:
        return
    
    MAX_TEXT_LENGTH = 15  # 控制峰值标注的最大字符数
    path = os.path.join(output_dir, "refined_sentence_flow_29_1.pdf")
    print("正在生成流图 PDF...")

    with PdfPages(path) as pdf:
        plots_per_page = 9
        rows, cols = 3, 3

        for start in range(0, len(plot_data), plots_per_page):
            fig, axes = plt.subplots(rows, cols, figsize=(24, 15))
            axes = axes.flatten()
            batch = plot_data[start : start + plots_per_page]

            for i in range(rows * cols):
                ax = axes[i]
                if i < len(batch):
                    item = batch[i]
                    # 安全校验 key 存在且类型正确
                    if not isinstance(item, dict) or 'sentences' not in item:
                        ax.axis('off')
                        continue
                    
                    sents = item['sentences']
                    if not isinstance(sents, list):
                        ax.axis('off')
                        continue

                    x = [s.get('idx') for s in sents]
                    y = [s.get('mean') for s in sents]
                    yerr = [s.get('std') for s in sents]

                    # 过滤 None / NaN 数据
                    valid_indices = [j for j in range(len(y)) if y[j] is not None and not np.isnan(y[j])]
                    if not valid_indices:
                        ax.axis('off')
                        continue

                    filtered_x = [x[j] for j in valid_indices]
                    filtered_y = [y[j] for j in valid_indices]
                    filtered_yerr = [yerr[j] for j in valid_indices]

                    # 绘制折线
                    ax.errorbar(filtered_x, filtered_y, yerr=filtered_yerr, fmt='-o', capsize=3,
                                color='#1f77b4', ecolor='lightgray', markersize=4)

                    # 标注最高点
                    if filtered_y:
                        max_idx_in_filtered = np.argmax(filtered_y)
                        original_index = valid_indices[max_idx_in_filtered]
                        max_val = filtered_y[max_idx_in_filtered]
                        max_text = sents[original_index].get('text', '').replace('\n', ' ').strip()

                        if len(max_text) > MAX_TEXT_LENGTH:
                            max_text = max_text[:MAX_TEXT_LENGTH] + "..."

                        ax.annotate(
                            f"Peak: {max_val:.2f}\n\"{max_text}\"",
                            xy=(filtered_x[max_idx_in_filtered], max_val),
                            xytext=(filtered_x[max_idx_in_filtered], max_val + 0.1),  # 微调偏移避免重叠
                            arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5),
                            fontsize=7, ha='center', color='#d62728',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
                        )

                    title_id = item.get('id', '')
                    strategy = item.get('strategy', '')
                    ax.set_title(f"{title_id}\n{strategy}", fontsize=9, fontweight='bold')

                    # === 新增修改开始 ===
                    ax.set_ylim(0, 1)  # 固定纵坐标最大值为 1
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 只显示整数作为 X 轴刻度
                    # === 新增修改结束 ===

                    ax.grid(True, linestyle='--', alpha=0.5)
                else:
                    ax.axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ 流图 PDF 已保存至: {path}")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading Data...")
    data = load_data(INPUT_FILE)
    
    # 核心处理
    df_details, df_max, plot_data = process_data(data)
    
    if df_details.empty:
        print("无有效数据（可能所有样本都只有1句话，被删除后为空）。")
        return

    # 1. 保存详细 CSV
    save_detailed_csv(df_details, OUTPUT_DIR)
    
    # 2. 绘制并保存峰值分布图
    plot_max_entropy_distribution(df_max, OUTPUT_DIR)
    
    # 3. 绘制并保存流图 PDF
    save_flow_pdf(plot_data, OUTPUT_DIR)
    
    print("\n所有任务完成！")

if __name__ == "__main__":
    main()