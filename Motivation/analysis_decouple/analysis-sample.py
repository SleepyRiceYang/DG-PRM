import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
import os
import re
import numpy as np

# === 配置区域 ===
INPUT_FILE = "results_decoupled_2025-11-26T18-46-54.json"  # 替换为你的输入文件
OUTPUT_DIR = "analysis_all_samples_vis"     # 结果输出目录
# =============

# 绘图设置
sns.set_theme(style="whitegrid")
# 字体设置，优先尝试支持中文的字体
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

def process_all_samples(data):
    """提取所有样本的思考过程"""
    samples = []
    for session_idx, session in enumerate(data):
        user_id = session.get('user_id', f'User_{session_idx}')
        history = session.get('detailed_history', [])
        
        for turn_idx, turn in enumerate(history):
            if turn['role'] == 'Persuader':
                metrics = turn.get('strategy_entropy_metrics')
                if not metrics: continue
                
                # 只保留思考部分
                reasoning_m, _, _ = segment_metrics(metrics)
                
                # 过滤过短的序列
                if len(reasoning_m) < 5:
                    continue

                final_strategy = turn.get('strategy', 'Unknown Strategy')

                samples.append({
                    "id": f"{user_id}_Turn{turn.get('round', turn_idx+1)}",
                    "strategy": final_strategy,
                    "metrics": reasoning_m,
                    "full_text": "".join([m['token'] for m in reasoning_m])
                })
    return samples

def save_all_plots_to_pdf(samples, output_dir):
    """
    将所有样本绘制到 PDF，采用优化的视觉风格：
    1. 灰色连线 + 彩色散点
    2. 增加高熵阈值横线
    3. 标注所有高于横线的 Token
    """
    if not samples:
        print("无样本可绘制")
        return

    pdf_path = os.path.join(output_dir, "all_think_entropy_analysis_v2.pdf")
    print(f"正在生成可视化 PDF (共 {len(samples)} 个样本)...")
    
    plots_per_page = 9
    rows, cols = 3, 3
    
    with PdfPages(pdf_path) as pdf:
        # 分批处理，每页9张图
        for page_start in range(0, len(samples), plots_per_page):
            fig, axes = plt.subplots(rows, cols, figsize=(24, 15)) # 画布调大一点以容纳标注
            axes = axes.flatten()
            
            batch_samples = samples[page_start : page_start + plots_per_page]
            
            for i in range(rows * cols):
                ax = axes[i]
                
                if i < len(batch_samples):
                    sample = batch_samples[i]
                    metrics = sample['metrics']
                    entropies = [m['entropy'] for m in metrics]
                    x = range(len(entropies))
                    
                    if not entropies:
                        ax.axis('off'); continue

                    # === 1. 动态计算高熵阈值 ===
                    # 规则：平均值 + 1倍标准差。为了避免整体都很低时标出一堆废话，设置底限 0.8
                    mean_val = np.mean(entropies)
                    std_val = np.std(entropies)
                    threshold = max(mean_val + std_val, 0.8) 

                    # === 2. 绘制基础图形 ===
                    # 灰色折线
                    ax.plot(x, entropies, color='gray', alpha=0.4, linewidth=1, zorder=1)
                    
                    # 彩色散点 (Viridis 色阶)
                    sc = ax.scatter(x, entropies, c=entropies, cmap='viridis', s=15, zorder=2)
                    
                    # 绘制阈值横线
                    ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=1, alpha=0.6, label='High Entropy Threshold')

                    # === 3. 标注阈值之上的所有 Token ===
                    # 为了防止文字重叠，我们让高度上下错开
                    annotated_count = 0
                    for j, val in enumerate(entropies):
                        if val > threshold:
                            token_text = metrics[j]['token'].strip().replace('\n', '\\n')
                            
                            # 如果 token 太长，截断一下
                            if len(token_text) > 8: token_text = token_text[:7] + "."
                            
                            # 错位高度逻辑：偶数索引高一点，奇数索引低一点
                            offset = 0.2 if (annotated_count % 2 == 0) else 0.5
                            annotated_count += 1
                            
                            ax.text(j, val + offset, token_text,
                                    fontsize=6, color='#333333', rotation=90,
                                    ha='center', va='bottom',
                                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
                            
                            # 画一条细线连到点上
                            ax.plot([j, j], [val, val + offset], color='red', linewidth=0.5, alpha=0.3)

                    # === 4. 设置标签与标题 ===
                    title_str = f"ID: {sample['id']} | Strat: {sample['strategy']}"
                    ax.set_title(title_str, fontsize=10, fontweight='bold', pad=15)
                    
                    # 动态调整 Y 轴范围，留出上方空间给标注
                    y_max = max(max(entropies), threshold)
                    ax.set_ylim(0, y_max * 1.4) 
                    
                    # 只在边缘子图显示坐标轴标签，保持整洁
                    if i >= 6: 
                        ax.set_xlabel("Token Index", fontsize=9)
                    if i % 3 == 0: 
                        ax.set_ylabel("Entropy", fontsize=9)
                        
                else:
                    ax.axis('off')
            
            plt.tight_layout()
            pdf.savefig(fig) 
            plt.close(fig)
            
    print(f"✅ PDF 可视化报告已保存至: {pdf_path}")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading Data...")
    data = load_data(INPUT_FILE)
    
    print("Processing samples...")
    samples = process_all_samples(data)
    
    if not samples:
        print("No valid thinking samples found.")
        return

    print("Generating Enhanced Visualization...")
    save_all_plots_to_pdf(samples, OUTPUT_DIR)
    
    print("\nDone!")

if __name__ == "__main__":
    main()