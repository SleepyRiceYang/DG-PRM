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
OUTPUT_DIR = "analysis_all_samples_vis_v2"  # 结果输出目录
# =============


# 绘图设置
sns.set_theme(style="whitegrid")
# 字体设置，优先尝试支持中文的字体
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_high_entropy_report(samples, output_dir):
    """
    保存高熵 Token 的详细统计报告。
    新增功能：计算高熵 Token 占比 (High Entropy Ratio)。
    """
    if not samples: return

    file_path = os.path.join(output_dir, "high_entropy_tokens_analysis_2std_0.8.txt")
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("HIGH ENTROPY TOKEN ANALYSIS REPORT (Threshold: Mean + 1*Std)\n")
        f.write("此报告列出了在可视化图中被标注的所有高熵 Token 及其上下文信息。\n")
        f.write("="*80 + "\n\n")

        count_samples = 0
        total_high_tokens = 0

        for sample in samples:
            metrics = sample['metrics']
            entropies = [m['entropy'] for m in metrics]
            
            if not entropies: continue

            # 1. 计算阈值
            mean_val = np.mean(entropies)
            std_val = np.std(entropies)
            threshold = max(mean_val + std_val*2, 0.8)

            # 2. 筛选高熵 Token
            high_ent_tokens = []
            for idx, (m, val) in enumerate(zip(metrics, entropies)):
                if val > threshold:
                    high_ent_tokens.append({
                        'index': idx,
                        'token': m['token'],
                        'entropy': val,
                        'prob': m.get('top_5', [{}])[0].get('prob', 0)
                    })
            
            # 如果该样本有高熵 Token，则写入报告
            if high_ent_tokens:
                count_samples += 1
                total_high_tokens += len(high_ent_tokens)

                # ================= [新增] 计算比例逻辑 =================
                total_token_count = len(metrics)            # 总 Token 数
                high_token_count = len(high_ent_tokens)     # 高熵 Token 数
                ratio = high_token_count / total_token_count # 比例
                # ======================================================

                f.write(f"[Sample ID]: {sample['id']}\n")
                f.write(f"Strategy   : {sample['strategy']}\n")
                
                # 写入基础统计
                f.write(f"Statistics : Mean={mean_val:.4f} | Std={std_val:.4f} | Threshold={threshold:.4f}\n")
                
                # ================= [新增] 写入比例信息 =================
                # 格式示例: Proportion : 5/120 tokens (4.17%)
                f.write(f"Proportion : {high_token_count}/{total_token_count} tokens are high entropy ({ratio:.2%})\n")
                # ======================================================
                
                f.write("-" * 20 + " Full Thinking Process " + "-" * 20 + "\n")
                f.write(f"{sample['full_text']}\n")
                
                f.write("-" * 20 + " High Entropy Tokens " + "-" * 20 + "\n")
                f.write(f"{'Idx':<5} | {'Entropy':<8} | {'Prob':<6} | {'Token'}\n")
                f.write("-" * 60 + "\n")
                
                for item in high_ent_tokens:
                    token_clean = item['token'].replace('\n', '\\n').replace('\r', '').strip()
                    f.write(f"{item['index']:<5} | {item['entropy']:.4f}   | {item['prob']:.4f} | {token_clean}\n")
                
                f.write("\n" + "="*80 + "\n\n")

    print(f"✅ 高熵 Token 详细报告已保存至: {file_path}")
    print(f"   (共记录 {count_samples} 个样本，涉及 {total_high_tokens} 个高熵 Token)")
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
    绘制 PDF：
    1. 灰色连线 + 彩色散点
    2. 橙色虚线划分高熵
    3. 文字紧贴点上方，45度倾斜，灰蓝色加粗
    4. [新增] 左上角显示高熵 Token 占比 (Ratio)
    """
    if not samples:
        print("无样本可绘制")
        return

    pdf_path = os.path.join(output_dir, "all_think_entropy_analysis_2std_0.8.pdf")
    print(f"正在生成可视化 PDF (共 {len(samples)} 个样本)...")
    
    plots_per_page = 9
    rows, cols = 3, 3
    
    with PdfPages(pdf_path) as pdf:
        for page_start in range(0, len(samples), plots_per_page):
            fig, axes = plt.subplots(rows, cols, figsize=(24, 15))
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
                    mean_val = np.mean(entropies)
                    std_val = np.std(entropies)
                    threshold = max(mean_val + std_val*2, 0.8) 

                    # === 2. 绘制基础图形 ===
                    ax.plot(x, entropies, color='gray', alpha=0.4, linewidth=1, zorder=1)
                    sc = ax.scatter(x, entropies, c=entropies, cmap='viridis', s=20, zorder=2)
                    ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=1, alpha=0.8)

                    # === 3. 标注阈值之上的 Token 并统计数量 ===
                    high_ent_count = 0 # [新增] 计数器
                    
                    for j, val in enumerate(entropies):
                        if val > threshold:
                            high_ent_count += 1 # [新增] 计数
                            
                            token_text = metrics[j]['token'].strip().replace('\n', '\\n')
                            if len(token_text) > 10: token_text = token_text[:9] + "."
                            
                            # 标注 Token (灰蓝色、加粗)
                            ax.text(j, val + 0.05, 
                                    token_text,
                                    fontsize=6, 
                                    color='#4C6E81', 
                                    fontweight='bold',
                                    rotation=45,
                                    ha='left', 
                                    va='bottom')

                    # === 4. [新增] 在左上角显示占比信息 ===
                    total_tokens = len(entropies)
                    ratio = high_ent_count / total_tokens if total_tokens > 0 else 0
                    
                    # 准备显示的文本
                    ratio_text = f"High Ent Ratio: {ratio:.1%}\n({high_ent_count}/{total_tokens} tokens)"
                    
                    # 绘制文本框 (使用相对坐标 transAxes，0,1 表示左上角)
                    ax.text(0.02, 0.95, ratio_text, 
                            transform=ax.transAxes, # 使用相对坐标系
                            fontsize=9, 
                            fontweight='bold',
                            color='#333333',
                            verticalalignment='top', 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', alpha=0.8, edgecolor='none'))

                    # === 5. 设置标签与标题 ===
                    title_str = f"ID: {sample['id']} | Strat: {sample['strategy']}"
                    ax.set_title(title_str, fontsize=10, fontweight='bold', pad=10)
                    
                    y_max = max(max(entropies), threshold)
                    ax.set_ylim(0, y_max * 1.5) 
                    
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
   

    print("Generating High Entropy Text Report...")
    save_high_entropy_report(samples, OUTPUT_DIR)
    
    print("\nDone!")

if __name__ == "__main__":
    main()