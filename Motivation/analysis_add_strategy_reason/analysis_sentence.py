import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
import re
import argparse

# === 配置区域 ===
DEFAULT_OUTPUT_DIR = "analysis_result_4stages" # 改为4stages以示区分
# =============

# 绘图设置
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: 文件 {file_path} 不存在")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==============================================================================
# 核心逻辑: Token 分割与断句
# ==============================================================================

def segment_metrics_by_phase(metrics, phase):
    """
    根据 XML 标签提取指定阶段，或者返回整体
    """
    # 新增: 如果是 overall，直接返回全部 metrics
    if phase == "overall":
        return metrics

    if phase == "state_analysis":
        start_tag, end_tag = "<state_analysis>", "</state_analysis>"
    elif phase == "strategy":
        start_tag, end_tag = "<strategy>", "</strategy>"
    elif phase == "response":
        start_tag, end_tag = "<response>", "</response>"
    else:
        return []

    full_text = ""
    token_indices = []
    
    for m in metrics:
        start_idx = len(full_text)
        token_str = m['token']
        full_text += token_str
        end_idx = len(full_text)
        token_indices.append((start_idx, end_idx))
    
    start_pos = -1
    match_start = re.search(re.escape(start_tag), full_text, re.IGNORECASE)
    if match_start:
        start_pos = match_start.end()
    else:
        return []

    end_pos = -1
    match_end = re.search(re.escape(end_tag), full_text[start_pos:], re.IGNORECASE)
    if match_end:
        end_pos = start_pos + match_end.start()
    else:
        end_pos = len(full_text)

    list_start_idx = -1
    list_end_idx = -1

    for i, (t_start, t_end) in enumerate(token_indices):
        if list_start_idx == -1 and t_end > start_pos:
            list_start_idx = i
        if t_start < end_pos:
            list_end_idx = i + 1
        else:
            break
    
    if list_start_idx != -1 and list_end_idx != -1 and list_start_idx < list_end_idx:
        return metrics[list_start_idx:list_end_idx]
    
    return []

def split_into_sentences(metrics, min_len=3):
    """断句算法"""
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
        elif ('.' in token or '?' in token or '!' in token or '。' in token):
            is_end = True
            
        if is_end:
            stripped_text = current_text_buffer.strip()
            # 过滤掉标签文本本身，防止把 <strategy> 当作一句话
            if stripped_text.startswith("<") and stripped_text.endswith(">"):
                 # 清空 buffer 但不添加句子
                current_tokens = []
                current_entropies = []
                current_text_buffer = ""
                continue

            if len(stripped_text) > min_len: 
                sentences.append({
                    "text": stripped_text,
                    "entropies": current_entropies,
                })
                current_tokens = []
                current_entropies = []
                current_text_buffer = ""
            elif len(stripped_text) == 0:
                current_tokens = []
                current_entropies = []
                current_text_buffer = ""

    if len(current_text_buffer.strip()) > min_len:
        # 同样检查最后的 buffer 是否是标签
        stripped_text = current_text_buffer.strip()
        if not (stripped_text.startswith("<") and stripped_text.endswith(">")):
            sentences.append({
                "text": stripped_text,
                "entropies": current_entropies,
            })
            
    return sentences

# ==============================================================================
# PART 1: 趋势分析 (Trend Analysis)
# ==============================================================================

def extract_trend_data(data):
    """提取趋势数据，包含 Overall"""
    records = []
    print("--- [Trend Analysis] Extracting Data ---")
    
    # 新增 "overall"
    phases = ["state_analysis", "strategy", "response", "overall"]
    
    for session_idx, session in enumerate(data):
        user_id = session.get('user_id', f'User_{session_idx}')
        history = session.get('detailed_history', [])
        
        manual_turn_count = 0
        
        for turn in history:
            role = turn['role']
            if role == 'Persuader':
                manual_turn_count += 1
                metrics = turn.get('metrics') 
                if not metrics: continue
                
                for phase in phases:
                    phase_metrics = segment_metrics_by_phase(metrics, phase)
                    if len(phase_metrics) < 2: continue
                    
                    min_len = 1 if phase == "strategy" else 3
                    sentences = split_into_sentences(phase_metrics, min_len=min_len)
                    
                    if not sentences: continue
                    
                    means = [np.mean(s['entropies']) for s in sentences]
                    max_mean_entropy = max(means)
                    
                    records.append({
                        "User": user_id,
                        "Turn": manual_turn_count,
                        "Phase": phase,
                        "Phase_Label": phase.replace("_", " ").title(),
                        "Max Sentence Entropy": max_mean_entropy
                    })
    
    return pd.DataFrame(records)

def plot_trend_charts(df, output_dir, remark=""):
    """绘制趋势折线图"""
    if df.empty:
        print("无有效数据可绘图 (Trend Analysis)")
        return

    phases = df['Phase_Label'].unique()
    
    # 1. 分阶段绘制
    for phase_label in phases:
        phase_data = df[df['Phase_Label'] == phase_label]
        if phase_data.empty: continue
            
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=phase_data, 
            x="Turn", 
            y="Max Sentence Entropy", 
            hue="User", 
            marker="o",
            linewidth=2.5,
            palette="tab10"
        )

        plt.title(f"Entropy Trend - {phase_label}", fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("Dialogue Turn", fontsize=12)
        plt.ylabel("Max Sentence Entropy", fontsize=12)
        plt.xticks(range(1, int(phase_data["Turn"].max()) + 1))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="User ID")
        plt.tight_layout()

        file_name = f"trend_{phase_label.replace(' ', '_').lower()}{'_'+remark if remark else ''}.png"
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ [Trend] {phase_label} 趋势图已保存")

    # 2. 综合对比图 (不包含 Overall，因为 Overall 可能会混淆视听，或者单独画)
    # 这里我们排除 Overall 来做对比图，因为 Overall 是其他三个的集合，放在一起比较意义不大
    df_compare = df[df['Phase'] != 'overall']
    if not df_compare.empty:
        plt.figure(figsize=(12, 7))
        sns.lineplot(
            data=df_compare,
            x="Turn",
            y="Max Sentence Entropy",
            hue="Phase_Label",
            style="Phase_Label",
            markers=True,
            dashes=False,
            linewidth=3,
            errorbar=None
        )
        plt.title("Average Entropy Trend Across Phases (Components Only)", fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("Dialogue Turn", fontsize=12)
        plt.ylabel("Avg Max Entropy", fontsize=12)
        plt.xticks(range(1, int(df["Turn"].max()) + 1))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title="Phase", fontsize=10)
        plt.tight_layout()
        
        file_name = f"trend_phases_comparison{'_'+remark if remark else ''}.png"
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ [Trend] 阶段对比汇总图已保存")

# ==============================================================================
# PART 2: 详细流与分布分析
# ==============================================================================

def process_detailed_data(data):
    """增加 Overall 的详细处理"""
    detailed_logs = []
    # 新增 overall 键
    max_means_data = {"state_analysis": [], "strategy": [], "response": [], "overall": []}
    plot_data_by_phase = {"state_analysis": [], "strategy": [], "response": [], "overall": []}
    
    print("--- [Detailed Analysis] Processing Data ---")
    
    phases = ["state_analysis", "strategy", "response", "overall"]

    for session_idx, session in enumerate(data):
        user_id = session.get('user_id', f'User_{session_idx}')
        history = session.get('detailed_history', [])
        
        turn_count = 0
        for turn in history:
            if turn['role'] == 'Persuader':
                turn_count += 1
                metrics = turn.get('metrics')
                if not metrics: continue
                
                strategy_name = turn.get('strategy_name', 'Unknown')
                
                for phase in phases:
                    phase_metrics = segment_metrics_by_phase(metrics, phase)
                    min_len = 1 if phase == "strategy" else 3
                    
                    if len(phase_metrics) > 0:
                        sentences = split_into_sentences(phase_metrics, min_len=min_len)
                        if not sentences: continue
                        
                        turn_id = f"{user_id}_T{turn_count}"
                        current_turn_means = []
                        sentences_info = []
                        
                        for s_idx, sent in enumerate(sentences):
                            if not sent['entropies']: continue
                            
                            mean_ent = np.mean(sent['entropies'])
                            max_ent = np.max(sent['entropies'])
                            std_ent = np.std(sent['entropies'])
                            
                            current_turn_means.append(mean_ent)
                            
                            detailed_logs.append({
                                "Sample_ID": turn_id,
                                "Round": turn_count,
                                "Phase": phase,
                                "Sentence_Index": s_idx + 1,
                                "Text": sent['text'],
                                "Mean_Entropy": mean_ent,
                                "Max_Entropy": max_ent,
                                "Std_Entropy": std_ent,
                                "Strategy_Used": strategy_name
                            })
                            
                            sentences_info.append({
                                "idx": s_idx + 1,
                                "mean": mean_ent,
                                "std": std_ent,
                                "text": sent['text']
                            })
                        
                        if current_turn_means:
                            peak = max(current_turn_means)
                            max_means_data[phase].append({
                                "Peak_Entropy": peak,
                                "Phase": phase
                            })
                        
                        plot_data_by_phase[phase].append({
                            "id": turn_id,
                            "strategy": strategy_name,
                            "sentences": sentences_info
                        })

    return (
        pd.DataFrame(detailed_logs),
        {k: pd.DataFrame(v) for k, v in max_means_data.items()},
        plot_data_by_phase
    )

def plot_distributions(df_dict, output_dir, remark=""):
    """绘制分布图，包含 Overall"""
    plt.figure(figsize=(10, 6))
    has_data = False
    # 新增 overall 颜色 (紫色)
    colors = {
        "state_analysis": "#1f77b4", 
        "strategy": "#2ca02c", 
        "response": "#d62728",
        "overall": "#9467bd" 
    }
    labels = {
        "state_analysis": "State Analysis", 
        "strategy": "Strategy", 
        "response": "Response",
        "overall": "Overall (Whole Output)"
    }
    
    # 调整绘制顺序，让 Overall 在最底层或者最上层，看需求。
    # 这里建议单独画一张 Overall，或者放在一起比较（如果 Overall 覆盖了其他分布，可能会乱）
    # 策略：先画 Overall 的填充较淡，再画各个组件
    
    # 1. 先画 Overall
    if "overall" in df_dict and not df_dict["overall"].empty:
        sns.kdeplot(data=df_dict["overall"], x="Peak_Entropy", fill=True, 
                    color=colors["overall"], label=labels["overall"], alpha=0.1, linewidth=2)
        has_data = True

    # 2. 再画各个组件
    for phase in ["state_analysis", "strategy", "response"]:
        if phase in df_dict and not df_dict[phase].empty:
            sns.kdeplot(data=df_dict[phase], x="Peak_Entropy", fill=True, 
                        color=colors[phase], label=labels[phase], alpha=0.3)
            has_data = True
            
    if has_data:
        plt.title("Distribution of Peak Entropy (Including Overall)", fontsize=16)
        plt.xlabel("Max Mean Entropy (Sentence Level)", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()
        plt.tight_layout()
        
        file_name = f"distribution_with_overall{'_'+remark if remark else ''}.png"
        save_path = os.path.join(output_dir, file_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ [Dist] 包含 Overall 的分布图已保存")

def save_flow_pdfs(plot_data_dict, output_dir, remark=""):
    """生成流图 PDF"""
    MAX_TEXT_LEN = 20
    
    for phase, plot_list in plot_data_dict.items():
        if not plot_list: continue
        
        file_name = f"flow_{phase}{'_'+remark if remark else ''}.pdf"
        save_path = os.path.join(output_dir, file_name)
        print(f"正在生成 PDF: {file_name} ...")
        
        with PdfPages(save_path) as pdf:
            plots_per_page = 9
            for start_idx in range(0, len(plot_list), plots_per_page):
                batch = plot_list[start_idx : start_idx + plots_per_page]
                
                fig, axes = plt.subplots(3, 3, figsize=(20, 12))
                axes = axes.flatten()
                
                phase_title = phase.replace('_', ' ').title()
                fig.suptitle(f"Entropy Flow - {phase_title}", fontsize=16)
                
                for i in range(9):
                    ax = axes[i]
                    if i < len(batch):
                        item = batch[i]
                        sents = item['sentences']
                        
                        if not sents:
                            ax.axis('off'); continue
                            
                        x = [s['idx'] for s in sents]
                        y = [s['mean'] for s in sents]
                        yerr = [s['std'] for s in sents]
                        
                        ax.errorbar(x, y, yerr=yerr, fmt='-o', capsize=3, 
                                    color='#1f77b4', ecolor='lightgray', markersize=4)
                        
                        if y:
                            max_idx = np.argmax(y)
                            max_val = y[max_idx]
                            max_text = sents[max_idx]['text']
                            if len(max_text) > MAX_TEXT_LEN:
                                max_text = max_text[:MAX_TEXT_LEN] + "..."
                                
                            ax.annotate(
                                f"{max_val:.2f}\n\"{max_text}\"",
                                xy=(x[max_idx], max_val),
                                xytext=(x[max_idx], max_val + 0.2),
                                arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5),
                                fontsize=8, ha='center', color='darkred',
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
                            )
                        
                        title_str = f"{item['id']}"
                        if phase != "strategy":
                            title_str += f"\nStrat: {item['strategy']}"
                            
                        ax.set_title(title_str, fontsize=10, fontweight='bold')
                        ax.set_ylim(0, max(y + [1.0]) * 1.3)
                        ax.grid(True, linestyle=':', alpha=0.6)
                        ax.set_xlabel("Step")
                    else:
                        ax.axis('off')
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)
        
        print(f"✅ [Flow] {phase} 流图 PDF 已保存")

# ==============================================================================
# 主函数
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Stage Dialogue Entropy Analysis")
    parser.add_argument("--input", required=True, help="Input JSON result file")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--remark", default="", help="Suffix for output filenames")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"=== Loading Data: {args.input} ===")
    data = load_data(args.input)
    if not data: return

    print("\n>>> Phase 1: Trend Analysis (Including Overall)")
    df_trend = extract_trend_data(data)
    plot_trend_charts(df_trend, args.output_dir, args.remark)
    if not df_trend.empty:
        df_trend.to_csv(os.path.join(args.output_dir, f"trend_data_all{'_'+args.remark if args.remark else ''}.csv"), index=False)

    print("\n>>> Phase 2: Detailed Flow & Distribution (Including Overall)")
    df_details, df_max_means_dict, plot_data_dict = process_detailed_data(data)
    
    if not df_details.empty:
        csv_path = os.path.join(args.output_dir, f"detailed_logs_all{'_'+args.remark if args.remark else ''}.csv")
        df_details.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 详细日志已保存: {csv_path}")
        
        plot_distributions(df_max_means_dict, args.output_dir, args.remark)
        save_flow_pdfs(plot_data_dict, args.output_dir, args.remark)
    else:
        print("未提取到详细数据，跳过 Phase 2 绘图。")

    print(f"\n=== All Tasks Completed. Results in: {args.output_dir} ===")

if __name__ == "__main__":
    main()