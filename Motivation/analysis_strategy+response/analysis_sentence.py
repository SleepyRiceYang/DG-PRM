import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
import re
import argparse # <--- 新增导入
# === 配置区域 ===

OUTPUT_DIR = "analysis_sentence"     # 结果输出目录
# =============

# 绘图设置
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: 文件 {file_path} 不存在")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ==============================================================================
# PART 1: 趋势分析逻辑 (Trend Analysis Logic - 对应第一个文件)
# ==============================================================================

def segment_metrics_simple(metrics):
    """
    [PART 1] 提取推理过程 (Reasoning Phase) - 简单模式
    在新格式中，推理位于 <reasoning> 和 </reasoning> 标签之间。
    """
    split_index = -1
    for i, item in enumerate(metrics):
        token = item['token']
        if "<strategy>" in token or "</reasoning>" in token:
            split_index = i
            break
    
    if split_index != -1:
        return metrics[:split_index]
    else:
        return metrics

def split_into_sentences_simple(metrics):
    """
    [PART 1] 基于 Token 断句 - 简单模式 (阈值 > 1)
    """
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
    [PART 1] 提取趋势绘图所需数据
    """
    records = []
    print("--- [Trend Analysis] Extracting Data ---")
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
                
                # 使用 PART 1 的简单分割逻辑
                reasoning_metrics = segment_metrics_simple(metrics)
                if len(reasoning_metrics) < 5: continue
                
                # 使用 PART 1 的简单断句逻辑
                sentences = split_into_sentences_simple(reasoning_metrics)
                
                if len(sentences) > 0:
                    sentences = sentences[:-1]
                
                if not sentences: continue
                
                means = [np.mean(s['entropies']) for s in sentences]
                if not means: continue
                
                max_mean_entropy = max(means)
                
                records.append({
                    "User": user_id,
                    "Turn": manual_turn_count,
                    "Max Sentence Entropy": max_mean_entropy
                })
    
    return pd.DataFrame(records)

def plot_trend_chart(df, output_dir,remark=""):
    """
    [PART 1] 绘制趋势折线图
    """
    if df.empty:
        print("无有效数据可绘图 (Trend Analysis)")
        return

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df, 
        x="Turn", 
        y="Max Sentence Entropy", 
        hue="User", 
        style="User",
        markers=True, 
        dashes=False,
        linewidth=2.0,
        palette="tab10",
        markersize=8
    )

    plt.title("Trend of Peak Thinking Difficulty per Turn", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Dialogue Turn", fontsize=12)
    plt.ylabel("Max Sentence Entropy", fontsize=12)
    
    if not df.empty:
        max_turn = df["Turn"].max()
        if pd.notna(max_turn) and max_turn > 0:
            plt.xticks(range(1, int(max_turn) + 1))
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="User ID")
    plt.tight_layout()

    suffix = f"_{remark}" if remark else ""
    save_path = os.path.join(output_dir, f"think_entropy_trend{suffix}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ [Trend] 趋势对比图已保存至: {save_path}")

# ==============================================================================
# PART 2: 详细分析逻辑 (Detailed Flow & Distribution Analysis - 对应第二个文件)
# ==============================================================================

def segment_metrics_reasoning_robust(metrics):
    """
    [PART 2] 提取推理过程 - 健壮模式
    查找 <reasoning> ... </reasoning>
    """
    tokens_text = "".join([m['token'] for m in metrics])
    
    start_pos = tokens_text.find("<reasoning>")
    if start_pos == -1:
        start_pos = tokens_text.find("<reason")
        if start_pos != -1:
            close_bracket_pos = tokens_text.find(">", start_pos)
            if close_bracket_pos != -1:
                start_pos = close_bracket_pos + 1
            else:
                start_pos = -1
    
    end_pos = tokens_text.find("</reasoning>", start_pos)
    
    if start_pos != -1 and end_pos != -1:
        start_index = 0
        current_pos = 0
        for i, item in enumerate(metrics):
            if current_pos <= start_pos < current_pos + len(item['token']):
                start_index = i
                break
            current_pos += len(item['token'])
        
        end_index = 0
        current_pos = 0
        for i, item in enumerate(metrics):
            if current_pos <= end_pos < current_pos + len(item['token']):
                end_index = i
                break
            current_pos += len(item['token'])
        
        if start_index < end_index:
            return metrics[start_index:end_index]
    
    if start_pos != -1:
        start_index = 0
        current_pos = 0
        for i, item in enumerate(metrics):
            if current_pos <= start_pos < current_pos + len(item['token']):
                start_index = i
                break
            current_pos += len(item['token'])
        return metrics[start_index:]
    
    return []

def segment_metrics_response_robust(metrics):
    """
    [PART 2] 提取响应过程 - 健壮模式
    查找 <response> ... </response>
    """
    tokens_text = "".join([m['token'] for m in metrics])
    
    start_pos = tokens_text.find("<response>")
    if start_pos == -1:
        start_pos = tokens_text.find("<response")
        if start_pos != -1:
            close_bracket_pos = tokens_text.find(">", start_pos)
            if close_bracket_pos != -1:
                start_pos = close_bracket_pos + 1
            else:
                start_pos = -1
    
    end_pos = tokens_text.find("</response>", start_pos)
    
    if start_pos != -1 and end_pos != -1:
        start_index = 0
        current_pos = 0
        for i, item in enumerate(metrics):
            if current_pos <= start_pos < current_pos + len(item['token']):
                start_index = i
                break
            current_pos += len(item['token'])
        
        end_index = 0
        current_pos = 0
        for i, item in enumerate(metrics):
            if current_pos <= end_pos < current_pos + len(item['token']):
                end_index = i
                break
            current_pos += len(item['token'])
        
        if start_index < end_index:
            return metrics[start_index:end_index]
    
    if start_pos != -1:
        start_index = 0
        current_pos = 0
        for i, item in enumerate(metrics):
            if current_pos <= start_pos < current_pos + len(item['token']):
                start_index = i
                break
            current_pos += len(item['token'])
        return metrics[start_index:]
    
    return []

def split_into_sentences_robust(metrics):
    """
    [PART 2] 基于 Token 流的断句算法 - 健壮模式 (阈值 > 3)
    注意：此处的逻辑与 PART 1 略有不同，增加了 buffer 长度判定
    """
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
        if '\n' in token and len(current_text_buffer.strip()) > 3:
            is_end = True
        elif ('.' in token or '?' in token or '!' in token) and len(current_text_buffer.strip()) > 3:
            is_end = True
            
        if is_end:
            if len(current_text_buffer.strip()) > 3: 
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

    if len(current_text_buffer.strip()) > 3:
        sentences.append({
            "text": current_text_buffer.strip(),
            "entropies": current_entropies,
        })
    return sentences

def process_data_detailed(data):
    """
    [PART 2] 处理数据：提取 Persuader 的推理和响应过程 -> 断句 -> 统计
    """
    detailed_logs_reasoning = []
    detailed_logs_response = []
    turn_max_means_reasoning = []
    turn_max_means_response = []
    plot_data_reasoning = []
    plot_data_response = []
    
    print("--- [Detailed Analysis] Processing Data ---")
    print(f"总共会话数: {len(data)}")
    
    for session_idx, session in enumerate(data):
        user_id = session.get('user_id', f'User_{session_idx}')
        history = session.get('detailed_history', [])
        
        print(f"处理会话 {session_idx+1}/{len(data)}, 用户ID: {user_id}")
        
        manual_turn_counter = 0
        
        for turn_idx, turn in enumerate(history):
            if turn['role'] == 'Persuader':
                manual_turn_counter += 1
                metrics = turn.get('metrics')
                if not metrics: continue
                
                # --- 提取推理部分 (Reasoning Phase) ---
                reasoning_m = segment_metrics_reasoning_robust(metrics)
                
                if len(reasoning_m) >= 5:
                    sentences = split_into_sentences_robust(reasoning_m)
                    
                    if sentences:
                        sentences = sentences[:-1] if len(sentences) > 1 else sentences
                        
                        if sentences:
                            round_num = turn.get('round', manual_turn_counter)
                            turn_id = f"{user_id}_Turn{round_num}"
                            strategy = turn.get('strategy', 'Unknown')
                            
                            current_turn_means = []
                            sentences_info_for_plot = []
                            
                            for s_idx, sent in enumerate(sentences):
                                if not sent['entropies'] or len(sent['entropies']) == 0: continue
                                    
                                mean_ent = np.mean(sent['entropies'])
                                max_ent = np.max(sent['entropies'])
                                std_ent = np.std(sent['entropies'])
                                
                                current_turn_means.append(mean_ent)
                                
                                detailed_logs_reasoning.append({
                                    "Sample_ID": turn_id,
                                    "Round": round_num,
                                    "Role": "Persuader",
                                    "Strategy": strategy,
                                    "Sentence_Index": s_idx + 1,
                                    "Mean_Entropy": mean_ent,
                                    "Max_Entropy": max_ent,
                                    "Std_Entropy": std_ent,
                                    "Text_Content": sent['text'].replace('\n', ' ').strip(),
                                    "Phase": "Reasoning"
                                })
                                
                                sentences_info_for_plot.append({
                                    "idx": s_idx + 1,
                                    "mean": mean_ent,
                                    "std": std_ent,
                                    "text": sent['text']
                                })
                            
                            if current_turn_means:
                                peak_difficulty = max(current_turn_means)
                                turn_max_means_reasoning.append({
                                    "Sample_ID": turn_id,
                                    "Strategy": strategy,
                                    "Peak_Mean_Entropy": peak_difficulty
                                })
                            
                            plot_data_reasoning.append({
                                "id": turn_id,
                                "strategy": strategy,
                                "sentences": sentences_info_for_plot
                            })

                # --- 提取响应部分 (Response Phase) ---
                response_m = segment_metrics_response_robust(metrics)
                
                if len(response_m) >= 5:
                    sentences = split_into_sentences_robust(response_m)
                    
                    if sentences:
                        round_num = turn.get('round', manual_turn_counter)
                        turn_id = f"{user_id}_Turn{round_num}"
                        strategy = turn.get('strategy', 'Unknown')
                        
                        current_turn_means = []
                        sentences_info_for_plot = []
                        
                        for s_idx, sent in enumerate(sentences):
                            if not sent['entropies'] or len(sent['entropies']) == 0: continue
                                
                            mean_ent = np.mean(sent['entropies'])
                            max_ent = np.max(sent['entropies'])
                            std_ent = np.std(sent['entropies'])
                            
                            current_turn_means.append(mean_ent)
                            
                            detailed_logs_response.append({
                                "Sample_ID": turn_id,
                                "Round": round_num,
                                "Role": "Persuader",
                                "Strategy": strategy,
                                "Sentence_Index": s_idx + 1,
                                "Mean_Entropy": mean_ent,
                                "Max_Entropy": max_ent,
                                "Std_Entropy": std_ent,
                                "Text_Content": sent['text'].replace('\n', ' ').strip(),
                                "Phase": "Response"
                            })
                            
                            sentences_info_for_plot.append({
                                "idx": s_idx + 1,
                                "mean": mean_ent,
                                "std": std_ent,
                                "text": sent['text']
                            })
                        
                        if current_turn_means:
                            peak_difficulty = max(current_turn_means)
                            turn_max_means_response.append({
                                "Sample_ID": turn_id,
                                "Strategy": strategy,
                                "Peak_Mean_Entropy": peak_difficulty
                            })
                        
                        plot_data_response.append({
                            "id": turn_id,
                            "strategy": strategy,
                            "sentences": sentences_info_for_plot
                        })

    detailed_logs = detailed_logs_reasoning + detailed_logs_response
    
    return (
        pd.DataFrame(detailed_logs),
        pd.DataFrame(turn_max_means_reasoning),
        pd.DataFrame(turn_max_means_response),
        plot_data_reasoning,
        plot_data_response
    )

def save_detailed_csv(df, output_dir,remark=""):
    """[PART 2] 保存详细 CSV"""
    suffix = f"_{remark}" if remark else ""
    path = os.path.join(output_dir, f"detailed_sentence_log{suffix}.csv")
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"✅ [Detail] 详细句子日志已保存至: {path}")

def plot_max_entropy_distribution(df_max_reasoning, df_max_response, output_dir,remark=""):
    """[PART 2] 绘制分布图"""
    if df_max_reasoning.empty and df_max_response.empty:
        print("警告：没有数据可用于绘制峰值分布图")
        return
    
    plt.figure(figsize=(12, 6))
    has_data = False
    
    if not df_max_reasoning.empty:
        sns.histplot(data=df_max_reasoning, x="Peak_Mean_Entropy", kde=True, 
                     color="#d62728", bins=15, alpha=0.7, label="Reasoning")
        has_data = True
    
    if not df_max_response.empty:
        sns.histplot(data=df_max_response, x="Peak_Mean_Entropy", kde=True, 
                     color="#2ca02c", bins=15, alpha=0.7, label="Response")
        has_data = True
    
    if not has_data: return
        
    plt.title("Distribution of Peak Entropy per Turn\n(Reasoning and Response Phases)", fontsize=14)
    plt.xlabel("Max Mean Entropy of a Sentence", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.tight_layout()

    suffix = f"_{remark}" if remark else ""
    path = os.path.join(output_dir, f"peak_entropy_distribution{suffix}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ [Detail] 峰值分布图已保存至: {path}")

def save_flow_pdf(plot_data, output_dir, filename_suffix,remark=""):
    """[PART 2] 绘制流图 PDF"""
    if not plot_data:
        print(f"警告：没有{filename_suffix}数据可用于生成PDF")
        return
    
    MAX_TEXT_LENGTH = 15

    suffix = f"_{remark}" if remark else ""
    path = os.path.join(output_dir, f"flow_analysis_{filename_suffix}_{suffix}.pdf")
    print(f"正在生成{filename_suffix}流图 PDF...")

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
                    sents = item['sentences']
                    if not sents:
                        ax.axis('off')
                        continue

                    x = [s.get('idx') for s in sents]
                    y = [s.get('mean') for s in sents]
                    yerr = [s.get('std') for s in sents]

                    valid_indices = [j for j in range(len(y)) if y[j] is not None and not np.isnan(y[j])]
                    if not valid_indices:
                        ax.axis('off')
                        continue

                    filtered_x = [x[j] for j in valid_indices]
                    filtered_y = [y[j] for j in valid_indices]
                    filtered_yerr = [yerr[j] for j in valid_indices]

                    ax.errorbar(filtered_x, filtered_y, yerr=filtered_yerr, fmt='-o', capsize=3,
                                color='#1f77b4', ecolor='lightgray', markersize=4)

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
                            xytext=(filtered_x[max_idx_in_filtered], max_val + 0.15),
                            arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=5),
                            fontsize=7, ha='center', color='#d62728',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none")
                        )

                    title_id = item.get('id', '')
                    strategy = item.get('strategy', '')
                    ax.set_title(f"{title_id}\nStrategy: {strategy}", fontsize=9, fontweight='bold')

                    ax.set_ylim(0, 1.2)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.set_xlabel("Sentence Index (Steps)")
                    ax.set_ylabel("Mean Entropy")
                    ax.grid(True, linestyle='--', alpha=0.5)
                else:
                    ax.axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"✅ [Detail] {filename_suffix}流图 PDF 已保存至: {path}")

# ==============================================================================
# MAIN Execution
# ==============================================================================

def main():
    # 1. 设置参数解析
    parser = argparse.ArgumentParser(description="Sentence Entropy Analysis")
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--remark", default="", help="Remark suffix for filenames")
    args = parser.parse_args()

    # 2. 检查输入路径 (使用 args.input)
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("=== Loading Data ===")
    # 3. 加载数据 (使用 args.input)
    data = load_data(args.input)
    if not data:
        print("Error: 数据加载失败")
        return

    # --- 执行 PART 1: 趋势分析 ---
    print("\n>>> Starting Part 1: Trend Analysis (Think Entropy)")
    df_trend = extract_trend_data(data)
    print(f"Extracted {len(df_trend)} trend data points.")
    
    # 调用函数时传入 args.output_dir 和 args.remark
    plot_trend_chart(df_trend, args.output_dir, args.remark)
    
    if not df_trend.empty:
        suffix = f"_{args.remark}" if args.remark else ""
        csv_path = os.path.join(args.output_dir, f"think_entropy_trend_data{suffix}.csv")
        df_trend.to_csv(csv_path, index=False)
        print(f"✅ [Trend] 数据源 CSV 已保存至: {csv_path}")

    # --- 执行 PART 2: 详细流与分布分析 ---
    print("\n>>> Starting Part 2: Detailed Flow & Distribution Analysis")
    df_details, df_max_reasoning, df_max_response, plot_data_reasoning, plot_data_response = process_data_detailed(data)
    
    print(f"处理概况:")
    print(f"- Reasoning 样本数: {len(df_max_reasoning)}")
    print(f"- Response 样本数: {len(df_max_response)}")

    if not df_details.empty:
        # 调用函数时传入 args.output_dir 和 args.remark
        save_detailed_csv(df_details, args.output_dir, args.remark)
        plot_max_entropy_distribution(df_max_reasoning, df_max_response, args.output_dir, args.remark)
        save_flow_pdf(plot_data_reasoning, args.output_dir, "reasoning", args.remark)
        save_flow_pdf(plot_data_response, args.output_dir, "response", args.remark)
    else:
        print("无详细数据可供绘图。")
    
    print("\n=== All Tasks Completed ===")

if __name__ == "__main__":
    main()