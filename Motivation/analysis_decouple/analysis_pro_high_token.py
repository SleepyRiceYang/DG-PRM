import json
import numpy as np
import os
import re

# === 配置区域 ===
INPUT_FILE = "results_decoupled_2025-11-26T18-46-54.json"  # 替换为你的输入文件
OUTPUT_DIR = "analysis_all_samples_vis"     # 结果输出目录
# =============

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
        return metrics # 没找到分割点，默认全部分析

def save_deep_token_report(data, output_dir):
    """
    生成深度 Token 分析报告，包含 Top-K 候选词对比
    """
    file_path = os.path.join(output_dir, "deep_token_entropy_analysis_2sft_0.8.txt")
    
    with open(file_path, "w", encoding="utf-8") as f:
        # === 1. 写入报告头 ===
        f.write("="*80 + "\n")
        f.write("DEEP TOKEN ENTROPY & CANDIDATE ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write("此报告深入分析了思考过程中所有 '高熵 Token' (High Entropy Tokens)。\n")
        f.write("定义标准: Entropy > max(Mean + 1*Std, 0.8)\n")
        f.write("字段说明:\n")
        f.write("  - Chosen Token : 模型最终生成的词\n")
        f.write("  - Rank         : 该词在概率预测表中的排名 (1代表它是概率最高的)\n")
        f.write("  - Gap (Top1-2) : 第一名和第二名概率的差值 (值越小，代表竞争越激烈)\n")
        f.write("="*80 + "\n\n")

        for session_idx, session in enumerate(data):
            user_id = session.get('user_id', f'User_{session_idx}')
            history = session.get('detailed_history', [])
            
            for turn_idx, turn in enumerate(history):
                if turn['role'] == 'Persuader':
                    metrics = turn.get('strategy_entropy_metrics')
                    if not metrics: continue
                    
                    # 只分析思考阶段
                    reasoning_m = segment_metrics(metrics)
                    if len(reasoning_m) < 5: continue
                    
                    entropies = [m['entropy'] for m in reasoning_m]
                    
                    # 计算阈值
                    mean_val = np.mean(entropies)
                    std_val = np.std(entropies)
                    threshold = max(mean_val + std_val*2, 0.8)
                    
                    # 筛选高熵 Token
                    high_ent_indices = [i for i, e in enumerate(entropies) if e > threshold]
                    
                    if not high_ent_indices:
                        continue
                        
                    # 写入样本头信息
                    strategy_name = turn.get('strategy', 'Unknown')
                    turn_id = f"{user_id}_Turn{turn.get('round', turn_idx+1)}"
                    
                    f.write(f"\n{'='*30} [Sample: {turn_id}] {'='*30}\n")
                    f.write(f"Strategy: {strategy_name}\n")
                    f.write(f"Stats   : Mean={mean_val:.3f}, Threshold={threshold:.3f}\n")
                    f.write(f"Context : \"{''.join([m['token'] for m in reasoning_m[:50]])}...\"\n")
                    f.write("-" * 80 + "\n")
                    
                    # === 2. 逐个分析高熵 Token ===
                    for idx in high_ent_indices:
                        token_data = reasoning_m[idx]
                        token_str = token_data['token'].replace('\n', '\\n').strip()
                        entropy = token_data['entropy']
                        
                        # 获取 Top-K 列表
                        top_candidates = token_data.get('top_5', [])
                        if not top_candidates:
                            f.write(f"[Idx {idx}] '{token_str}' (Ent: {entropy:.3f}) - No Top-K data available.\n")
                            continue

                        # 分析：模型生成的词排第几？
                        chosen_rank = -1
                        chosen_prob = 0
                        
                        # 在 Top-5 里找生成的词
                        # 注意：需要去除空白进行比较，防止 ' however' 和 'however' 不匹配
                        for r, cand in enumerate(top_candidates):
                            # 简单的去空白比较
                            if cand['token'].strip() == token_data['token'].strip():
                                chosen_rank = r + 1
                                chosen_prob = cand['prob']
                                break
                        
                        # 如果没找到，说明排名在 Top 5 之外
                        rank_str = str(chosen_rank) if chosen_rank != -1 else ">5"
                        
                        # 计算竞争激烈程度 (Top 1 vs Top 2)
                        prob_gap = 0.0
                        if len(top_candidates) >= 2:
                            prob_gap = top_candidates[0]['prob'] - top_candidates[1]['prob']
                        
                        # === 写入该 Token 的详细分析块 ===
                        f.write(f"🔴 [High Entropy Token] Index: {idx} | Token: '{token_str}'\n")
                        f.write(f"   Entropy: {entropy:.4f} | Chosen Rank: {rank_str} | Chosen Prob: {chosen_prob:.4f}\n")
                        
                        # 分析结论字符串
                        analysis_note = ""
                        if prob_gap < 0.1:
                            analysis_note = ">> EXTREME CONFLICT (Top candidates are very close)"
                        elif chosen_rank > 1:
                            analysis_note = ">> UNUSUAL CHOICE (Model didn't pick the most likely token)"
                        
                        if analysis_note:
                            f.write(f"   {analysis_note}\n")
                        
                        f.write(f"   Candidate Distribution (Top 5):\n")
                        
                        # 绘制简单的条形图
                        for rank, cand in enumerate(top_candidates):
                            cand_tok = cand['token'].replace('\n', '\\n').strip()
                            cand_prob = cand['prob']
                            
                            # 标记哪个是被选中的
                            marker = "✅" if rank + 1 == chosen_rank else "  "
                            
                            # 简易 ASCII 条形图
                            bar_len = int(cand_prob * 20) 
                            bar_str = "█" * bar_len
                            
                            f.write(f"     {marker} {rank+1}. '{cand_tok:<15}' : {cand_prob:.4f} {bar_str}\n")
                        
                        f.write("\n") # 空行分隔 Token
                        
    print(f"✅ 深度分析报告已保存至: {file_path}")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading Data...")
    data = load_data(INPUT_FILE)
    
    print("Generating Deep Token Analysis Report...")
    save_deep_token_report(data, OUTPUT_DIR)
    
    print("\nDone!")

if __name__ == "__main__":
    main()