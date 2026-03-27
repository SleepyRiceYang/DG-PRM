import json
import numpy as np
import os
import re

# === 配置区域 ===
INPUT_FILE = "results_decoupled_2025-11-26T18-46-54.json"  # 替换为你的输入文件路径
OUTPUT_DIR = "analysis_strategy_conflicts"  # 结果输出目录
# =============

# 目标监控词集 (全部转为小写以便比较)
TARGET_VOCAB = {
    "logical", "emotion", "credibility", "foot", "self", 
    "personal", "donation", "source", "task", "appeal", "inquiry","related","story",
}

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

def clean_token_str(token_str):
    """
    清洗 Token 字符串以便匹配
    """
    t = token_str.strip().lower()
    t = t.replace('ġ', '').replace(' ', '')
    t = re.sub(r'[^\w\s]', '', t)
    return t

def is_target_word(token_str):
    """判断一个 Token 是否在目标词集中"""
    clean_t = clean_token_str(token_str)
    if not clean_t: return False
    return clean_t in TARGET_VOCAB

def save_target_vocab_analysis(data, output_dir):
    """
    核心分析函数：严格筛选策略词冲突
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_path = os.path.join(output_dir, "strict_strategy_conflict_report_1stf_0.6.txt")
    
    with open(file_path, "w", encoding="utf-8") as f:
        # === 写入报告头 ===
        f.write("="*80 + "\n")
        f.write("STRICT STRATEGY CONFLICT ANALYSIS REPORT\n")
        f.write("="*80 + "\n")
        f.write("筛选标准 (必须同时满足):\n")
        f.write("1. [高熵] Token 熵值 > Mean + Std\n")
        f.write("2. [本体] 被选中的 Token 必须在目标词集中\n")
        f.write("3. [竞品] Top-5 候选词中至少有一个在目标词集中 (且不能是本体)\n")
        f.write(f"目标词集: {TARGET_VOCAB}\n")
        f.write("="*80 + "\n\n")

        total_conflicts = 0

        for session_idx, session in enumerate(data):
            user_id = session.get('user_id', f'User_{session_idx}')
            history = session.get('detailed_history', [])
            
            for turn_idx, turn in enumerate(history):
                if turn['role'] == 'Persuader':
                    metrics = turn.get('strategy_entropy_metrics')
                    if not metrics: continue
                    
                    reasoning_m = segment_metrics(metrics)
                    if len(reasoning_m) < 5: continue
                    
                    entropies = [m['entropy'] for m in reasoning_m]
                    mean_val = np.mean(entropies)
                    std_val = np.std(entropies)
                    threshold = max(0,0)
                    
                    # === 逐个 Token 筛查 ===
                    for idx, (token_data, entropy) in enumerate(zip(reasoning_m, entropies)):
                        
                        current_token = token_data['token']
                        current_clean = clean_token_str(current_token)
                        top_candidates = token_data.get('top_5', [])
                        
                        # --- 条件 1: 高熵 ---
                        if entropy <= threshold:
                            continue

                        # --- 条件 2: 本体是目标词 ---
                        if not is_target_word(current_token):
                            continue
                            
                        # --- 条件 3: 竞品包含目标词 (且不是本体) ---
                        has_competitor = False
                        competitor_list = []
                        
                        for cand in top_candidates:
                            cand_clean = clean_token_str(cand['token'])
                            # 必须是目标词，且清洗后不能和本体一样 (排除 " Logic" 和 "Logic" 的假性竞争)
                            if is_target_word(cand['token']) and cand_clean != current_clean:
                                has_competitor = True
                                competitor_list.append(cand['token'].strip())
                        
                        if not has_competitor:
                            continue
                            
                        # === 满足所有条件，记录详细信息 ===
                        total_conflicts += 1
                        turn_id = f"{user_id}_Turn{turn.get('round', turn_idx+1)}"
                        strategy_name = turn.get('strategy', 'Unknown')
                        
                        f.write(f"Conflict Event #{total_conflicts} | Sample: {turn_id}\n")
                        f.write(f"Final Strategy: {strategy_name}\n")
                        f.write(f"Conflict Type : '{current_token.strip()}' vs {competitor_list}\n")
                        
                        # 上下文
                        start = max(0, idx - 5)
                        end = min(len(reasoning_m), idx + 6)
                        context_tokens = [m['token'] for m in reasoning_m[start:end]]
                        context_str = ""
                        for i, tok in enumerate(context_tokens):
                            if i == (idx - start):
                                context_str += f" [>>{tok.strip()}<<] "
                            else:
                                context_str += tok
                        f.write(f"Context       : ...{context_str.strip()}...\n")
                        
                        # 详细数据
                        f.write("-" * 60 + "\n")
                        f.write(f"Target Token  : '{current_token.strip()}'\n")
                        f.write(f"Entropy       : {entropy:.4f} (Threshold: {threshold:.4f})\n")
                        f.write("-" * 60 + "\n")
                        f.write("Top 5 Candidates Comparison:\n")
                        
                        chosen_rank = -1
                        for rank, cand in enumerate(top_candidates):
                            cand_tok = cand['token']
                            cand_clean = clean_token_str(cand_tok)
                            cand_prob = cand['prob']
                            
                            # 标记逻辑
                            # 🎯: 是目标词
                            is_target = is_target_word(cand_tok)
                            marker = "🎯" if is_target else "  " 
                            
                            # ✅: 是最终选中的词
                            is_chosen = (cand_tok.strip() == current_token.strip())
                            if is_chosen: chosen_rank = rank + 1
                            check_mark = "✅" if is_chosen else "  "
                            
                            # ⚔️: 是真正的竞争对手 (是目标词且不是本体)
                            is_competitor = is_target and (cand_clean != current_clean)
                            if is_competitor: marker = "⚔️" # 替换标记为双剑，表示冲突源
                            
                            bar = "█" * int(cand_prob * 20)
                            clean_tok_str = cand_tok.replace('\n', '\\n').strip()
                            
                            f.write(f" {marker} {check_mark} {rank+1}. '{clean_tok_str:<12}' : {cand_prob:.4f} {bar}\n")
                        
                        f.write("\n" + "="*80 + "\n\n")

    print(f"✅ 严格策略冲突分析已完成。")
    print(f"   共发现 {total_conflicts} 处 '策略内战' (High Entropy + Target Conflict)。")
    print(f"   报告已保存至: {file_path}")

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print("Loading Data...")
    data = load_data(INPUT_FILE)
    
    print("Running Strict Strategy Conflict Analysis...")
    save_target_vocab_analysis(data, OUTPUT_DIR)
    
    print("\nDone!")

if __name__ == "__main__":
    main()