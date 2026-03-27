import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from html import escape
import re
import argparse

# === 配置区域 ===
OUTPUT_DIR = "analysis_heatmap"         # 输出目录
# =============

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: 文件 {file_path} 不存在。")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            # 必须先执行加载赋值
            exp_data = json.load(f) 
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return []

    # 加载后再进行类型检查
    if isinstance(exp_data, dict):
        # 如果 JSON 是字典格式，尝试提取 'results' 键，否则将其放入列表
        return exp_data.get('results', [exp_data])
    
    return exp_data

# === 颜色映射配置 (白 -> 蓝 -> 红) ===
def create_custom_cmap():
    colors = ["#ffffff", "#3366ff", "#ff3333"] 
    nodes = [0.0, 0.5, 1.0]
    cmap = LinearSegmentedColormap.from_list("white_blue_red", list(zip(nodes, colors)))
    return cmap

CUSTOM_CMAP = create_custom_cmap()

def get_color_hex(value):
    rgba = CUSTOM_CMAP(value)
    return mcolors.to_hex(rgba)

def strict_segmentation(metrics):
    """
    严格切分 metrics 为三个部分:
    1. State_Analysis: <state_analysis> 标签之后 -> </state_analysis> 标签之前
    2. Strategy: <strategy> 标签之后 -> </strategy> 标签之前
    3. Response: <response> 标签之后 -> </response> 标签之前
    """
    # 将所有token组合成文本，方便查找
    tokens_text = "".join([m['token'] for m in metrics])
    
    # 查找标签的字符级位置
    state_analysis_start = tokens_text.find("<state_analysis>")
    state_analysis_end = tokens_text.find("</state_analysis>")
    strategy_start = tokens_text.find("<strategy>")
    strategy_end = tokens_text.find("</strategy>")
    response_start = tokens_text.find("<response>")
    response_end = tokens_text.find("</response>")
    
    state_analysis_metrics = []
    strategy_metrics = []
    response_metrics = []
    
    # 提取 state_analysis 部分
    if state_analysis_start != -1 and state_analysis_end != -1:
        start_index = -1
        end_index = -1
        current_pos = 0
        content_start_pos = state_analysis_start + len("<state_analysis>")
        
        for i, item in enumerate(metrics):
            # *** FIX START: 修正起始索引的计算 ***
            if start_index == -1 and current_pos >= content_start_pos:
                start_index = i
            # *** FIX END ***

            if end_index == -1 and current_pos <= state_analysis_end < current_pos + len(item['token']):
                end_index = i
            
            current_pos += len(item['token'])
            if start_index != -1 and end_index != -1:
                 break
        
        if start_index != -1 and end_index != -1 and start_index < end_index:
            state_analysis_metrics = metrics[start_index:end_index]

    # 提取 strategy 部分
    if strategy_start != -1 and strategy_end != -1:
        start_index = -1
        end_index = -1
        current_pos = 0
        content_start_pos = strategy_start + len("<strategy>")
        
        for i, item in enumerate(metrics):
            # *** FIX START: 修正起始索引的计算 ***
            if start_index == -1 and current_pos >= content_start_pos:
                start_index = i
            # *** FIX END ***
            
            if end_index == -1 and current_pos <= strategy_end < current_pos + len(item['token']):
                end_index = i
            
            current_pos += len(item['token'])
            if start_index != -1 and end_index != -1:
                break
        
        if start_index != -1 and end_index != -1 and start_index < end_index:
            strategy_metrics = metrics[start_index:end_index]

    # 提取 response 部分
    if response_start != -1 and response_end != -1:
        start_index = -1
        end_index = -1
        current_pos = 0
        content_start_pos = response_start + len("<response>")
        
        for i, item in enumerate(metrics):
            # *** FIX START: 修正起始索引的计算 ***
            if start_index == -1 and current_pos >= content_start_pos:
                start_index = i
            # *** FIX END ***
            
            if end_index == -1 and current_pos <= response_end < current_pos + len(item['token']):
                end_index = i

            current_pos += len(item['token'])
            if start_index != -1 and end_index != -1:
                break

        if start_index != -1 and end_index != -1 and start_index < end_index:
            response_metrics = metrics[start_index:end_index]
            
    return state_analysis_metrics, strategy_metrics, response_metrics

def generate_html_file(samples, filename_base, title, output_dir, remark=""):
    """通用的 HTML 生成函数"""
    if not samples:
        print(f"Skipping {filename_base}: No samples.")
        return

    # 构造带 remark 的文件名
    name_part, ext = os.path.splitext(filename_base)
    suffix = f"_{remark}" if remark else ""
    filename = f"{name_part}{suffix}{ext}"
    
    filepath = os.path.join(output_dir, filename)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f8; padding: 20px; }}
            .container {{ max-width: 1100px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .sample-box {{
                background-color: white; border: 1px solid #e0e0e0; margin-bottom: 25px;
                border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); padding: 20px;
            }}
            .meta-info {{
                font-size: 14px; color: #666; margin-bottom: 15px; padding-bottom: 10px;
                border-bottom: 1px solid #eee; display: flex; justify-content: space-between;
            }}
            .token {{
                display: inline-block; white-space: pre-wrap; padding: 1px 0;
                cursor: pointer; border-radius: 2px; transition: 0.1s;
            }}
            .token:hover {{ outline: 2px solid #333; z-index: 10; position: relative; }}
            .token-newline {{ display: block; height: 8px; width: 100%; }}
            
            #info-box {{
                display: none; position: absolute; background: white; border: 1px solid #ccc;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2); border-radius: 6px; padding: 10px;
                z-index: 1000; width: 280px; font-size: 13px;
            }}
            .legend {{
                position: sticky; top: 0; background: rgba(244, 246, 248, 0.95);
                padding: 10px; text-align: center; z-index: 900; border-bottom: 1px solid #ccc;
            }}
            .legend span {{ display: inline-block; width: 100px; height: 10px; margin: 0 5px; }}
        </style>
    </head>
    <body>
    <div id="info-box"></div>
    <div class="container">
        <div class="legend">
            Low Entropy (Certain) 
            <span style="background: linear-gradient(to right, white, #3366ff);"></span> 
            <span style="background: linear-gradient(to right, #3366ff, #ff3333);"></span> 
            High Entropy (Uncertain)
        </div>
        <div class="header"><h2>{title}</h2></div>
    """

    for s in samples:
        metrics = s['metrics']
        if not metrics: continue
        
        # 计算统计量用于归一化
        entropies = [m['entropy'] for m in metrics]
        mean_val = np.mean(entropies)
        std_val = np.std(entropies)
        vmax = max(mean_val + 2.5 * std_val, 0.5)
        norm = plt.Normalize(vmin=0, vmax=vmax)

        html_content += f"""
        <div class="sample-box">
            <div class="meta-info">
                <span><b>User:</b> {s['user_id']}</span>
                <span><b>Turn:</b> {s['turn']}</span>
                <span><b>Strategy:</b> {s['strategy']}</span>
                <span><b>Avg Entropy:</b> {mean_val:.4f}</span>
            </div>
            <div class="content">
        """

        for m in metrics:
            token = m['token']
            entropy = m['entropy']
            top_5 = m.get('top_5', [])
            
            # 颜色
            color = get_color_hex(norm(entropy))
            text_color = "white" if norm(entropy) > 0.5 else "black"
            
            # 数据
            token_data = {
                "t": token, "e": round(entropy, 4), "p": round(top_5[0]['prob'], 4) if top_5 else 0,
                "c": [{"t": c['token'], "p": round(c['prob'], 4)} for c in top_5]
            }
            json_str = json.dumps(token_data).replace('"', '&quot;')
            
            safe_text = escape(token)
            if token.strip() == "" and '\n' in token:
                html_content += '<span class="token-newline"></span>'
                continue
                
            html_content += f'<span class="token" style="background:{color}; color:{text_color};" onclick="show(this, event)" data-info="{json_str}">{safe_text}</span>'
            
        html_content += "</div></div>"

    # JS 部分
    html_content += """
    </div>
    <script>
        const box = document.getElementById('info-box');
        function show(el, e) {
            e.stopPropagation();
            const d = JSON.parse(el.getAttribute('data-info'));
            let h = `<b>'${esc(d.t)}'</b><br>Ent: ${d.e} | Prob: ${d.p}<hr style='margin:5px 0'>`;
            d.c.forEach((c,i) => {
                h += `<div style='display:flex;justify-content:space-between;font-size:12px;font-family:monospace'>
                      <span>${i+1}. ${esc(c.t)}</span><span>${c.p.toFixed(3)}</span></div>`;
            });
            box.innerHTML = h; box.style.display='block';
            box.style.left=(e.pageX+10)+'px'; box.style.top=(e.pageY+10)+'px';
        }
        function esc(t){return t?t.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"):""}
        document.onclick = () => box.style.display='none';
    </script>
    </body></html>
    """
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✅ 生成文件: {filepath}")

def process_and_generate(data, output_dir, remark=""):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    state_analysis_samples = []
    strategy_samples = []
    response_samples = []
    
    print("Processing data...")
    for session_idx, session in enumerate(data):
        user_id = session.get('user_id', f'User_{session_idx}')
        history = session.get('detailed_history', [])
        
        manual_turn = 0
        for turn in history:
            if turn['role'] == 'Persuader':
                manual_turn += 1
                
                # 获取完整 metrics
                raw_metrics = turn.get('metrics')
                if not raw_metrics: 
                    print(f"Warning: No metrics found for user {user_id}, turn {manual_turn}")
                    continue # 跳过无 metrics 数据
                
                strategy_name = turn.get('strategy_name', 'Unknown')
                
                # === 切分 ===
                m_state_analysis, m_strategy, m_response = strict_segmentation(raw_metrics)
                
                # (可选) 打印切分结果用于调试
                # print(f"User {user_id}, Turn {manual_turn}: "
                #       f"State Analysis tokens: {len(m_state_analysis)}, "
                #       f"Strategy tokens: {len(m_strategy)}, "
                #       f"Response tokens: {len(m_response)}")
                
                # 存入 State_Analysis 列表 (降低限制到>=1)
                if len(m_state_analysis) >= 1:
                    state_analysis_samples.append({
                        "user_id": user_id, "turn": manual_turn, "strategy": strategy_name,
                        "metrics": m_state_analysis
                    })
                
                # 存入 Strategy 列表 (降低限制到>=1)
                if len(m_strategy) >= 1:
                    strategy_samples.append({
                        "user_id": user_id, "turn": manual_turn, "strategy": strategy_name,
                        "metrics": m_strategy
                    })
                
                # 存入 Response 列表 (降低限制到>=1)
                if len(m_response) >= 1:
                    response_samples.append({
                        "user_id": user_id, "turn": manual_turn, "strategy": strategy_name,
                        "metrics": m_response
                    })

    # 生成三个独立文件
    print(f"Generating State Analysis Heatmap ({len(state_analysis_samples)} samples)...")
    generate_html_file(state_analysis_samples, "analysis_state_analysis_heatmap.html", "State Analysis Process Entropy", output_dir, remark)
    
    print(f"Generating Strategy Heatmap ({len(strategy_samples)} samples)...")
    if len(strategy_samples) == 0:
        print("Warning: No strategy samples found. Check if <strategy> tags are correctly parsed.")
    generate_html_file(strategy_samples, "analysis_strategy_heatmap.html", "Strategy Selection Entropy", output_dir, remark)
    
    print(f"Generating Response Heatmap ({len(response_samples)} samples)...")
    generate_html_file(response_samples, "analysis_response_heatmap.html", "Response Generation Entropy", output_dir, remark)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--remark", default="")
    args = parser.parse_args()

    data = load_data(args.input)
    if not data: return
    process_and_generate(data, args.output_dir, args.remark)

if __name__ == "__main__":
    main()