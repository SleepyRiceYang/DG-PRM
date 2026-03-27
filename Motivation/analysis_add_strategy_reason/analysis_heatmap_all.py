import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from html import escape
import argparse # <--- 新增导入


# === 配置区域 ===
# 请替换为你最新的 all 格式输出文件
OUTPUT_DIR = "analysis_heatmap"     # 结果输出目录
# =============


# 在 analysis_heatmap_all.py 中添加以下函数用于解析新格式
def parse_model_output(full_text):
    """
    解析新的模型输出格式:
    <state_analysis>...</state_analysis>
    <strategy>Strategy Name: Strategy Reason</strategy>
    <response>...</response>
    """
    import re
    
    # 提取 state_analysis 部分
    state_analysis_pattern = r"<state_analysis>(.*?)</state_analysis>"
    state_analysis_match = re.search(state_analysis_pattern, full_text, re.DOTALL)
    state_analysis = state_analysis_match.group(1).strip() if state_analysis_match else ""
    
    # 提取 strategy 部分
    strategy_pattern = r"<strategy>(.*?)</strategy>"
    strategy_match = re.search(strategy_pattern, full_text, re.DOTALL)
    strategy_content = strategy_match.group(1).strip() if strategy_match else ""
    
    # 分离策略名称和原因
    if ':' in strategy_content:
        parts = strategy_content.split(':', 1)
        strategy_name = parts[0].strip()
        strategy_reason = parts[1].strip()
    else:
        strategy_name = strategy_content
        strategy_reason = ""
    
    # 提取 response 部分
    response_pattern = r"<response>(.*?)</response>"
    response_match = re.search(response_pattern, full_text, re.DOTALL)
    response = response_match.group(1).strip() if response_match else ""
    
    return {
        "state_analysis": state_analysis,
        "strategy_name": strategy_name,
        "strategy_reason": strategy_reason,
        "response": response
    }


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

# === 1. 自定义颜色映射 (白 -> 蓝 -> 红) ===
# 熵值低(自信) -> 白色
# 熵值中等 -> 蓝色
# 熵值高(困惑/多样性高) -> 红色
def create_custom_cmap():
    colors = ["#ffffff", "#3366ff", "#ff3333"] 
    nodes = [0.0, 0.5, 1.0]
    cmap = LinearSegmentedColormap.from_list("white_blue_red", list(zip(nodes, colors)))
    return cmap

CUSTOM_CMAP = create_custom_cmap()

def get_color_hex(value):
    """将归一化的熵值转换为 Hex 颜色代码"""
    rgba = CUSTOM_CMAP(value)
    return mcolors.to_hex(rgba)

def generate_interactive_html(samples, output_dir,remark=""):
    """
    生成交互式 HTML 热力图。
    功能：展示完整生成内容（Reasoning+Strategy+Response），点击 Token 查看 Top-5。
    """
    if not samples:
        print("无样本可生成热力图。")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    suffix = f"_{remark}" if remark else ""
    html_path = os.path.join(output_dir, f"interactive_heatmap_all{suffix}.html")
    
    # === HTML 头部 (包含 CSS 和 JS) ===
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>all Generation Entropy Analysis</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; padding: 20px; }
            .container { max_width: 1200px; margin: 0 auto; }
            
            /* 样本卡片样式 */
            .sample-box {
                background-color: white;
                border: 1px solid #ddd;
                padding: 25px;
                margin-bottom: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                line-height: 1.6; /* 稍微紧凑一点 */
                font-size: 16px;
                color: #333;
                position: relative;
            }
            .meta-info {
                font-size: 14px;
                color: #666;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid #f0f0f0;
                display: flex;
                justify-content: space-between;
                background: #fafafa;
                padding: 10px;
                border-radius: 8px;
            }
            
            /* Token 样式 */
            .token {
                display: inline-block; /* 改为 inline-block 以便更好地处理空格 */
                white-space: pre-wrap; /* 保留空格和换行 */
                padding: 1px 0;
                cursor: pointer;
                transition: all 0.1s;
                border-radius: 2px;
            }
            .token:hover {
                outline: 2px solid #333;
                z-index: 10;
                position: relative;
            }

            /* 换行符的特殊处理 */
            .token-newline {
                display: block;
                height: 10px;
                width: 100%;
                margin-bottom: 2px;
            }

            /* 悬浮窗 (Info Box) 样式 */
            #info-box {
                display: none;
                position: absolute;
                background: white;
                border: 1px solid #ccc;
                box-shadow: 0 8px 24px rgba(0,0,0,0.25);
                border-radius: 8px;
                padding: 15px;
                z-index: 1000;
                width: 320px;
                font-size: 14px;
            }
            #info-box h4 {
                margin: 0 0 10px 0;
                font-size: 16px;
                border-bottom: 1px solid #eee;
                padding-bottom: 5px;
                word-break: break-all;
            }
            
            /* 候选词表格样式 */
            .cand-table { width: 100%; border-collapse: collapse; }
            .cand-table td { padding: 4px; border-bottom: 1px solid #f5f5f5; }
            .cand-token { font-family: monospace; font-weight: bold; color: #2c3e50; white-space: pre-wrap;}
            .cand-prob { text-align: right; color: #666; font-size: 12px; width: 60px;}
            .bar-container { width: 80px; vertical-align: middle; }
            .bar-fill { height: 6px; background-color: #3366ff; border-radius: 3px; opacity: 0.7; }
            .highlight-row { background-color: #f0f8ff; }

            /* 图例样式 */
            .legend {
                position: sticky;
                top: 0;
                background: rgba(240, 242, 245, 0.95);
                padding: 15px 0;
                margin-bottom: 20px;
                text-align: center;
                z-index: 900;
                backdrop-filter: blur(5px);
                border-bottom: 1px solid #ddd;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .legend-item {
                display: inline-block;
                padding: 5px 15px;
                margin: 0 10px;
                font-size: 13px;
                font-weight: bold;
                border-radius: 4px;
                color: white;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            }
        </style>
    </head>
    <body>
    
    <div id="info-box"></div>

    <div class="container">
        <h2 style="text-align: center; color: #333;">all Model Output Entropy Analysis</h2>
        <p style="text-align: center; color: #666;">Visualization of Reasoning Process, Strategy Selection, and Response Generation.</p>
        
        <div class="legend">
            <span class="legend-item" style="background: #ffffff; color: #333; border: 1px solid #ccc; text-shadow: none;">Low Entropy (Certain)</span>
            <span class="legend-item" style="background: #3366ff;">Medium Entropy</span>
            <span class="legend-item" style="background: #ff3333;">High Entropy (Uncertain)</span>
        </div>
    """

    for sample in samples:
        metrics = sample['metrics']
        if not metrics: continue
        
        entropies = [m['entropy'] for m in metrics]
        
        # 归一化处理：使得颜色分布更均匀
        # 使用 mean + 2*std 作为显示的上限，避免个别极高值导致整体偏白
        mean_val = np.mean(entropies)
        std_val = np.std(entropies)
        vmax = max(mean_val + 2.5 * std_val, 0.5) # 动态调整上限
        norm = plt.Normalize(vmin=0, vmax=vmax)
        
        html_content += f"""
        <div class="sample-box">
            <div class="meta-info">
                <span><strong>User:</strong> {sample['user_id']}</span>
                <span><strong>Turn:</strong> {sample['turn_num']}</span>
                <span><strong>Selected Strategy:</strong> {sample['strategy']}</span>
                <span><strong>Avg Entropy:</strong> {mean_val:.4f}</span>
            </div>
            <div class="text-content">
        """
        
        for m in metrics:
            token_text = m['token']
            entropy = m['entropy']
            top_5 = m.get('top_5', [])
            
            # 准备 Top-5 数据给 JS
            token_data = {
                "token": token_text,
                "entropy": round(entropy, 4),
                "prob": round(top_5[0]['prob'], 4) if top_5 else 0,
                "candidates": []
            }
            for cand in top_5:
                token_data['candidates'].append({
                    "t": cand['token'],
                    "p": round(cand['prob'], 4)
                })
            
            # JSON 序列化
            json_str = json.dumps(token_data).replace('"', '&quot;')
            
            # 计算颜色
            normalized_val = norm(entropy)
            color_hex = get_color_hex(normalized_val)
            
            # 字体颜色反转：背景深色时字体变白
            text_color = "white" if normalized_val > 0.5 else "black"
            
            # HTML 转义 (但不转义换行符，因为我们要特殊处理)
            safe_text = escape(token_text)
            
            if '\n' in token_text:
                # 如果包含换行符，且不仅仅是换行符（例如 "word\n"），拆分处理或简单地作为块处理
                # 这里简单处理：如果有换行符，强制换行
                # 对于纯换行符 token
                if token_text.strip() == "":
                    html_content += f'<span class="token-newline"></span>'
                    continue
            
            html_content += f"""
            <span class="token" 
                  style="background-color: {color_hex}; color: {text_color};" 
                  onclick="showTokenInfo(this, event)"
                  data-info="{json_str}">{safe_text}</span>"""
            
        html_content += """
            </div>
        </div>
        """

    # === JavaScript 逻辑 ===
    html_content += """
    </div>
    <script>
        const infoBox = document.getElementById('info-box');
        let currentOpenToken = null;

        function showTokenInfo(element, event) {
            event.stopPropagation();
            if (currentOpenToken === element) {
                closeInfoBox();
                return;
            }
            currentOpenToken = element;

            const dataStr = element.getAttribute('data-info');
            const data = JSON.parse(dataStr);
            
            let html = `
                <h4>Token: <span style="background:#eee; padding:2px 5px; border-radius:3px;">${escapeHtml(data.token)}</span></h4>
                <div style="margin-bottom:10px; font-size:13px; color:#555;">
                    Entropy: <strong>${data.entropy}</strong> | Top-1 Prob: <strong>${data.prob}</strong>
                </div>
                <table class="cand-table">
            `;
            
            data.candidates.forEach((cand, index) => {
                const mark = (index === 0) ? '✅' : '#' + (index + 1);
                // 简单的判断高亮
                const isChosen = (index === 0); 
                const rowClass = isChosen ? 'highlight-row' : '';
                
                html += `
                    <tr class="${rowClass}">
                        <td style="color:#888; font-size:11px;">${mark}</td>
                        <td class="cand-token">${escapeHtml(cand.t)}</td>
                        <td class="cand-prob">${cand.p.toFixed(4)}</td>
                        <td class="bar-container">
                            <div class="bar-fill" style="width: ${cand.p * 100}%"></div>
                        </td>
                    </tr>
                `;
            });
            html += `</table>`;
            
            infoBox.innerHTML = html;
            infoBox.style.display = 'block';
            
            const x = event.pageX + 10;
            const y = event.pageY + 10;
            infoBox.style.left = x + 'px';
            infoBox.style.top = y + 'px';
        }

        function closeInfoBox() {
            infoBox.style.display = 'none';
            currentOpenToken = null;
        }
        
        function escapeHtml(text) {
            if (!text) return "";
            return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
        }

        document.addEventListener('click', function(event) {
            if (infoBox.style.display === 'block') {
                closeInfoBox();
            }
        });
        
        infoBox.addEventListener('click', function(event) { event.stopPropagation(); });
    </script>
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ 交互式热力图已生成: {html_path}")

# 修改 process_all_samples 函数中关于 strategy 的处理部分
def process_all_samples(data):
    """
    提取完整样本 (Reasoning + Strategy + Response)。
    特别处理：Turn 的逻辑计数（Persuader + Persuadee = 1 Turn）。
    """
    samples = []
    
    for session_idx, session in enumerate(data):
        user_id = session.get('user_id', f'User_{session_idx}')
        history = session.get('detailed_history', [])
        
        # === 手动计数 Round ===
        # 逻辑：每次 Persuader 说话，就是一个新的 Turn 的开始
        manual_turn_count = 0 
        
        for turn in history:
            role = turn['role']
            
            if role == 'Persuader':
                manual_turn_count += 1
                
                # 适配新格式：直接获取 'metrics'
                metrics = turn.get('metrics')
                
                # 兼容性检查：如果 metrics 为空，尝试旧格式
                if not metrics:
                    strat_m = turn.get('strategy_entropy_metrics', [])
                    resp_m = turn.get('response_entropy_metrics', [])
                    if strat_m or resp_m:
                        metrics = (strat_m or []) + (resp_m or [])
                
                if not metrics or len(metrics) < 1:
                    continue

                # 处理 strategy 字段，支持新格式 "Name: Reason" 的情况
                strategy_raw = turn.get('strategy_name', 'Unknown')
                if ':' in strategy_raw:
                    strategy_name = strategy_raw.split(':', 1)[0].strip()
                else:
                    strategy_name = strategy_raw

                samples.append({
                    "user_id": user_id,
                    "turn_num": manual_turn_count, # 使用手动计数的 round
                    "strategy": strategy_name,  # 只使用策略名称，不包含原因
                    "metrics": metrics
                })
                
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--remark", default="")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found.")
        return
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print("Loading Data...")
    raw_data = load_data(args.input)[:3]
    
    print("Processing samples...")
    samples = process_all_samples(raw_data)
    
    if not samples:
        print("未找到包含 metrics 的数据。")
        return

    print(f"Generating Heatmap for {len(samples)} turns...")
    generate_interactive_html(samples, args.output_dir, args.remark)
    
    print("\nDone!")

if __name__ == "__main__":
    main()# 在 analysis_heatmap_all.py 中添加以下函数用于解析新格式
def parse_model_output(full_text):
    """
    解析新的模型输出格式:
    <state_analysis>...</state_analysis>
    <strategy>Strategy Name: Strategy Reason</strategy>
    <response>...</response>
    """
    import re
    
    # 提取 state_analysis 部分
    state_analysis_pattern = r"<state_analysis>(.*?)</state_analysis>"
    state_analysis_match = re.search(state_analysis_pattern, full_text, re.DOTALL)
    state_analysis = state_analysis_match.group(1).strip() if state_analysis_match else ""
    
    # 提取 strategy 部分
    strategy_pattern = r"<strategy>(.*?)</strategy>"
    strategy_match = re.search(strategy_pattern, full_text, re.DOTALL)
    strategy_content = strategy_match.group(1).strip() if strategy_match else ""
    
    # 分离策略名称和原因
    if ':' in strategy_content:
        parts = strategy_content.split(':', 1)
        strategy_name = parts[0].strip()
        strategy_reason = parts[1].strip()
    else:
        strategy_name = strategy_content
        strategy_reason = ""
    
    # 提取 response 部分
    response_pattern = r"<response>(.*?)</response>"
    response_match = re.search(response_pattern, full_text, re.DOTALL)
    response = response_match.group(1).strip() if response_match else ""
    
    return {
        "state_analysis": state_analysis,
        "strategy_name": strategy_name,
        "strategy_reason": strategy_reason,
        "response": response
    }