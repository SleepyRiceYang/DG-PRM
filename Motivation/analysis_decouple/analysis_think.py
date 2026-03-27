import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from html import escape
import re

# === 配置区域 ===
INPUT_FILE = "results_decoupled_2025-11-29T11-02-41.json"  # 替换为你的输入文件
OUTPUT_DIR = "analysis_all_samples_vis"     # 结果输出目录
# =============

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# === 1. 自定义颜色映射 (白 -> 蓝 -> 红) ===
def create_custom_cmap():
    colors = ["#ffffff", "#3366ff", "#ff3333"] 
    nodes = [0.0, 0.5, 1.0]
    cmap = LinearSegmentedColormap.from_list("white_blue_red", list(zip(nodes, colors)))
    return cmap

CUSTOM_CMAP = create_custom_cmap()

def get_color_hex(value):
    rgba = CUSTOM_CMAP(value)
    return mcolors.to_hex(rgba)

def generate_interactive_html(samples, output_dir):
    """
    生成交互式 HTML 热力图。
    功能：点击 Token 弹出悬浮窗，显示 Top-5 候选词及概率条形图。
    """
    if not samples:
        print("无样本可生成热力图。")
        return

    html_path = os.path.join(output_dir, "interactive_heatmap_strategy_11_29_1.html")
    
    # === HTML 头部 (包含 CSS 和 JS) ===
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Interactive Entropy Analysis</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f0f2f5; padding: 20px; }
            .container { max_width: 1100px; margin: 0 auto; }
            
            /* 样本卡片样式 */
            .sample-box {
                background-color: white;
                border: 1px solid #ddd;
                padding: 25px;
                margin-bottom: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                line-height: 1.8;
                font-size: 18px;
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
            }
            
            /* Token 样式 */
            .token {
                display: inline;
                padding: 2px 1px;
                border-radius: 3px;
                cursor: pointer;
                transition: outline 0.1s;
                position: relative;
            }
            .token:hover {
                outline: 2px solid #333;
                z-index: 10;
            }

            /* 悬浮窗 (Info Box) 样式 */
            #info-box {
                display: none;
                position: absolute;
                background: white;
                border: 1px solid #ccc;
                box-shadow: 0 8px 24px rgba(0,0,0,0.2);
                border-radius: 8px;
                padding: 15px;
                z-index: 1000;
                width: 320px;
                font-size: 14px;
                pointer-events: none; /* 让鼠标事件穿透，或者设为 auto 允许复制 */
                pointer-events: auto;
            }
            #info-box h4 {
                margin: 0 0 10px 0;
                font-size: 16px;
                border-bottom: 1px solid #eee;
                padding-bottom: 5px;
            }
            
            /* 候选词表格样式 */
            .cand-table {
                width: 100%;
                border-collapse: collapse;
            }
            .cand-table td {
                padding: 4px;
                border-bottom: 1px solid #f5f5f5;
            }
            .cand-token { font-family: monospace; font-weight: bold; color: #2c3e50; }
            .cand-prob { text-align: right; color: #666; font-size: 12px; width: 60px;}
            .bar-container { width: 80px; vertical-align: middle; }
            .bar-fill {
                height: 6px;
                background-color: #3366ff;
                border-radius: 3px;
                opacity: 0.7;
            }
            .highlight-row { background-color: #f0f8ff; } /* 选中词的高亮 */

            /* 图例样式 */
            .legend {
                position: sticky;
                top: 0;
                background: rgba(240, 242, 245, 0.95);
                padding: 10px 0;
                margin-bottom: 20px;
                text-align: center;
                z-index: 900;
                backdrop-filter: blur(5px);
                border-bottom: 1px solid #ddd;
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
    
    <!-- 悬浮窗容器 -->
    <div id="info-box"></div>

    <div class="container">
        <h2 style="text-align: center; color: #333;">Interactive Entropy Analysis</h2>
        <p style="text-align: center; color: #666;">Click on any token to inspect top candidates.</p>
        
        <div class="legend">
            <span class="legend-item" style="background: #ffffff; color: #333; border: 1px solid #ccc; text-shadow: none;">Low Entropy (Confident)</span>
            <span class="legend-item" style="background: #3366ff;">Medium (Hesitant)</span>
            <span class="legend-item" style="background: #ff3333;">High Entropy (Confused)</span>
        </div>
    """

    for sample in samples:
        metrics = sample['metrics']
        if not metrics: continue
        
        entropies = [m['entropy'] for m in metrics]
        
        # 归一化
        mean_val = np.mean(entropies)
        std_val = np.std(entropies)
        vmax = max(mean_val + 2 * std_val, 1.0)
        norm = plt.Normalize(vmin=0, vmax=vmax)
        
        html_content += f"""
        <div class="sample-box">
            <div class="meta-info">
                <span><strong>ID:</strong> {sample['id']}</span>
                <span><strong>Strategy:</strong> {sample['strategy']}</span>
                <span><strong>Avg Entropy:</strong> {mean_val:.4f}</span>
            </div>
            <div class="text-content">
        """
        
        for m in metrics:
            token_text = m['token']
            entropy = m['entropy']
            top_5 = m.get('top_5', [])
            
            # 准备数据给 JS (序列化并转义)
            # 构造一个简单的 dict 传给前端
            token_data = {
                "token": token_text,
                "entropy": round(entropy, 4),
                "prob": round(top_5[0]['prob'], 4) if top_5 else 0,
                "candidates": []
            }
            
            # 处理候选词数据
            for cand in top_5:
                token_data['candidates'].append({
                    "t": cand['token'],
                    "p": round(cand['prob'], 4)
                })
            
            # 转为 JSON 字符串并转义，以便放入 data-info 属性
            json_str = json.dumps(token_data).replace('"', '&quot;')
            
            # 颜色计算
            normalized_val = norm(entropy)
            color_hex = get_color_hex(normalized_val)
            text_color = "white" if normalized_val > 0.4 else "black"
            
            # 简单的 title 提示，点击显示详情
            safe_text = escape(token_text).replace('\n', '<br>')
            
            html_content += f"""
            <span class="token" 
                  style="background-color: {color_hex}; color: {text_color};" 
                  onclick="showTokenInfo(this, event)"
                  data-info="{json_str}">
                  {safe_text}
            </span>"""
            
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

        // 点击 Token 显示详情
        function showTokenInfo(element, event) {
            event.stopPropagation(); // 防止冒泡到 document 点击关闭
            
            // 如果点击同一个，则关闭
            if (currentOpenToken === element) {
                closeInfoBox();
                return;
            }
            currentOpenToken = element;

            // 获取数据
            const dataStr = element.getAttribute('data-info');
            const data = JSON.parse(dataStr);
            
            // 构建 HTML 内容
            let html = `
                <h4>Current Token: <span style="background:#eee; padding:2px 5px; border-radius:3px;">${escapeHtml(data.token)}</span></h4>
                <div style="margin-bottom:10px; font-size:13px; color:#555;">
                    Entropy: <strong>${data.entropy}</strong> | Top-1 Prob: <strong>${data.prob}</strong>
                </div>
                <table class="cand-table">
            `;
            
            // 遍历候选词
            data.candidates.forEach((cand, index) => {
                const percent = (cand.p * 100).toFixed(1);
                // 判断是否是当前选中的词 (简单的字符串对比)
                const isChosen = (cand.t.trim() === data.token.trim());
                const rowClass = isChosen ? 'highlight-row' : '';
                const mark = isChosen ? '✅' : '#' + (index + 1);
                
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
            
            // 填充内容
            infoBox.innerHTML = html;
            infoBox.style.display = 'block';
            
            // 计算位置 (优先显示在鼠标点击位置的右下方，如果靠边则调整)
            const x = event.pageX + 10;
            const y = event.pageY + 10;
            infoBox.style.left = x + 'px';
            infoBox.style.top = y + 'px';
        }

        function closeInfoBox() {
            infoBox.style.display = 'none';
            currentOpenToken = null;
        }
        
        // 简单的 HTML 转义工具
        function escapeHtml(text) {
            if (!text) return "";
            return text
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;")
                .replace(/\\n/g, "↵"); // 将换行显示为符号
        }

        // 点击空白处关闭悬浮窗
        document.addEventListener('click', function(event) {
            if (infoBox.style.display === 'block') {
                closeInfoBox();
            }
        });
        
        // 防止点击悬浮窗内部时关闭
        infoBox.addEventListener('click', function(event) {
            event.stopPropagation();
        });

    </script>
    </body>
    </html>
    """

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"✅ 交互式热力图已生成: {html_path}")
    print("   请用浏览器打开，点击 Token 查看 Top-5 候选词。")

def process_all_samples(data):
    """提取完整样本 (Think + Strategy)"""
    samples = []
    for session_idx, session in enumerate(data):
        user_id = session.get('user_id', f'User_{session_idx}')
        history = session.get('detailed_history', [])
    
        for turn_idx, turn in enumerate(history):
            if turn['role'] == 'Persuader':
                metrics = turn.get('strategy_entropy_metrics')
                if not metrics or len(metrics) < 5: continue
                samples.append({
                    "id": f"{user_id}_Turn{turn.get('round', turn_idx+1)}",
                    "strategy": turn.get('strategy', 'Unknown'),
                    "metrics": metrics
                })
    return samples

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Loading Data...")
    raw_data = load_data(INPUT_FILE)
    
    print("Processing samples...")
    samples = process_all_samples(raw_data)
    
    print(f"Generating Interactive Heatmap for {len(samples)} samples...")
    generate_interactive_html(samples, OUTPUT_DIR)
    
    print("\nDone!")

if __name__ == "__main__":
    main()