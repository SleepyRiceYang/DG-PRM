import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import argparse
import math
from matplotlib.backends.backend_pdf import PdfPages

# === 全局样式配置 (与 Token Level 保持一致) ===
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义标准颜色和样式
TYPE_STYLES = {
    "State_Variance": {'marker': 'o', 'linestyle': '-', 'color': '#1f77b4', 'label': 'State Variance (Var(H))'},
    "Action_Variance": {'marker': 's', 'linestyle': '--', 'color': '#ff7f0e', 'label': 'Action Variance (Var(H))'}
}

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: 文件 {file_path} 不存在")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def segment_metrics_new_format(metrics):
    """
    根据新格式切分 State Analysis、Strategy 和 Response 部分
    格式: <state_analysis>...</state_analysis><strategy>...</strategy><response>...</response>
    """
    # 将所有token组合成文本，方便查找
    tokens_text = "".join([m['token'] for m in metrics])
    
    # 查找各部分的位置
    state_analysis_start = tokens_text.find("<state_analysis>")
    state_analysis_end = tokens_text.find("</state_analysis>")
    
    strategy_start = tokens_text.find("<strategy>")
    strategy_end = tokens_text.find("</strategy>")
    
    response_start = tokens_text.find("<response>")
    response_end = tokens_text.find("</response>")
    
    state_analysis_part = []
    strategy_part = []
    response_part = []
    
    # 提取 State Analysis 部分
    if state_analysis_start != -1 and state_analysis_end != -1:
        start_index = -1
        end_index = -1
        current_pos = 0
        content_start_pos = state_analysis_start + len("<state_analysis>")
        
        for i, item in enumerate(metrics):
            # 修正后的逻辑：找到第一个起始位置 >= 标签结束位置的 token
            if start_index == -1 and current_pos >= content_start_pos:
                start_index = i
            
            # 结束索引的逻辑是正确的
            if end_index == -1 and current_pos <= state_analysis_end < current_pos + len(item['token']):
                end_index = i

            current_pos += len(item['token'])
            if start_index != -1 and end_index != -1:
                 break
        
        if start_index != -1 and end_index != -1 and start_index < end_index:
            state_analysis_part = metrics[start_index:end_index]
    
    # 提取 Strategy 部分
    if strategy_start != -1 and strategy_end != -1:
        start_index = -1
        end_index = -1
        current_pos = 0
        content_start_pos = strategy_start + len("<strategy>")

        for i, item in enumerate(metrics):
            # 修正后的逻辑
            if start_index == -1 and current_pos >= content_start_pos:
                start_index = i
            
            if end_index == -1 and current_pos <= strategy_end < current_pos + len(item['token']):
                end_index = i
            
            current_pos += len(item['token'])
            if start_index != -1 and end_index != -1:
                break
        
        if start_index != -1 and end_index != -1 and start_index < end_index:
            strategy_part = metrics[start_index:end_index]
    
    # 提取 Response 部分
    if response_start != -1 and response_end != -1:
        start_index = -1
        end_index = -1
        current_pos = 0
        content_start_pos = response_start + len("<response>")
        
        for i, item in enumerate(metrics):
            # 修正后的逻辑
            if start_index == -1 and current_pos >= content_start_pos:
                start_index = i
            
            if end_index == -1 and current_pos <= response_end < current_pos + len(item['token']):
                end_index = i

            current_pos += len(item['token'])
            if start_index != -1 and end_index != -1:
                break

        if start_index != -1 and end_index != -1 and start_index < end_index:
            response_part = metrics[start_index:end_index]
            
    return state_analysis_part, strategy_part, response_part
def collect_varentropy_records(data):
    all_records = []
    for session in data:
        user_id = session.get('user_id', 'Unknown_User')
        success_status = session.get('success', False)
        history = session.get('detailed_history', [])
        manual_turn = 0
        for turn in history:
            if turn['role'] == 'Persuader':
                manual_turn += 1
                metrics = turn.get('metrics')
                if not metrics: continue
                state_m, strategy_m, response_m = segment_metrics_new_format(metrics)
                if state_m:
                    entropies = [m['entropy'] for m in state_m]
                    if len(entropies) > 1:
                        all_records.append({"User": user_id, "Turn": manual_turn, "Type": "State_Variance", "Variance": np.var(entropies), "Success": success_status})
                action_m = strategy_m + response_m
                if action_m:
                    entropies = [m['entropy'] for m in action_m]
                    if len(entropies) > 1:
                        all_records.append({"User": user_id, "Turn": manual_turn, "Type": "Action_Variance", "Variance": np.var(entropies), "Success": success_status})
    return pd.DataFrame(all_records)

def plot_varentropy_to_pdf(df, output_dir, remark=""):
    if df.empty: return
    users = df['User'].unique()
    users_per_page = 4
    num_pages = math.ceil(len(users) / users_per_page)
    suffix = f"_{remark}" if remark else ""
    pdf_path = os.path.join(output_dir, f"varentropy_analysis_report{suffix}.pdf")
    
    with PdfPages(pdf_path) as pdf:
        for page in range(num_pages):
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes_flat = axes.flatten()
            page_users = users[page*users_per_page : (page+1)*users_per_page]
            for i, user in enumerate(page_users):
                ax = axes_flat[i]
                user_data = df[df['User'] == user]
                is_success = user_data['Success'].iloc[0] if 'Success' in user_data.columns else False
                for type_name, style in TYPE_STYLES.items():
                    subset = user_data[user_data['Type'] == type_name]
                    if not subset.empty:
                        ax.plot(subset['Turn'], subset['Variance'], marker=style['marker'], linestyle=style['linestyle'],
                               color=style['color'], label=style['label'], linewidth=2.5, markersize=8, alpha=0.8)
                
                if is_success:
                    ax.text(1.05, 1.05, '★', transform=ax.transAxes, fontsize=25, color='gold', ha='right', va='top', fontweight='bold')
                
                ax.set_title(f"User: {user}", fontsize=14, fontweight='bold', pad=10)
                ax.set_xlabel("Turn"); ax.set_ylabel("Entropy Variance (Var(H))")
                ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.6)
                turns = user_data['Turn'].unique()
                if len(turns) > 0: ax.set_xticks(range(1, int(max(turns)) + 1))
            for j in range(len(page_users), 4): axes_flat[j].axis('off')
            plt.suptitle(f"Entropy Variance Analysis - Page {page+1}", fontsize=18, y=0.98, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig); plt.close()
    print(f"✅ PDF 报告已生成: {pdf_path}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--remark", default="", help="备注后缀")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 1. 加载数据
    data = load_data(args.input)
    if not data: return

    # 2. 提取数据 (合并 Strategy + Response 为 Action)
    df_var = collect_varentropy_records(data)

    # 3. 绘图 (2x2 规格 PDF)
    plot_varentropy_to_pdf(df_var, args.output_dir, args.remark)

    # 4. 保存 CSV 存档
    csv_path = os.path.join(args.output_dir, f"varentropy_stats_{args.remark}.csv")
    df_var.to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()