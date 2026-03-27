import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re
import math
from matplotlib.backends.backend_pdf import PdfPages

# ==============================================================================
# 1. 环境与样式配置
# ==============================================================================
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局色彩与标记定义
COLORS = {'State': '#1f77b4', 'Action': '#ff7f0e'}  # 蓝色-状态, 橙色-动作
MARKERS = {'State': 'o', 'Action': 's'}
LINESTYLES = {'State': '-', 'Action': '--'}

# ==============================================================================
# 2. 数据处理核心工具
# ==============================================================================
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

def segment_metrics(metrics):
    """严格按照标签切分状态和动作阶段"""
    full_text = "".join([m['token'] for m in metrics])
    
    def get_range(start_tag, end_tag):
        s_match = re.search(re.escape(start_tag), full_text, re.I)
        e_match = re.search(re.escape(end_tag), full_text, re.I)
        if not s_match or not e_match: return []
        start_pos, end_pos = s_match.end(), e_match.start()
        
        res = []
        curr_len = 0
        for m in metrics:
            m_len = len(m['token'])
            if curr_len + m_len > start_pos and curr_len < end_pos:
                res.append(m)
            curr_len += m_len
        return res

    state_m = get_range("<state_analysis>", "</state_analysis>")
    strategy_m = get_range("<strategy>", "</strategy>")
    response_m = get_range("<response>", "</response>")
    # return state_m, strategy_m + response_m  # Action = Strategy + Response
    return state_m, strategy_m # UPDATE: 删除 Response

def load_all_data(exp_path, persona_path):
    """合并实验数据与人格数据"""
    with open(exp_path, 'r') as f: exp_data = json.load(f)

     # --- 添加下面这两行：确保数据是列表格式 ---
    if isinstance(exp_data, dict):
        exp_data = list(exp_data.values())
        
    with open(persona_path, 'r') as f: persona_map = json.load(f)
    
    
    records = []
    for session in exp_data:
        uid = session['user_id']
        # 假设 uid 格式如 "persona_1", 直接匹配。如果不匹配，需做正则提取。
        p_info = persona_map.get(uid, {})


        big_five = str(p_info.get("Big-Five Personality", "Unknown")).strip().title()
        style = str(p_info.get("Decision-Making Style", "Unknown")).strip().title()

        success = session.get('success', False)
        
        history = session.get('detailed_history', [])
        turn_count = 0
        for turn in history:
            if turn['role'] == 'Persuader':
                turn_count += 1
                metrics = turn.get('metrics')
                if not metrics: continue
                
                state_m, action_m = segment_metrics(metrics)
                
                # 计算指标
                if state_m:
                    h_vals = [m['entropy'] for m in state_m]
                    records.append({
                        "User": uid, "BigFive": big_five, "Style": style, "Success": success,
                        "Turn": turn_count, "Type": "State", 
                        "Mean_H": np.mean(h_vals), "Var_H": np.var(h_vals)
                    })
                if action_m:
                    h_vals = [m['entropy'] for m in action_m]
                    records.append({
                        "User": uid, "BigFive": big_five, "Style": style, "Success": success,
                        "Turn": turn_count, "Type": "Action", 
                        "Mean_H": np.mean(h_vals), "Var_H": np.var(h_vals)
                    })
    return pd.DataFrame(records)

# ==============================================================================
# 3. 统计汇总与全局图表
# ==============================================================================

def generate_personality_summary(df, output_dir, remark):
    """生成人格/风格统计 JSON 和 矩阵图"""
    summary = {}
    # 计算每个组合的成功率和熵值均值
    # 注意：Success 是按 User 算的，不是按 Turn 算的
    user_level = df.groupby(['BigFive', 'Style', 'User', 'Success']).size().reset_index()
    
    # 1. 构建 JSON
    for (bf, st), group in df.groupby(['BigFive', 'Style']):
        if bf not in summary: summary[bf] = {}
        u_group = user_level[(user_level['BigFive']==bf) & (user_level['Style']==st)]
        
        success_num = int(u_group['Success'].sum())
        total_num = len(u_group)
        
        summary[bf][st] = {
            "success_count": success_num,
            "fail_count": total_num - success_num,
            "success_rate": success_num / total_num if total_num > 0 else 0,
            "avg_state_h": group[group['Type']=='State']['Mean_H'].mean(),
            "avg_action_h": group[group['Type']=='Action']['Mean_H'].mean(),
            "avg_state_var": group[group['Type']=='State']['Var_H'].mean(),
            "avg_action_var": group[group['Type']=='Action']['Var_H'].mean()
        }
    
    with open(os.path.join(output_dir, f"personality_style_summary_{remark}.json"), 'w') as f:
        json.dump(summary, f, indent=4)

    # 2. 矩阵图可视化
    # 准备矩阵绘图数据
    metrics_to_plot = ['success_rate', 'avg_state_h', 'avg_action_h', 'avg_action_var']
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    for i, m in enumerate(metrics_to_plot):
        plot_data = []
        for bf, styles in summary.items():
            for st, vals in styles.items():
                plot_data.append({"BigFive": bf, "Style": st, "Value": vals[m]})
        
        m_df = pd.DataFrame(plot_data).pivot(index="BigFive", columns="Style", values="Value")
        sns.heatmap(m_df, annot=True, cmap="YlGnBu", ax=axes[i], fmt=".3f")
        axes[i].set_title(f"Metric: {m.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
        
    plt.suptitle(f"Personality & Style Performance Matrix ({remark})", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, f"personality_matrix_{remark}.png"), dpi=300)
    plt.close()

def plot_global_trends(df, output_dir, remark):
    """生成成功 vs 失败全局趋势 PNG"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    for status, ls in zip([True, False], ['-', '--']):
        label_prefix = "Success" if status else "Fail"
        subset = df[df['Success'] == status]
        
        for t, color in zip(['State', 'Action'], [COLORS['State'], COLORS['Action']]):
            data = subset[subset['Type'] == t].groupby('Turn')[['Mean_H', 'Var_H']].mean().reset_index()
            ax1.plot(data['Turn'], data['Mean_H'], marker='o', label=f"{label_prefix} {t}", color=color, linestyle=ls)
            ax2.plot(data['Turn'], data['Var_H'], marker='s', label=f"{label_prefix} {t}", color=color, linestyle=ls)
            
    ax1.set_title("Global Mean Entropy Trend", fontweight='bold')
    ax2.set_title("Global VarEntropy Trend", fontweight='bold')
    ax1.set_ylabel("Mean Entropy (H)"); ax2.set_ylabel("Variance of Entropy (Var H)")
    for ax in [ax1, ax2]:
        ax.legend(); ax.set_xlabel("Turn")
    
    plt.savefig(os.path.join(output_dir, f"global_trends_{remark}.png"), dpi=300)
    plt.close()

# ==============================================================================
# 4. 个人报告 PDF 生成 (左右双列图)
# ==============================================================================

def plot_individual_pdfs(df, output_dir, remark):
    """为每个人格生成多页 PDF"""
    big_fives = df['BigFive'].unique()
    
    for bf in big_fives:
        if bf == "Unknown": continue
        bf_df = df[df['BigFive'] == bf].sort_values(by=['Style', 'User', 'Turn'])
        users = bf_df['User'].unique()
        
        pdf_path = os.path.join(output_dir, f"{bf}_Analysis_Report_{remark}.pdf")
        with PdfPages(pdf_path) as pdf:
            # 每页画 3 个用户（行）
            users_per_page = 3
            for i in range(0, len(users), users_per_page):
                page_users = users[i : i + users_per_page]
                fig, axes = plt.subplots(len(page_users), 2, figsize=(16, 5 * len(page_users)), squeeze=False)
                
                for u_idx, user in enumerate(page_users):
                    u_data = bf_df[bf_df['User'] == user]
                    success = u_data['Success'].iloc[0]
                    style = u_data['Style'].iloc[0]
                    
                    # 左图：Entropy
                    ax_l = axes[u_idx, 0]
                    # 右图：Variance
                    ax_r = axes[u_idx, 1]
                    
                    for t in ['State', 'Action']:
                        t_data = u_data[u_data['Type'] == t]
                        ax_l.plot(t_data['Turn'], t_data['Mean_H'], color=COLORS[t], 
                                  marker=MARKERS[t], linestyle=LINESTYLES[t], label=f"{t} H")
                        ax_r.plot(t_data['Turn'], t_data['Var_H'], color=COLORS[t], 
                                  marker=MARKERS[t], linestyle=LINESTYLES[t], label=f"{t} Var")
                    
                    # 装饰
                    star = " ★" if success else ""
                    ax_l.set_title(f"User: {user} | Style: {style}{star}", loc='left', fontweight='bold', color='darkred' if success else 'black')
                    ax_l.set_ylabel("Mean Entropy")
                    ax_r.set_ylabel("VarEntropy")
                    
                    for ax in [ax_l, ax_r]:
                        ax.legend(fontsize=8)
                        ax.set_xlabel("Turn")
                        if not t_data.empty:
                            ax.set_xticks(range(1, int(t_data['Turn'].max()) + 1))

                plt.suptitle(f"Personality Group: {bf} | Details Page {i//users_per_page + 1}", fontsize=16, fontweight='bold', y=0.98)
                plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                pdf.savefig(fig)
                plt.close()
    print(f"✅ Individual PDF reports generated in {output_dir}")


def plot_success_rate_dashboard(df, output_dir, remark):
    """
    修正版：确保 N 值和成功率百分比准确对应到排序后的柱状图位置
    """
    # 提取用户级数据
    u_df = df[['User', 'BigFive', 'Style', 'Success']].drop_duplicates()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    for ax, col, title in zip([ax1, ax2], ['BigFive', 'Style'], ["Personality (Big-Five)", "Decision-Making Style"]):
        # 计算统计数据
        stats = u_df.groupby(col)['Success'].agg(['count', 'sum']).reset_index()
        stats.columns = [col, 'Total', 'Success_Count']
        stats['Success_Rate'] = stats['Success_Count'] / stats['Total']
        
        # 按照成功率降序排列
        stats = stats.sort_values('Success_Rate', ascending=False).reset_index(drop=True)

        # 1. 绘制柱状图 (x轴使用 category 字符串，sns 会自动按顺序排列)
        sns.barplot(x=col, y='Total', data=stats, color='#bdc3c7', label='Total Users', ax=ax)
        sns.barplot(x=col, y='Success_Count', data=stats, color='#2ecc71', label='Success Users', ax=ax)
        
        # 2. 绘制成功率折线图 (twinx)
        ax_twin = ax.twinx()
        # 注意：这里需要明确指定 x 为数字序列 [0, 1, 2...] 确保与柱子中心对齐
        ax_twin.plot(range(len(stats)), stats['Success_Rate'], color='#e74c3c', 
                     marker='D', linewidth=3, markersize=8, label='Success Rate')
        ax_twin.set_ylim(0, 1.1)
        ax_twin.set_ylabel("Success Rate (%)", fontsize=12, fontweight='bold')
        ax_twin.grid(False) # 避免双重网格线

        # 3. 添加数值标注 (关键修正：使用 enumerate 获取当前在图中的物理位置 i)
        for i, row in stats.iterrows():
            # i 是 reset_index 后的 0, 1, 2... 对应从左到右的柱子位置
            
            # 在柱子顶部标注总人数 N
            ax.text(i, row['Total'] + 0.2, f"N={row['Total']}", 
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # 在折线点上方标注成功率百分比
            ax_twin.text(i, row['Success_Rate'] + 0.03, f"{row['Success_Rate']:.1%}", 
                         ha='center', va='bottom', color='#c0392b', 
                         fontweight='bold', fontsize=12)

        ax.set_title(f"Success Performance Dashboard by {title}", fontsize=16, fontweight='bold', pad=25)
        ax.set_xlabel(title, fontsize=12, labelpad=10)
        ax.set_ylabel("User Count", fontsize=12)
        
        # 合并图例
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc='upper right', frameon=True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35) # 调整上下图间距
    save_path = os.path.join(output_dir, f"success_dashboard_{remark}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 已修正并生成看板图: {save_path}")

def plot_categorical_entropy_trends(df, output_dir, remark):
    """
    重构版：生成 4x2 布局的 8 张子图
    每张子图独立包含左下角图例，美观且不干扰数据。
    """
    for cat_col, title in zip(['BigFive', 'Style'], ["Big-Five Personality", "Decision-Making Style"]):
        # 创建 4行 2列 的画布，增加高度以确保各子图清晰
        fig, axes = plt.subplots(4, 2, figsize=(18, 30))
        
        # 配置绘图矩阵的逻辑映射
        # row_configs: (行, 列, 数据类型, 是否显示阴影, Y轴字段, 标题后缀)
        configs = [
            # 第一部分：仅线段 (Rows 0-1)
            (0, 0, 'State',  None, 'Mean_H', "State Entropy (H) - Trend Only"),
            (0, 1, 'State',  None, 'Var_H',  "State Var-Entropy (Var H) - Trend Only"),
            (1, 0, 'Action', None, 'Mean_H', "Action Entropy (H) - Trend Only"),
            (1, 1, 'Action', None, 'Var_H',  "Action Var-Entropy (Var H) - Trend Only"),
            # 第二部分：线段 + 阴影 (Rows 2-3)
            (2, 0, 'State',  'sd', 'Mean_H', "State Entropy (H) - Mean & StdDev"),
            (2, 1, 'State',  'sd', 'Var_H',  "State Var-Entropy (Var H) - Mean & StdDev"),
            (3, 0, 'Action', 'sd', 'Mean_H', "Action Entropy (H) - Mean & StdDev"),
            (3, 1, 'Action', 'sd', 'Var_H',  "Action Var-Entropy (Var H) - Mean & StdDev")
        ]

        for r, c, t_type, err, y_col, sub_title in configs:
            ax = axes[r, c]
            # 过滤数据
            plot_data = df[df['Type'] == t_type]
            
            # 绘图
            sns.lineplot(
                data=plot_data, 
                x='Turn', 
                y=y_col, 
                hue=cat_col, 
                errorbar=err, 
                marker='o' if err is None else None, 
                palette='viridis' if 'Mean' in y_col else 'flare',
                linewidth=2.5,
                ax=ax,
                legend=True # 开启子图图例
            )

            # --- 子图细节美化 ---
            ax.set_title(sub_title, fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
            ax.set_xticks(range(1, int(df['Turn'].max()) + 1))
            ax.set_xlabel("Turn (Round)", fontsize=10)
            ax.set_ylabel("Metric Value", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.3)

            # --- 设置每张子图的独立图例 ---
            # 使用 loc='lower left'，设置较小字号和半透明背景
            ax.legend(
                loc='lower left', 
                fontsize='x-small', 
                title=f"{cat_col}", 
                title_fontsize='small',
                frameon=True, 
                framealpha=0.6, # 背景半透明
                edgecolor='gray'
            )

        # 调整总标题
        plt.suptitle(f"Comprehensive Categorical Dynamics: {title}\n(Rows 1-2: Clear Trends | Rows 3-4: Variance & Stability)", 
                     fontsize=24, y=0.98, fontweight='bold', color='#1a252f')

        # 调整布局，确保子图标题和坐标轴不重叠
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        # 保存图片
        save_path = os.path.join(output_dir, f"categorical_detailed_trends_{cat_col}_{remark}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已生成带有独立图例的面板图: {save_path}")

def plot_outcome_path_comparison(df, output_dir, remark):
    """
    可视化 3: 深度轨迹分歧分析——对比成功与失败会话在不同人格下的心理状态与动作表现
    """
    # 过滤掉 Unknown 人格
    cats = [c for c in df['BigFive'].unique() if c != "Unknown"]
    
    if not cats:
        print("⚠️ No valid personality categories found for comparison.")
        return

    # 动态调整高度
    fig, axes = plt.subplots(len(cats), 2, figsize=(20, 6 * len(cats)), squeeze=False)
    
    # --- 修正点：定义标签映射和与之完全匹配的颜色字典 ---
    label_success = "Persuaded (Success)"
    label_fail = "Non-Persuaded (Fail)"
    
    label_map = {True: label_success, False: label_fail}
    # 颜色字典的键必须是映射后的字符串
    color_palette = {
        label_success: "#27ae60", 
        label_fail: "#e74c3c"
    }
    
    for i, cat in enumerate(cats):
        sub_df = df[df['BigFive'] == cat].copy()
        if sub_df.empty: continue
        
        # 应用映射
        sub_df['Outcome'] = sub_df['Success'].map(label_map)
        
        # --- 左列: 认知确定性分析 (Mean Entropy) ---
        ax_l = axes[i, 0]
        sns.lineplot(
            data=sub_df, x='Turn', y='Mean_H', hue='Outcome', style='Type',
            palette=color_palette, ax=ax_l, markers=True, markersize=8, linewidth=2.5,
            dashes={'State': (None, None), 'Action': (2, 2)}
        )
        ax_l.set_title(f"[{cat}] Confidence Profile: Mean Entropy Trajectory", 
                       fontsize=14, fontweight='bold', pad=15)
        ax_l.set_ylabel("Average Token Entropy (Mean H)")

        # --- 右列: 信息稳定性分析 (Var-Entropy) ---
        ax_r = axes[i, 1]
        sns.lineplot(
            data=sub_df, x='Turn', y='Var_H', hue='Outcome', style='Type',
            palette=color_palette, ax=ax_r, markers=True, markersize=8, linewidth=2.5,
            dashes={'State': (None, None), 'Action': (2, 2)}
        )
        ax_r.set_title(f"[{cat}] Stability Profile: Var-Entropy Trajectory", 
                       fontsize=14, fontweight='bold', pad=15)
        ax_r.set_ylabel("Information Volatility (Var H)")
        
        # 细节统一美化
        for ax in [ax_l, ax_r]:
            ax.set_xlabel("Conversation Turn", fontsize=11)
            # 确保 x 轴刻度为整数
            max_turn = int(sub_df['Turn'].max()) if not sub_df.empty else 1
            ax.set_xticks(range(1, max_turn + 1))
            ax.grid(True, linestyle='--', alpha=0.4)
            
            # 固定图例于左下角
            ax.legend(
                loc='lower left', 
                fontsize='x-small', 
                frameon=True, 
                framealpha=0.6,
                title="Outcome & Stage",
                title_fontsize='small'
            )
            sns.despine(ax=ax)

    plt.suptitle(f"Persuasion Trajectory Divergence Analysis: Success vs. Failure Patterns\n({remark})", 
                 fontsize=22, y=0.99, fontweight='bold', color='#2c3e50')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    save_path = os.path.join(output_dir, f"outcome_trajectory_divergence_{remark}.png")
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"✅ 已修正并生成轨迹分歧分析图: {save_path}")

# ==============================================================================
# 7. 更新后的主程序入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Persona Entropy Analysis System")
    parser.add_argument("--input", required=True)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--remark", default="run_v1")
    args = parser.parse_args()

    # 1. 目录准备
    global_dir = os.path.join(args.output_dir, "global_trends")
    pers_dir = os.path.join(args.output_dir, "personality_reports")
    stats_dir = os.path.join(args.output_dir, "statistical_summary") # 新增目录
    for d in [global_dir, pers_dir, stats_dir]:
        os.makedirs(d, exist_ok=True)

    print("Step 1: Loading Data...")
    full_df = load_all_data(args.input, args.persona)

    print("Step 2: Plotting Global Trends...")
    plot_global_trends(full_df, global_dir, args.remark) 

    print("Step 3: Generating Statistical Dashboards...")
    # --- 调用新增函数 ---
    plot_success_rate_dashboard(full_df, stats_dir, args.remark)
    plot_categorical_entropy_trends(full_df, stats_dir, args.remark)
    plot_outcome_path_comparison(full_df, stats_dir, args.remark)
    
    print("Step 4: Generating Personality Matrix & Reports...")
    generate_personality_summary(full_df, args.output_dir, args.remark)
    plot_individual_pdfs(full_df, pers_dir, args.remark)

    print(f"\n✨ Full analysis for {args.remark} completed successfully.")
    print(f"📁 Statistical visuals are saved in: {stats_dir}")

if __name__ == "__main__":
    main()