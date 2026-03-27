import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import warnings
import textwrap
import warnings
import re # 引入正则表达式用于清洗 token
warnings.filterwarnings("ignore")

# =====================================================================
# 模块 1：基础配置与策略映射字典
# =====================================================================

SYS_STRATEGY_MAP = {
    "Greeting": "Exploration", "Source Related Inquiry": "Exploration",
    "Task Related Inquiry": "Exploration", "Personal Related Inquiry": "Exploration",
    "Logical Appeal": "Core Persuasion", "Emotion Appeal": "Core Persuasion",
    "Credibility Appeal": "Core Persuasion", "Personal Story": "Core Persuasion",
    "Foot in the Door": "Action Facilitation", "Self Modeling": "Action Facilitation",
    "Donation Information": "Action Facilitation"
}

def is_traj_success(trajectory):
    if not trajectory: return False
    if trajectory.get('success'): return True
    turns = trajectory.get('turns', [])
    if not turns: return False
    for turn in turns:
        if turn.get('reward', 0.0) >= 1.0: return True
    if turns[-1].get('reward', 0.0) >= 1.0: return True
    return False

def wrap_labels(text, width=22):
    parts = str(text).split(' -> ')
    if len(parts) == 3: return f"{parts[0]}\n↓\n{parts[1]}\n↓\n{parts[2]}"
    return textwrap.fill(str(text), width)

# =====================================================================
# 模块 2：Token 解析与数据底座引擎
# =====================================================================
def segment_metrics_three_parts(metrics):
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

def analysis_token_entropy_wordcloud(metrics_filepath, output_dir, prefix):
    """
    【核心方法】渲染三区独立的 Token 熵值云图。
    WordCloud 大小映射为：该 Token 的全局平均熵值 (Mean Entropy)。
    """
    print(f"\n  -> Running Token Entropy WordCloud Analysis for {prefix}...")
    
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("     [Error] Please install wordcloud first: pip install wordcloud")
        return

    if not os.path.exists(metrics_filepath):
        print(f"     [Warning] Metrics file not found: {metrics_filepath}. Skipping WordCloud.")
        return

    with open(metrics_filepath, 'r', encoding='utf-8') as f:
        metrics_data = json.load(f)

    # 数据桶：收集每个 token 的历史熵值
    token_stats = {
        'State Analysis': defaultdict(list),
        'Strategy': defaultdict(list),
        'Response': defaultdict(list)
    }

    def clean_token(t_str):
        # 清洗 Tokenizer 自带的特殊符号（如 RoBERTa/Qwen 常用的 'Ġ' 或 '_'）
        t = t_str.replace('Ġ', '').replace(' ', '').replace(' ', '').strip()
        # 移除非字母数字的纯符号噪音，防止云图被标点符号填满
        t = re.sub(r'[^\w\s]', '', t)
        return t

    for user in metrics_data:
        for traj in user.get('trajectories', []):
            for turn in traj.get('metrics', []):
                m_list = turn.get('metrics', [])
                p1, p2, p3 = segment_metrics_three_parts(m_list)

                for tok in p1: 
                    ct = clean_token(tok['token'])
                    if ct: token_stats['State Analysis'][ct].append(tok.get('entropy', 0.0))
                for tok in p2: 
                    ct = clean_token(tok['token'])
                    if ct: token_stats['Strategy'][ct].append(tok.get('entropy', 0.0))
                for tok in p3: 
                    ct = clean_token(tok['token'])
                    if ct: token_stats['Response'][ct].append(tok.get('entropy', 0.0))

    # 准备绘图 1x3 画布
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    sections = ['State Analysis', 'Strategy', 'Response']
    colormaps = ['Blues', 'Greens', 'Oranges'] # 用不同颜色区分三个思维阶段

    for i, section in enumerate(sections):
        weights = {}
        for token, entropies in token_stats[section].items():
            # 【关键抗噪设计】：仅保留全局出现次数 >= 5 次的 token，防止孤立乱码支配云图
            if len(entropies) >= 5:
                # 权重 = 该 token 的平均熵值
                weights[token] = np.mean(entropies)
        
        if not weights:
            axes[i].set_title(f"{section}\n(Not enough data)", fontsize=18)
            axes[i].axis('off')
            continue
            
        # 生成词云
        wc = WordCloud(
            width=800, height=800, 
            background_color='white', 
            colormap=colormaps[i],
            max_words=150,
            contour_width=0
        ).generate_from_frequencies(weights)
        
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].set_title(f"{section}\n(Size = Mean Entropy)", fontsize=20, fontweight='bold', pad=15)
        axes[i].axis('off')

    plt.suptitle(f"Token Entropy Landscape across Generation Phases - {prefix}\n(Highlighting High-Uncertainty Forking Tokens)", fontsize=26, y=1.05, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, f"{prefix}_Entropy_WordCloud.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     ✅ Saved Entropy WordCloud to: {out_path}")

def segment_metrics_new_format(metrics):
    tokens_text = "".join([m['token'] for m in metrics])
    s1 = tokens_text.find("<state_analysis>")
    e1 = tokens_text.find("</state_analysis>")
    s2 = tokens_text.find("<strategy>")
    e2 = tokens_text.find("</strategy>")
    
    if s1 != -1 and e1 == -1:
        e1 = s2 if (s2 != -1 and s2 > s1) else len(tokens_text)
    if s2 != -1 and e2 == -1:
        s3 = tokens_text.find("<response>")
        e2 = s3 if (s3 != -1 and s3 > s2) else len(tokens_text)
            
    valid_tokens = []
    current_pos = 0
    for m in metrics:
        token_len = len(m['token'])
        token_start = current_pos
        token_end = current_pos + token_len
        in_state = (s1 != -1 and e1 != -1 and token_end > s1 + len("<state_analysis>") and token_start < e1)
        in_strat = (s2 != -1 and e2 != -1 and token_end > s2 + len("<strategy>") and token_start < e2)
        if in_state or in_strat:
            valid_tokens.append(m)
        current_pos += token_len
    return valid_tokens

def extract_trajectory_turns(trajectory, user_metrics=None):
    traj_id = trajectory.get('id', 'Unknown')
    turns_dict = {}
    current_round = 1  
    
    for turn in trajectory.get('turns', []):
        r = turn.get('round')
        if r is not None: current_round = r
            
        role = turn.get('role')
        if role == 'Persuader':
            sys_action = turn.get('strategy_name', "Unknown")
            hs, ha = turn.get('hs'), turn.get('ha')
            
            h_coupled = None
            if user_metrics and traj_id in user_metrics:
                m_traj = user_metrics[traj_id]
                m_round = next((m for m in m_traj.get('metrics', []) if m.get('round') == current_round), None)
                if m_round and 'metrics' in m_round:
                    valid_tokens = segment_metrics_new_format(m_round['metrics'])
                    ents = [tok.get('entropy', 0.0) for tok in valid_tokens]
                    if ents: h_coupled = float(np.mean(ents))

            if current_round not in turns_dict: turns_dict[current_round] = {}
            turns_dict[current_round]['Sys_Spec'] = sys_action
            turns_dict[current_round]['Hc'] = h_coupled
            if hs is not None and ha is not None:
                turns_dict[current_round]['Hs'] = float(hs)
                turns_dict[current_round]['Ha'] = float(ha)
            
        elif role == 'Persuadee':
            user_action = turn.get('user_strategy', "None")
            if current_round not in turns_dict: turns_dict[current_round] = {}
            turns_dict[current_round]['User_Spec'] = user_action
            
    sorted_turns = []
    for r in sorted(turns_dict.keys()):
        sorted_turns.append({
            'Turn': r, 
            'Sys_Spec': turns_dict[r].get('Sys_Spec', 'Other'), 
            'User_Spec': turns_dict[r].get('User_Spec', 'Other'), 
            'Hs': turns_dict[r].get('Hs'),
            'Ha': turns_dict[r].get('Ha'),
            'Hc': turns_dict[r].get('Hc')
        })
    return sorted_turns

def process_sequences(user_item, exp_name, metrics_lookup):
    records = []
    user_id = user_item.get('user_id', 'Unknown') 
    trajectories = user_item.get('trajectories', [])
    root_traj = next((t for t in trajectories if t.get('id') == 'root'), None)
    if not root_traj: return []
    
    user_metrics = metrics_lookup.get(user_id, {})
    root_seq = extract_trajectory_turns(root_traj, user_metrics)
    
    def yield_records(seq, outcome, start_turn, hist_sys, p_u, p_hs, p_ha):
        _hist_sys = list(hist_sys)
        _p_u = p_u
        _p_hs, _p_ha = p_hs, p_ha
        
        for turn_data in seq:
            t = turn_data['Turn']
            c_s, c_u = turn_data['Sys_Spec'], turn_data['User_Spec']
            c_hs, c_ha, c_hc = turn_data.get('Hs'), turn_data.get('Ha'), turn_data.get('Hc')
            
            if t >= start_turn and c_s != "Other":
                int_triplet = f"{_hist_sys[-1]} -> {_p_u} -> {c_s}" if len(_hist_sys) >= 1 and _p_u != "START" else None
                sys_triplet = f"{_hist_sys[-2]} -> {_hist_sys[-1]} -> {c_s}" if len(_hist_sys) >= 2 else None
                
                record = {
                    'Experiment': exp_name, 'Outcome': outcome, 'Turn': t,
                    'Interaction_Triplet': int_triplet,
                    'System_Triplet': sys_triplet,
                    'H_state': c_hs, 'H_action': c_ha, 'H_coupled': c_hc,
                    'Delta_H_state': c_hs - _p_hs if c_hs is not None and _p_hs is not None else None, 
                    'Delta_H_action': c_ha - _p_ha if c_ha is not None and _p_ha is not None else None
                }
                records.append(record)
            
            _hist_sys.append(c_s)
            _p_u = c_u
            _p_hs, _p_ha = c_hs, c_ha

    yield_records(root_seq, 'Success' if is_traj_success(root_traj) else 'Fail', start_turn=1, hist_sys=[], p_u="START", p_hs=None, p_ha=None)
    
    for b in [t for t in trajectories if t.get('id') != 'root']:
        t_branch = b.get('branch_at_turn')
        if t_branch is not None:
            branch_seq = extract_trajectory_turns(b, user_metrics)
            hist_sys = [turn['Sys_Spec'] for turn in root_seq if turn['Turn'] < t_branch]
            context_turn = next((turn for turn in root_seq if turn['Turn'] == t_branch - 1), None)
            if context_turn:
                i_user = context_turn['User_Spec']
                i_hs, i_ha = context_turn['Hs'], context_turn['Ha']
            else:
                i_user = "START"
                i_hs = i_ha = None
                
            yield_records(branch_seq, 'Success' if is_traj_success(b) else 'Fail', start_turn=t_branch,
                          hist_sys=hist_sys, p_u=i_user, p_hs=i_hs, p_ha=i_ha)
    return records

def build_datasets(experiments_config):
    all_data = []
    for config in experiments_config:
        filepath, exp_name = config.get('path'), config.get('name')
        metrics_filepath = filepath.replace('.json', '_metrics.json')
        
        if not os.path.exists(filepath): continue
        with open(filepath, 'r', encoding='utf-8') as f: struct_data = json.load(f)
            
        metrics_lookup = {}
        if os.path.exists(metrics_filepath):
            with open(metrics_filepath, 'r', encoding='utf-8') as f: metrics_data = json.load(f)
            for m_user in metrics_data:
                metrics_lookup[m_user.get('user_id')] = {t['id']: t for t in m_user.get('trajectories', [])}

        for user_item in struct_data:
            all_data.extend(process_sequences(user_item, exp_name, metrics_lookup))
            
    return pd.DataFrame(all_data)

# =====================================================================
# 模块 3：核心验证画图 (加入 Top-K 参数)
# =====================================================================

def validation_decoupling_and_trend(df, output_dir, prefix, top_k=50):
    print(f"\n--- Running Validation: Decoupling and Trend Analysis ({prefix}) ---")
    
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  -> [Error] Please install plotly first: pip install plotly")
        return

    valid_df = df.dropna(subset=['H_state', 'H_action', 'H_coupled', 'Delta_H_state', 'Delta_H_action']).copy()
    if valid_df.empty: return

    df_int = valid_df.dropna(subset=['Interaction_Triplet']).copy()
    df_int['Pattern_Type'] = 'Interaction (Sys->User->Sys)'
    df_int['Pattern'] = df_int['Interaction_Triplet']

    df_sys = valid_df.dropna(subset=['System_Triplet']).copy()
    df_sys['Pattern_Type'] = 'System (Sys->Sys->Sys)'
    df_sys['Pattern'] = df_sys['System_Triplet']

    combined_df = pd.concat([df_int, df_sys])

    # 生成基础统计文件
    stats_outcome = combined_df.groupby(['Pattern_Type', 'Pattern', 'Outcome']).agg(
        Count=('Turn', 'count'),
        Mean_Hs=('H_state', 'mean'), Mean_Ha=('H_action', 'mean'), Mean_Hc=('H_coupled', 'mean'),
        Mean_dHs=('Delta_H_state', 'mean'), Mean_dHa=('Delta_H_action', 'mean')
    ).reset_index()
    stats_outcome.to_csv(os.path.join(output_dir, f"{prefix}_Patterns_Outcome_Stats.csv"), index=False)

    stats_total = combined_df.groupby(['Pattern_Type', 'Pattern']).agg(
        Total_Count=('Turn', 'count'),
        Success_Count=('Outcome', lambda x: (x == 'Success').sum()),
        Fail_Count=('Outcome', lambda x: (x == 'Fail').sum())
    ).reset_index()
    stats_total = stats_total.sort_values(by='Total_Count', ascending=False)
    stats_total.to_csv(os.path.join(output_dir, f"{prefix}_Patterns_Merged_Frequency.csv"), index=False)

    # 提取分类型的 Top-10 (用于 Mode 1/2/3)
    top10_int = stats_total[stats_total['Pattern_Type'] == 'Interaction (Sys->User->Sys)'].head(10)['Pattern'].tolist()
    top10_sys = stats_total[stats_total['Pattern_Type'] == 'System (Sys->Sys->Sys)'].head(10)['Pattern'].tolist()

    df_int_top10 = df_int[df_int['Pattern'].isin(top10_int)].copy()
    df_sys_top10 = df_sys[df_sys['Pattern'].isin(top10_sys)].copy()

    import textwrap
    df_int_top10['Short_Pattern'] = df_int_top10['Pattern'].apply(lambda x: textwrap.fill(x, 35))
    df_sys_top10['Short_Pattern'] = df_sys_top10['Pattern'].apply(lambda x: textwrap.fill(x, 35))

    # ==========================================================
    # 💡 模式 1：全新的 1D 耦合熵箱线对比图 (Boxplot)
    # ==========================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(20, 10))
    colors_hex = {'Success': '#2ecc71', 'Fail': '#e74c3c'}

    sns.boxplot(data=df_int_top10, x='H_coupled', y='Short_Pattern', hue='Outcome', palette=colors_hex, ax=axes1[0], width=0.6, fliersize=3)
    axes1[0].set_title("Interaction Patterns (Top 10)", fontsize=16, pad=10)
    axes1[0].set_xlabel("Coupled Entropy ($H_c$)", fontsize=14)
    axes1[0].set_ylabel("")

    sns.boxplot(data=df_sys_top10, x='H_coupled', y='Short_Pattern', hue='Outcome', palette=colors_hex, ax=axes1[1], width=0.6, fliersize=3)
    axes1[1].set_title("System Patterns (Top 10)", fontsize=16, pad=10)
    axes1[1].set_xlabel("Coupled Entropy ($H_c$)", fontsize=14)
    axes1[1].set_ylabel("")

    plt.suptitle(f"Mode 1: Can Coupled Entropy ($H_c$) distinguish Success vs Fail?\n(The 'Averaging Trap' - Success and Fail distributions heavily overlap)", fontsize=20, y=1.05)
    plt.tight_layout()
    mode1_path = os.path.join(output_dir, f"{prefix}_Mode1_Coupled_Entropy_Comparison.png")
    plt.savefig(mode1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     ✅ Saved Mode 1 (Boxplot Comparison) to: {mode1_path}")

    # ==========================================================
    # 💡 模式 2 & 3 数据准备
    # ==========================================================
    centroid_top10 = stats_outcome[
        ((stats_outcome['Pattern_Type'] == 'Interaction (Sys->User->Sys)') & (stats_outcome['Pattern'].isin(top10_int))) |
        ((stats_outcome['Pattern_Type'] == 'System (Sys->Sys->Sys)') & (stats_outcome['Pattern'].isin(top10_sys)))
    ].copy()

    global_mean_hs = valid_df['H_state'].mean()
    global_mean_ha = valid_df['H_action'].mean()
    symbols = {'Success': 'circle', 'Fail': 'x'}

    def create_2d_plotly_figure(df_plot, x_col, y_col, title, x_title, y_title, is_trend=False):
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Interaction Patterns (Top 10)", "System Patterns (Top 10)"], horizontal_spacing=0.08)
        
        for col_idx, p_type in enumerate(['Interaction (Sys->User->Sys)', 'System (Sys->Sys->Sys)'], 1):
            for outcome in ['Success', 'Fail']:
                group_df = df_plot[(df_plot['Pattern_Type'] == p_type) & (df_plot['Outcome'] == outcome)]
                if group_df.empty: continue
                
                hover_text = group_df.apply(lambda row: 
                    f"<b>Outcome: {row['Outcome']}</b><br><b>模式:</b> {row['Pattern']}<br>"
                    f"Hs: {row['Mean_Hs']:.3f} | Ha: {row['Mean_Ha']:.3f}<br>"
                    f"ΔHs: {row['Mean_dHs']:.3f} | ΔHa: {row['Mean_dHa']:.3f}<br>"
                    f"频次: {row['Count']} 次", axis=1)

                fig.add_trace(go.Scatter(
                    x=group_df[x_col], y=group_df[y_col], mode='markers',
                    marker=dict(color=colors_hex[outcome], symbol=symbols[outcome], size=18, line=dict(color='black', width=1.5), opacity=0.85),
                    name=f"{outcome}", legendgroup=outcome, showlegend=(col_idx==1),
                    hoverinfo='text', hovertext=hover_text
                ), row=1, col=col_idx)

            if is_trend:
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6, row=1, col=col_idx)
                fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.6, row=1, col=col_idx)
            else:
                fig.add_hline(y=global_mean_ha, line_dash="dash", line_color="gray", opacity=0.6, row=1, col=col_idx)
                fig.add_vline(x=global_mean_hs, line_dash="dash", line_color="gray", opacity=0.6, row=1, col=col_idx)

        fig.update_layout(height=800, width=1600, title_text=title, title_font_size=22, title_x=0.5, plot_bgcolor='white', hovermode="closest")
        fig.update_xaxes(title_text=x_title, showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=False)
        fig.update_yaxes(title_text=y_title, showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=False)
        return fig

    fig2 = create_2d_plotly_figure(
        centroid_top10, 'Mean_Hs', 'Mean_Ha', 
        f"Mode 2: 2D Original Values (Hs vs Ha) - Top 10<br><sup>Quadrants based on Global Mean (Hs={global_mean_hs:.2f}, Ha={global_mean_ha:.2f})</sup>",
        "Original State Entropy (Hs)", "Original Action Entropy (Ha)", is_trend=False
    )
    fig2.write_html(os.path.join(output_dir, f"{prefix}_Mode2_Interactive.html"))

    fig3 = create_2d_plotly_figure(
        centroid_top10, 'Mean_dHs', 'Mean_dHa', 
        f"Mode 3: 2D Trend Quadrants (ΔHs vs ΔHa) - Top 10<br><sup>The Ultimate Disentanglement</sup>",
        "State Entropy Change (ΔHs)", "Action Entropy Change (ΔHa)", is_trend=True
    )
    fig3.write_html(os.path.join(output_dir, f"{prefix}_Mode3_Interactive.html"))

    # ==========================================================
    # 💡 模式 4：引入 Top-K 的上帝视角图
    # ==========================================================
    print(f"  -> Generating Mode 4: Top-{top_k} Patterns Global Map (Opacity mapped to Frequency)")
    fig4 = make_subplots(rows=1, cols=2, subplot_titles=[f"Interaction Patterns (Top {top_k})", f"System Patterns (Top {top_k})"], horizontal_spacing=0.08)
    
    # 依然使用全局最大的 Count 来标准化颜色的透明度，以保证对比度的一致性
    max_count = stats_outcome['Count'].max()

    for col_idx, p_type in enumerate(['Interaction (Sys->User->Sys)', 'System (Sys->Sys->Sys)'], 1):
        # 【关键修改】：针对每种模式类型，只提取频率排名前 K 的模式
        top_k_patterns_for_type = stats_total[stats_total['Pattern_Type'] == p_type].head(top_k)['Pattern'].tolist()
        
        for outcome in ['Success', 'Fail']:
            group_df = stats_outcome[(stats_outcome['Pattern_Type'] == p_type) & 
                                     (stats_outcome['Outcome'] == outcome) &
                                     (stats_outcome['Pattern'].isin(top_k_patterns_for_type))].copy()
            if group_df.empty: continue
            
            alphas = np.clip(np.log1p(group_df['Count']) / np.log1p(max_count), 0.15, 1.0)
            color_array = [f'rgba(46, 204, 113, {a})' if outcome == 'Success' else f'rgba(231, 76, 60, {a})' for a in alphas]
            sizes = 10 + (alphas * 15)

            hover_text = group_df.apply(lambda row: 
                f"<b>Outcome: {row['Outcome']}</b><br><b>模式:</b> {row['Pattern']}<br>"
                f"ΔHs: {row['Mean_dHs']:.3f} | ΔHa: {row['Mean_dHa']:.3f}<br>"
                f"频次: {row['Count']} 次", axis=1)

            fig4.add_trace(go.Scatter(
                x=group_df['Mean_dHs'], y=group_df['Mean_dHa'], mode='markers',
                marker=dict(color=color_array, symbol=symbols[outcome], size=sizes, line=dict(color='black', width=0.5)),
                name=f"{outcome}", legendgroup=outcome, showlegend=(col_idx==1),
                hoverinfo='text', hovertext=hover_text
            ), row=1, col=col_idx)

        fig4.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.8, line_width=2, row=1, col=col_idx)
        fig4.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.8, line_width=2, row=1, col=col_idx)

    fig4.update_layout(
        height=850, width=1600,
        title_text=f"Mode 4: Global Pattern Map (Top {top_k} DATA) - {prefix}<br><sup>Color Depth & Size mapped to Frequency (Count)</sup>",
        title_font_size=22, title_x=0.5, plot_bgcolor='white', hovermode="closest"
    )
    fig4.update_xaxes(title_text="State Entropy Change (ΔHs)", showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=False)
    fig4.update_yaxes(title_text="Action Entropy Change (ΔHa)", showgrid=True, gridwidth=1, gridcolor='LightGray', zeroline=False)

    mode4_path = os.path.join(output_dir, f"{prefix}_Mode4_Top{top_k}Patterns_Map.html")
    fig4.write_html(mode4_path)
    print(f"     ✅ Saved Mode 4 (Top-{top_k} Pattern Map HTML) to: {mode4_path}")

# =====================================================================
# 主程序
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', required=True, help="List of EXP result json files")
    parser.add_argument("--names", nargs='+', required=True, help="List of Model Names")
    parser.add_argument("--output_dir", default="eval_results", help="Directory to save analysis results")
    # 【新增参数】：控制 Mode 4 显示的策略数量
    parser.add_argument("--top_k", type=int, default=50, help="Number of top patterns to show in Mode 4 (default: 50)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    EXPERIMENTS_CONFIG = [{"name": n, "path": p} for p, n in zip(args.files, args.names)]
    
    print("\n[Stage 1] Building Dataset and calculating Entropies...")
    df_master = build_datasets(EXPERIMENTS_CONFIG)
    
    if df_master.empty:
        print("[Error] No valid data parsed.")
        return

    for config in EXPERIMENTS_CONFIG:
        filepath = config["path"]
        exp_name = config["name"]
        metrics_filepath = filepath.replace('.json', '_metrics.json')

        df_model = df_master[df_master["Experiment"] == exp_name]
        validation_decoupling_and_trend(df_model, args.output_dir, prefix=exp_name, top_k=args.top_k)
        analysis_token_entropy_wordcloud(metrics_filepath, args.output_dir, prefix=exp_name)
        
    print("\n✅ All validation visualizations generated successfully!")

if __name__ == "__main__":
    main()