import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import entropy, mannwhitneyu
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# 模块 1：基础配置与策略字典
# =====================================================================

SYS_STRATEGY_MAP = {
    "Greeting": "Exploration",
    "Source Related Inquiry": "Exploration",
    "Task Related Inquiry": "Exploration",
    "Personal Related Inquiry": "Exploration",
    "Logical Appeal": "Core Persuasion",
    "Emotion Appeal": "Core Persuasion",
    "Credibility Appeal": "Core Persuasion",
    "Personal Story": "Core Persuasion",
    "Foot in the Door": "Action Facilitation",
    "Self Modeling": "Action Facilitation",
    "Donation Information": "Action Facilitation"
}

USER_STRATEGY_MAP = {
    "Donate": "Compliance",
    "Information Inquiry": "Evasion",
    "Hesitance": "Evasion",
    "Others": "Evasion",
    "Personal Choice": "Defensive",
    "Self Pity": "Defensive",
    "Source Derogation": "Offensive",
    "Counter Argument": "Offensive",
    "Self-assertion": "Offensive"
}

unknown_strategy_count = {}  # 全局字典，统计未知策略的详细分布

def is_traj_success(trajectory):
    """判断单条轨迹是否成功"""
    if not trajectory: return False
    if trajectory.get('success'): return True
    turns = trajectory.get('turns', [])
    if not turns: return False
    for turn in turns:
        if turn.get('reward', 0.0) >= 1.0: return True
    if turns[-1].get('reward', 0.0) >= 1.0: return True
    return False

# =====================================================================
# 模块 2：时序轨迹全量重建 (新增 4D 认知特征：Hs, Ha, ΔHs, ΔHa)
# =====================================================================

def extract_trajectory_turns(trajectory, user_id="Unknown"):
    """解析按轮次对齐的字典，提取粗细粒度策略及熵值绝对值"""
    global unknown_strategy_count

    is_success = is_traj_success(trajectory)
    traj_id = trajectory.get('id', 'Unknown')

    turns_dict = {}
    current_round = 1  
    
    for turn in trajectory.get('turns', []):
        r = turn.get('round')
        if r is not None: current_round = r
            
        role = turn.get('role')
        if role == 'Persuader':
            sys_action = turn.get('strategy_name', "Unknown")
            hs = turn.get('hs')
            ha = turn.get('ha')
            
            # 未知策略统计逻辑
            if sys_action not in SYS_STRATEGY_MAP:
                if sys_action not in unknown_strategy_count:
                    unknown_strategy_count[sys_action] = {"Total_Count": 0, "Success_Count": 0, "Occurrences": []}
                unknown_strategy_count[sys_action]["Total_Count"] += 1
                if is_success: unknown_strategy_count[sys_action]["Success_Count"] += 1
                unknown_strategy_count[sys_action]["Occurrences"].append({
                    "User_ID": user_id, "Trajectory_ID": traj_id, "Turn": current_round, "Is_Success": is_success
                })

            if current_round not in turns_dict: turns_dict[current_round] = {}
            turns_dict[current_round]['Sys_Spec'] = sys_action
            turns_dict[current_round]['Sys_Cat'] = SYS_STRATEGY_MAP.get(sys_action, "Other")
            if hs is not None and ha is not None:
                turns_dict[current_round]['Hs'] = float(hs)
                turns_dict[current_round]['Ha'] = float(ha)
            
        elif role == 'Persuadee':
            user_action = turn.get('user_strategy', "None")
            if current_round not in turns_dict: turns_dict[current_round] = {}
            turns_dict[current_round]['User_Spec'] = user_action
            turns_dict[current_round]['User_Cat'] = USER_STRATEGY_MAP.get(user_action, "Other")
            
    # 转换为按轮次排序的列表
    sorted_turns = []
    for r in sorted(turns_dict.keys()):
        sorted_turns.append({
            'Turn': r, 
            'Sys_Spec': turns_dict[r].get('Sys_Spec', 'Other'), 
            'Sys_Cat': turns_dict[r].get('Sys_Cat', 'Other'), 
            'User_Spec': turns_dict[r].get('User_Spec', 'Other'), 
            'User_Cat': turns_dict[r].get('User_Cat', 'Other'),
            'Hs': turns_dict[r].get('Hs'),
            'Ha': turns_dict[r].get('Ha')
        })
        
    return sorted_turns

def process_sequences_single_user(user_item, exp_name):
    """提取三元组及 4D 认知状态向量，彻底剔除前缀冗余"""
    records = []
    user_id = user_item.get('user_id', 'Unknown') 
    
    trajectories = user_item.get('trajectories', [])
    root_traj = next((t for t in trajectories if t.get('id') == 'root'), None)
    if not root_traj: return []
    
    root_seq = extract_trajectory_turns(root_traj, user_id)
    root_success = is_traj_success(root_traj)
    
    def yield_records(seq, outcome, start_turn, 
                      init_p_sys_cat="START", init_p_sys_spec="START", 
                      init_p_user_cat="START", init_p_user_spec="START",
                      init_p_hs=None, init_p_ha=None):
        p_s_cat, p_s_spec = init_p_sys_cat, init_p_sys_spec
        p_u_cat, p_u_spec = init_p_user_cat, init_p_user_spec
        p_hs, p_ha = init_p_hs, init_p_ha
        
        for turn_data in seq:
            t = turn_data['Turn']
            c_s_cat, c_s_spec = turn_data['Sys_Cat'], turn_data['Sys_Spec']
            c_u_cat, c_u_spec = turn_data['User_Cat'], turn_data['User_Spec']
            c_hs, c_ha = turn_data['Hs'], turn_data['Ha']
            
            if t >= start_turn:
                if t <= 3: phase = 'Early (2-3)'
                elif t <= 6: phase = 'Mid (4-6)'
                else: phase = 'Late (7-10)'
                    
                record = {
                    'Experiment': exp_name, 'Outcome': outcome, 'Turn': t, 'Phase': phase,
                    'Prev_Sys_Cat': p_s_cat, 'Prev_Sys_Spec': p_s_spec,
                    'Prev_User_Cat': p_u_cat, 'Prev_User_Spec': p_u_spec,
                    'Curr_Sys_Cat': c_s_cat, 'Curr_Sys_Spec': c_s_spec,
                    'Curr_User_Cat': c_u_cat, 'Curr_User_Spec': c_u_spec,
                    'H_state': c_hs, 'H_action': c_ha,
                    'Delta_H_state': None, 'Delta_H_action': None, 'Dissonance': None
                }
                
                # 计算变化量与失调度
                if c_hs is not None and p_hs is not None: record['Delta_H_state'] = c_hs - p_hs
                if c_ha is not None and p_ha is not None: record['Delta_H_action'] = c_ha - p_ha
                if c_hs is not None and c_ha is not None: record['Dissonance'] = c_hs - c_ha
                    
                records.append(record)
            
            p_s_cat, p_s_spec = c_s_cat, c_s_spec
            p_u_cat, p_u_spec = c_u_cat, c_u_spec
            p_hs, p_ha = c_hs, c_ha

    # 1. Root 轨迹全提取
    yield_records(root_seq, 'Success' if root_success else 'Fail', start_turn=1)
    
    # 2. Branch 轨迹去重提取
    branch_trajs = [t for t in trajectories if t.get('id') != 'root']
    for b in branch_trajs:
        t_branch = b.get('branch_at_turn')
        if t_branch is not None:
            branch_seq = extract_trajectory_turns(b, user_id)
            context_turn = next((turn for turn in root_seq if turn['Turn'] == t_branch - 1), None)
            
            if context_turn:
                i_sys_cat, i_sys_spec = context_turn['Sys_Cat'], context_turn['Sys_Spec']
                i_user_cat, i_user_spec = context_turn['User_Cat'], context_turn['User_Spec']
                i_hs, i_ha = context_turn['Hs'], context_turn['Ha']
            else:
                i_sys_cat = i_sys_spec = i_user_cat = i_user_spec = "START"
                i_hs = i_ha = None
                
            yield_records(branch_seq, 'Success' if is_traj_success(b) else 'Fail', start_turn=t_branch,
                          init_p_sys_cat=i_sys_cat, init_p_sys_spec=i_sys_spec,
                          init_p_user_cat=i_user_cat, init_p_user_spec=i_user_spec,
                          init_p_hs=i_hs, init_p_ha=i_ha)
            
    return records

def build_sequence_datasets(experiments_config):
    all_data = []
    for config in experiments_config:
        filepath, exp_name = config.get('path'), config.get('name')
        if not os.path.exists(filepath): continue
        print(f"[Info] 解析轨迹数据与认知熵: {exp_name}")
        with open(filepath, 'r', encoding='utf-8') as f:
            for user_item in json.load(f):
                all_data.extend(process_sequences_single_user(user_item, exp_name))
    return pd.DataFrame(all_data)

# =====================================================================
# 模块 3：原有分析实验 (A / B / C) - 已全面融入分阶段逻辑
# =====================================================================

def plot_stacked_bar(data, col_name, title, filename, output_dir):
    pivot = pd.crosstab(index=[data['Outcome'], data['Phase']], columns=data[col_name], normalize='index') * 100
    phase_order = ['Early (2-3)', 'Mid (4-6)', 'Late (7-10)']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    colors = sns.color_palette("tab20", len(pivot.columns))
    
    for i, outcome in enumerate(['Success', 'Fail']):
        ax = axes[i]
        if outcome in pivot.index.get_level_values('Outcome'):
            sub_data = pivot.xs(outcome, level='Outcome').reindex(phase_order)
            sub_data.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='white')
            ax.set_title(f"{outcome} Trajectories", fontsize=14, pad=10)
            ax.set_xlabel('Dialogue Phase', fontsize=12)
            ax.tick_params(axis='x', rotation=0)
            if i == 0: ax.set_ylabel('Percentage (%)', fontsize=12)
            ax.legend().set_visible(False)
            
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=col_name.replace('_', ' '), loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4)
    plt.suptitle(title, fontsize=16, y=1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def analysis_A_phased_strategy_distribution(df, output_dir, prefix, granularity="Category"):
    print(f"  -> Running Exp A: Phased Strategy Distribution ({granularity})")
    sys_col = 'Curr_Sys_Cat' if granularity == "Category" else 'Curr_Sys_Spec'
    user_col = 'Curr_User_Cat' if granularity == "Category" else 'Curr_User_Spec'
    
    sys_df = df[df[sys_col] != 'Other']
    plot_stacked_bar(sys_df, sys_col, f'System Strategy Evolution ({granularity})', f'{prefix}_ExpA_System_Phase_{granularity}.png', output_dir)
    user_df = df[df[user_col] != 'Other']
    plot_stacked_bar(user_df, user_col, f'User Resistance Evolution ({granularity})', f'{prefix}_ExpA_User_Phase_{granularity}.png', output_dir)

def analysis_B_system_transition_heatmap(df, output_dir, prefix):
    print(f"  -> Running Exp B: System Transition Heatmap (Fine-grained Probabilities)")
    trans_df = df[(df['Prev_Sys_Spec'] != 'START') & (df['Prev_Sys_Spec'] != 'Other') & (df['Curr_Sys_Spec'] != 'Other')]
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), sharey=True)
    
    for i, outcome in enumerate(['Success', 'Fail']):
        ax = axes[i]
        sub_df = trans_df[trans_df['Outcome'] == outcome]
        if sub_df.empty: continue
        transition_matrix = pd.crosstab(sub_df['Prev_Sys_Spec'], sub_df['Curr_Sys_Spec'], normalize='index')
        sns.heatmap(transition_matrix, annot=True, cmap='Blues' if outcome=='Success' else 'Reds', 
                    fmt=".2f", ax=ax, cbar_kws={'label': 'Transition Probability'}, vmin=0, vmax=1.0, linewidths=0.5)
        ax.set_title(f"{outcome} Network (Transition Probabilities)", fontsize=15, pad=15)
        ax.set_xlabel('Current System Strategy ($S_t$)', fontsize=12)
        if i == 0: ax.set_ylabel('Previous System Strategy ($S_{t-1}$)', fontsize=12)
        else: ax.set_ylabel('')

    plt.suptitle(f"System Strategy Transition Heatmap ({prefix})", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_ExpB_Transition_Heatmap.png"), dpi=300)
    plt.close()

def analysis_C_interactive_markov_triplets(df, output_dir, prefix, granularity="Category"):
    """【需求 1】升级版：分阶段 (Early, Mid, Late, Overall) 三元组挖掘"""
    print(f"  -> Running Exp C: Interactive Markov Triplet Mining ({granularity}) - Stage-Faceted")
    
    sys_p = 'Prev_Sys_Cat' if granularity == "Category" else 'Prev_Sys_Spec'
    user_p = 'Prev_User_Cat' if granularity == "Category" else 'Prev_User_Spec'
    sys_c = 'Curr_Sys_Cat' if granularity == "Category" else 'Curr_Sys_Spec'
    
    valid_df = df[(df[sys_p] != 'START') & (df[user_p] != 'START') & (df[sys_c] != 'Other')].copy()
    valid_df['Triplet'] = valid_df[sys_p] + " -> " + valid_df[user_p] + " -> " + valid_df[sys_c]
    
    phases = ['Overall', 'Early (2-3)', 'Mid (4-6)', 'Late (7-10)']
    
    for phase in phases:
        phase_df = valid_df if phase == 'Overall' else valid_df[valid_df['Phase'] == phase]
        if phase_df.empty: continue
            
        succ_counts = phase_df[phase_df['Outcome'] == 'Success']['Triplet'].value_counts(normalize=True).reset_index(name='Prob_Success')
        fail_counts = phase_df[phase_df['Outcome'] == 'Fail']['Triplet'].value_counts(normalize=True).reset_index(name='Prob_Fail')
        
        comp_df = pd.merge(succ_counts, fail_counts, on='Triplet', how='outer').fillna(0.0001)
        if len(comp_df) > 1:
            kl_div = entropy(comp_df['Prob_Success'], comp_df['Prob_Fail'])
            print(f"     [{phase}] KL Divergence: {kl_div:.4f}")
            
        phase_str = phase.split(' ')[0]
        
        comp_df['Golden_Diff'] = comp_df['Prob_Success'] - comp_df['Prob_Fail']
        top_golden = comp_df.sort_values(by='Golden_Diff', ascending=False).head(15)
        top_golden.to_csv(os.path.join(output_dir, f"{prefix}_ExpC_Golden_{granularity}_{phase_str}.csv"), index=False)
        
        comp_df['Deadlock_Diff'] = comp_df['Prob_Fail'] - comp_df['Prob_Success']
        top_deadlock = comp_df.sort_values(by='Deadlock_Diff', ascending=False).head(15)
        top_deadlock.to_csv(os.path.join(output_dir, f"{prefix}_ExpC_Deadlock_{granularity}_{phase_str}.csv"), index=False)

# =====================================================================
# 模块 4：全新 4D 认知动力学分析 (E1 / E2 / E3)
# =====================================================================

def extract_top_rules(df, granularity="Specific"):
    """内部辅助函数：获取全局最核心的 Golden 和 Deadlock 模式"""
    sys_p = 'Prev_Sys_Cat' if granularity == "Category" else 'Prev_Sys_Spec'
    user_p = 'Prev_User_Cat' if granularity == "Category" else 'Prev_User_Spec'
    sys_c = 'Curr_Sys_Cat' if granularity == "Category" else 'Curr_Sys_Spec'
    
    v_df = df[(df[sys_p] != 'START') & (df[user_p] != 'START') & (df[sys_c] != 'Other')].copy()
    v_df['Triplet'] = v_df[sys_p] + " -> " + v_df[user_p] + " -> " + v_df[sys_c]
    
    succ = v_df[v_df['Outcome'] == 'Success']['Triplet'].value_counts(normalize=True)
    fail = v_df[v_df['Outcome'] == 'Fail']['Triplet'].value_counts(normalize=True)
    comp = pd.DataFrame({'Prob_Success': succ, 'Prob_Fail': fail}).fillna(0.0001)
    
    top_golden = (comp['Prob_Success'] - comp['Prob_Fail']).sort_values(ascending=False).head(5).index.tolist()
    top_deadlock = (comp['Prob_Fail'] - comp['Prob_Success']).sort_values(ascending=False).head(5).index.tolist()
    return top_golden, top_deadlock, v_df

def analysis_E1_stage_faceted_dynamics(df, output_dir, prefix):
    """【需求 2-E1】分阶段动作熵与变分箱线图及统计学检验"""
    print("  -> Running Exp E1: Stage-Faceted Dynamics Boxplots (Ha & ΔHa)")
    
    top_golden, top_deadlock, valid_df = extract_top_rules(df, "Specific")
    valid_df = valid_df.dropna(subset=['H_action', 'Delta_H_action']).copy()
    
    valid_df['Group'] = valid_df['Triplet'].apply(lambda t: 'Golden Rule' if t in top_golden else ('Deadlock' if t in top_deadlock else 'Other'))
    df_focus = valid_df[valid_df['Group'] != 'Other'].copy()
    
    # 构造包含 Overall 的数据副本
    df_overall = df_focus.copy()
    df_overall['Phase'] = 'Overall'
    df_plot = pd.concat([df_focus, df_overall])
    
    phase_order = ['Early (2-3)', 'Mid (4-6)', 'Late (7-10)', 'Overall']
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharey='row')
    colors = {'Golden Rule': '#2ecc71', 'Deadlock': '#e74c3c'}
    
    for i, phase in enumerate(phase_order):
        phase_data = df_plot[df_plot['Phase'] == phase]
        
        # 绝对动作熵 H_a
        sns.boxplot(data=phase_data, x='Group', y='H_action', palette=colors, ax=axes[0, i], width=0.5)
        axes[0, i].set_title(f"Phase: {phase}", fontsize=14)
        axes[0, i].set_xlabel("")
        axes[0, i].set_ylabel("Absolute Action Entropy ($H_a$)" if i == 0 else "", fontsize=12)
        
        # 动作熵变化量 ΔH_a
        sns.boxplot(data=phase_data, x='Group', y='Delta_H_action', palette=colors, ax=axes[1, i], width=0.5)
        axes[1, i].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1, i].set_xlabel("")
        axes[1, i].set_ylabel("Action Entropy Change ($\Delta H_a$)" if i == 0 else "", fontsize=12)
        
        # Mann-Whitney U 检验标注
        g_data = phase_data[phase_data['Group'] == 'Golden Rule']
        d_data = phase_data[phase_data['Group'] == 'Deadlock']
        if len(g_data) > 0 and len(d_data) > 0:
            _, p_ha = mannwhitneyu(g_data['H_action'], d_data['H_action'], alternative='two-sided')
            _, p_dha = mannwhitneyu(g_data['Delta_H_action'], d_data['Delta_H_action'], alternative='two-sided')
            axes[0, i].text(0.5, 0.9, f"p = {p_ha:.3f}", transform=axes[0, i].transAxes, ha='center', fontsize=11, fontweight='bold')
            axes[1, i].text(0.5, 0.9, f"p = {p_dha:.3f}", transform=axes[1, i].transAxes, ha='center', fontsize=11, fontweight='bold')

    plt.suptitle("Exp E1: Stage-Faceted Action Solidification Dynamics", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_ExpE1_Stage_Faceted_Dynamics.png"), dpi=300)
    plt.close()

def analysis_E2_phase_space_quiver(df, output_dir, prefix):
    """【需求 2-E2】状态-动作相空间与梯度向量场"""
    print("  -> Running Exp E2: 2D Phase Space Quiver (Vector Field)")
    
    top_golden, top_deadlock, valid_df = extract_top_rules(df, "Specific")
    valid_df = valid_df.dropna(subset=['H_state', 'H_action', 'Delta_H_state', 'Delta_H_action']).copy()
    valid_df['Group'] = valid_df['Triplet'].apply(lambda t: 'Golden Rule' if t in top_golden else ('Deadlock' if t in top_deadlock else 'Other'))
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    colors = ['#2ecc71', '#e74c3c']
    max_hs, max_ha = valid_df['H_state'].max(), valid_df['H_action'].max()
    
    for i, group in enumerate(['Golden Rule', 'Deadlock']):
        ax = axes[i]
        group_df = valid_df[valid_df['Group'] == group]
        
        sns.scatterplot(data=group_df, x='H_state', y='H_action', color=colors[i], alpha=0.4, ax=ax, s=60, edgecolor='white')
        
        # 为了保证向量图的清晰度，进行随机降采样抽样
        sample_df = group_df.sample(min(250, len(group_df)), random_state=42) if len(group_df) > 250 else group_df
        ax.quiver(sample_df['H_state'], sample_df['H_action'], sample_df['Delta_H_state'], sample_df['Delta_H_action'], 
                  color=colors[i], alpha=0.7, angles='xy', scale_units='xy', scale=1, width=0.004, headwidth=4)
                  
        ax.plot([0, max_hs], [0, max_ha], 'k--', alpha=0.4, label="Alignment ($H_s = H_a$)")
        ax.axvspan(max_hs*0.6, max_hs, ymin=0, ymax=0.4, color='red', alpha=0.08)
        ax.text(max_hs*0.8, max_ha*0.2, "Gravity Well\n(Blind Confidence)", ha='center', color='darkred', alpha=0.7, fontsize=12, fontweight='bold')
        
        ax.set_title(f"{group} Transitions Vector Field", fontsize=15, pad=10)
        ax.set_xlabel("Absolute State Entropy ($H_s$) - Env Difficulty", fontsize=13)
        if i == 0: ax.set_ylabel("Absolute Action Entropy ($H_a$) - Execution Flex", fontsize=13)
        ax.legend(loc='upper left')

    plt.suptitle("Exp E2: Strategy Transitions as Cognitive Gradient Vectors", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_ExpE2_Phase_Space_Quiver.png"), dpi=300)
    plt.close()

def analysis_E3_cognitive_dissonance_evolution(df, output_dir, prefix):
    """【需求 2-E3】高维认知失调指数的阶段演化"""
    print("  -> Running Exp E3: Evolution of Cognitive Dissonance")
    
    top_golden, top_deadlock, valid_df = extract_top_rules(df, "Specific")
    valid_df = valid_df.dropna(subset=['Dissonance']).copy()
    valid_df['Group'] = valid_df['Triplet'].apply(lambda t: 'Golden Rule' if t in top_golden else ('Deadlock' if t in top_deadlock else 'Other'))
    df_focus = valid_df[valid_df['Group'] != 'Other'].copy()
    
    phase_mapping = {'Early (2-3)': 1, 'Mid (4-6)': 2, 'Late (7-10)': 3}
    df_focus['Phase_Num'] = df_focus['Phase'].map(phase_mapping)
    df_focus = df_focus.dropna(subset=['Phase_Num'])
    
    plt.figure(figsize=(10, 7))
    sns.lineplot(data=df_focus, x='Phase_Num', y='Dissonance', hue='Group', 
                 palette={'Golden Rule': '#2ecc71', 'Deadlock': '#e74c3c'}, 
                 marker='o', markersize=10, linewidth=3, errorbar=('ci', 95))
    
    plt.xticks([1, 2, 3], ['Early (2-3)', 'Mid (4-6)', 'Late (7-10)'], fontsize=12)
    plt.axhline(0, color='black', linestyle='--', alpha=0.6, label='Zero Dissonance ($H_s = H_a$)')
    
    plt.title("Exp E3: Evolution of Cognitive Dissonance ($D = H_s - H_a$)", fontsize=16, pad=15)
    plt.xlabel("Dialogue Phase", fontsize=13)
    plt.ylabel("Cognitive Dissonance ($D$)", fontsize=13)
    plt.legend(title='Triplet Type', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_ExpE3_Cognitive_Dissonance.png"), dpi=300)
    plt.close()

# =====================================================================
# 主程序入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', required=True, help="List of EXP result json files")
    parser.add_argument("--names", nargs='+', required=True, help="List of Model Names")
    parser.add_argument("--output_dir", default="eval_results", help="Directory to save analysis results")
    args = parser.parse_args()

    if len(args.files) != len(args.names):
        print("[Error] Number of files and names must match.")
        return
        
    EXPERIMENTS_CONFIG = [{"name": n, "path": p} for p, n in zip(args.files, args.names)]
    os.makedirs(args.output_dir, exist_ok=True)

    df_sequence = build_sequence_datasets(EXPERIMENTS_CONFIG)
    if df_sequence.empty:
        print("[Error] No sequence data parsed.")
        return
        
    df_sequence.to_csv(os.path.join(args.output_dir, "rl_sequence_dataset_4D_cognitive.csv"), index=False)
    print(f"\n✅ 成功！提取包含 4D 认知特征的交互节点记录: {len(df_sequence)} 条。")

    for model in EXPERIMENTS_CONFIG:
        exp_name = model["name"]
        print(f"\n{'='*55}\n--- Starting Full Joint Dynamics Analysis For: {exp_name} ---\n{'='*55}")
        df_model = df_sequence[df_sequence["Experiment"] == exp_name]
        prefix = f"{exp_name}_Sequence"
        
        # --- 保留的原有宏观分析 ---
        analysis_A_phased_strategy_distribution(df_model, args.output_dir, prefix, granularity="Category")
        analysis_A_phased_strategy_distribution(df_model, args.output_dir, prefix, granularity="Specific")
        analysis_B_system_transition_heatmap(df_model, args.output_dir, prefix)
        analysis_C_interactive_markov_triplets(df_model, args.output_dir, prefix, granularity="Category")
        analysis_C_interactive_markov_triplets(df_model, args.output_dir, prefix, granularity="Specific")
        
        # --- 全新 4D 认知动力学分析 ---
        analysis_E1_stage_faceted_dynamics(df_model, args.output_dir, prefix)
        analysis_E2_phase_space_quiver(df_model, args.output_dir, prefix)
        analysis_E3_cognitive_dissonance_evolution(df_model, args.output_dir, prefix)

    # 保存未知策略追踪日志
    global unknown_strategy_count
    output_path = os.path.join(args.output_dir, "unknown_strategy_tracking.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unknown_strategy_count, f, indent=4, ensure_ascii=False)
    print(f"\n[检查] 共发现 {len(unknown_strategy_count)} 种未在映射表中的系统策略。已保存至: {output_path}")

if __name__ == "__main__":
    main()