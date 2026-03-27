import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings("ignore")

# =====================================================================
# 模块 1：基础配置与策略字典 (Configurations & Maps)
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

PERSONA_BASELINE_SR_MAP = {
    "Qwen3-8B": {
        "Analytical": 0.3571, "Behavioral": 0.3846, "Conceptual": 0.1111, "Directive": 0.3571,
        "Agreeableness": 0.4167, "Conscientiousness": 0.3000, "Extraversion": 0.3750, 
        "Neuroticism": 0.4000, "Openness": 0.1000
    },
    "Qwen3-8B-Thinking": {
        "Analytical": 0.4286, "Behavioral": 0.5385, "Conceptual": 0.3333, "Directive": 0.3571,
        "Agreeableness": 0.8333, "Conscientiousness": 0.0001, "Extraversion": 0.5000, 
        "Neuroticism": 0.4000, "Openness": 0.3000
    }
}

def is_traj_success(trajectory):
    """判断单条轨迹是否成功 (返回 1.0 或 0.0)"""
    if not trajectory: return 0.0
    if trajectory.get('success'): return 1.0
    turns = trajectory.get('turns', [])
    if not turns: return 0.0
    for turn in turns:
        if turn.get('reward', 0.0) >= 1.0: return 1.0
    if turns[-1].get('reward', 0.0) >= 1.0: return 1.0
    return 0.0

def robust_persona_mapping(persona_str):
    if pd.isna(persona_str): return "Unknown"
    return str(persona_str).strip().title()

def get_v_bar(row):
    exp = row.get('Experiment', 'Unknown')
    persona = row.get('Persona', 'Unknown')
    model_dict = PERSONA_BASELINE_SR_MAP.get(exp, PERSONA_BASELINE_SR_MAP["Qwen3-8B-Thinking"])
    return model_dict.get(persona, 0.35)

def load_and_preprocess_data(csv_data=None, persona_dict=None, target_dimension="Big-Five Personality"):
    """安全的数据预处理，兼容节点级和动作级两张表"""
    df = csv_data.copy()
    df['Persona'] = df['User_ID'].apply(lambda uid: persona_dict.get(uid, {}).get(target_dimension, "Unknown"))
    df['Persona'] = df['Persona'].apply(robust_persona_mapping)
    df['V_bar_persona'] = df.apply(get_v_bar, axis=1)
    
    if 'Advantage' in df.columns:
        df['Abs_Advantage'] = df['Advantage'].abs()
    if 'Progress' in df.columns:
        df['Abs_Progress'] = df['Progress'].abs()
    return df

# =====================================================================
# 模块 2 & 3：强化学习指标计算与数据提取 (Node-Level / 原有逻辑保留)
# =====================================================================

def calculate_counterfactual_advantage(q_orig, branch_outcomes):
    if not branch_outcomes: return 0.0
    return q_orig - np.mean(branch_outcomes)

def calculate_step_progress(current_v, prev_v):
    if prev_v is None: return 0.0
    return current_v - prev_v

def extract_node_features(root_traj):
    turn_features = {}
    turns = root_traj.get('turns', [])
    last_user_strategy = "None"
    prev_hs, prev_ha = None, None
    for turn in turns:
        r, role = turn.get('round'), turn.get('role')
        if role == 'Persuadee':
            last_user_strategy = turn.get('user_strategy', "None")
        elif role == 'Persuader':
            hs_val, ha_val = turn.get('hs'), turn.get('ha')
            sys_strategy = turn.get('strategy_name', "Unknown")
            if hs_val is not None and ha_val is not None:
                hs, ha = float(hs_val), float(ha_val)
                dissonance = hs - ha
                delta_hs = hs - prev_hs if prev_hs is not None else 0.0
                delta_ha = ha - prev_ha if prev_ha is not None else 0.0
                turn_features[r] = {
                    'H_state': hs, 'H_action': ha, 'Dissonance': dissonance,
                    'Delta_H_state': delta_hs, 'Delta_H_action': delta_ha, 'S_Trend': delta_hs - delta_ha,
                    'Sys_Strategy': sys_strategy, 'Sys_Category': SYS_STRATEGY_MAP.get(sys_strategy, "Other"),
                    'Prev_User_Strategy': last_user_strategy, 'Prev_User_Category': USER_STRATEGY_MAP.get(last_user_strategy, "Other")
                }
                prev_hs, prev_ha = hs, ha
    return turn_features

def process_single_user(user_item, exp_name):
    """节点级数据处理 (Node-Level)，用于分析 1 和 2"""
    user_id = user_item.get('user_id', 'unknown')
    trajectories = user_item.get('trajectories', [])
    root_traj = next((t for t in trajectories if t.get('id') == 'root'), None)
    if not root_traj: return []
    
    q_orig = float(is_traj_success(root_traj))
    turn_features = extract_node_features(root_traj)
    
    branch_trajs = [t for t in trajectories if t.get('id') != 'root']
    branches_by_turn = defaultdict(list)
    for b in branch_trajs:
        t_branch = b.get('branch_at_turn')
        if t_branch is not None:
            branches_by_turn[t_branch].append(float(is_traj_success(b)))
            
    sorted_turns = sorted(list(branches_by_turn.keys()))
    node_records = []
    v_history = {}
    for t in sorted_turns:
        if t not in turn_features: continue
        branch_outcomes = branches_by_turn[t]
        v_t = np.mean(branch_outcomes + [q_orig])
        v_history[t] = v_t
        adv_t = calculate_counterfactual_advantage(q_orig, branch_outcomes)
        prev_idx = sorted_turns.index(t) - 1
        prev_v = v_history[sorted_turns[prev_idx]] if prev_idx >= 0 else v_t
        
        record = {
            'Experiment': exp_name, 'User_ID': user_id, 'Turn': t,
            'Q_orig': q_orig, 'V_state': v_t, 'Advantage': adv_t, 'Progress': calculate_step_progress(v_t, prev_v)
        }
        record.update(turn_features[t])
        node_records.append(record)
    return node_records

# =====================================================================
# [新增模块] 模块 3.5：纯动作级数据提取 (Action-Level Pipeline)
# =====================================================================

# =====================================================================
# [修复版] 模块 3.5：纯动作级数据提取 (Action-Level Pipeline)
# =====================================================================

def process_single_user_action_level(user_item, exp_name):
    user_id = user_item.get('user_id', 'unknown')
    trajectories = user_item.get('trajectories', [])
    root_traj = next((t for t in trajectories if t.get('id') == 'root'), None)
    if not root_traj: return []
        
    # 提取每个回合的共享上下文 (包括 Prev_User_Category 和 上一轮的熵值)
    state_context = {}
    last_user_strategy = "None"
    last_hs = None
    last_ha = None
    
    for turn in root_traj.get('turns', []):
        r = turn.get('round')
        role = turn.get('role')
        if role == 'Persuadee':
            last_user_strategy = turn.get('user_strategy', "None")
        elif role == 'Persuader':
            state_context[r] = {
                'Prev_User_Strategy': last_user_strategy,
                'Prev_User_Category': USER_STRATEGY_MAP.get(last_user_strategy, "Other"),
                'Prev_H_state': last_hs,
                'Prev_H_action': last_ha
            }
            # 记录当前轮的熵值，留给下一轮作为 Prev 使用
            hs_val, ha_val = turn.get('hs'), turn.get('ha')
            last_hs = float(hs_val) if hs_val is not None else None
            last_ha = float(ha_val) if ha_val is not None else None

    branch_trajs = [t for t in trajectories if t.get('id') != 'root']
    actions_by_turn = defaultdict(list)
    
    # 1. 压入 Root 动作
    for turn in root_traj.get('turns', []):
        if turn.get('role') == 'Persuader':
            r = turn.get('round')
            hs_val, ha_val = turn.get('hs'), turn.get('ha')
            actions_by_turn[r].append({
                'traj_type': 'root',
                'Sys_Strategy': turn.get('strategy_name', "Unknown"),
                'Outcome_Q': float(is_traj_success(root_traj)),
                'H_state': float(hs_val) if hs_val is not None else None,
                'H_action': float(ha_val) if ha_val is not None else None
            })
            
    # 2. 压入所有 Branch 动作
    for b in branch_trajs:
        t_branch = b.get('branch_at_turn')
        if t_branch is not None:
            for turn in b.get('turns', []):
                if turn.get('round') == t_branch and turn.get('role') == 'Persuader':
                    hs_val, ha_val = turn.get('hs'), turn.get('ha')
                    actions_by_turn[t_branch].append({
                        'traj_type': 'branch',
                        'Sys_Strategy': turn.get('strategy_name', "Unknown"),
                        'Outcome_Q': float(is_traj_success(b)),
                        'H_state': float(hs_val) if hs_val is not None else None,
                        'H_action': float(ha_val) if ha_val is not None else None
                    })
                    break

    action_records = []
    # 3. 计算每个动作的 Advantage 以及各种时序变化量
    for t, actions in actions_by_turn.items():
        if t not in state_context: continue
        if len(actions) <= 1: continue 
            
        v_state = np.mean([a['Outcome_Q'] for a in actions])
        
        for a in actions:
            sys_category = SYS_STRATEGY_MAP.get(a['Sys_Strategy'], "Other")
            action_advantage = a['Outcome_Q'] - v_state 
            
            hs = a['H_state']
            ha = a['H_action']
            prev_hs = state_context[t]['Prev_H_state']
            prev_ha = state_context[t]['Prev_H_action']
            
            # 计算失调和动态变化量 (如果缺失则补 0)
            dissonance = (hs - ha) if (hs is not None and ha is not None) else None
            delta_hs = (hs - prev_hs) if (hs is not None and prev_hs is not None) else 0.0
            delta_ha = (ha - prev_ha) if (ha is not None and prev_ha is not None) else 0.0
            
            record = {
                'Experiment': exp_name, 'User_ID': user_id, 'Turn': t,
                'Traj_Type': a['traj_type'], 'V_state': v_state, 
                'Action_Q': a['Outcome_Q'], 'Advantage': action_advantage, 
                'H_state': hs, 'H_action': ha, 
                'Delta_H_state': delta_hs, 'Delta_H_action': delta_ha, # [新增] 熵的时序变化量
                'Dissonance': dissonance,
                'Sys_Strategy': a['Sys_Strategy'], 'Sys_Category': sys_category,
                'Prev_User_Strategy': state_context[t]['Prev_User_Strategy'],
                'Prev_User_Category': state_context[t]['Prev_User_Category']
            }
            action_records.append(record)
            
    return action_records

def build_rl_datasets(experiments_config):
    """同时生成节点级 (Node-level) 和 动作级 (Action-level) 两个数据集"""
    all_node_data = []
    all_action_data = []
    for config in experiments_config:
        exp_name = config.get('name')
        filepath = config.get('path')
        if not os.path.exists(filepath): continue
            
        print(f"[Info] 解析 RL 数据: {exp_name}")
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for user_item in data:
                all_node_data.extend(process_single_user(user_item, exp_name))
                all_action_data.extend(process_single_user_action_level(user_item, exp_name))
                
    return pd.DataFrame(all_node_data), pd.DataFrame(all_action_data)

# =====================================================================
# 模块 4：三大核心分析画图函数
# =====================================================================

def analysis_1_cognitive_signals_over_time(df, output_dir="results", prefix=None):
    # (保留原有代码逻辑)
    early_stage = df[df['Turn'] < 4]
    late_stage = df[df['Turn'] >= 4]
    signals = ['H_action', 'Dissonance', 'S_Trend']
    results = []
    for signal in signals:
        r_early, _ = spearmanr(early_stage[signal], early_stage['Abs_Advantage'], nan_policy='omit')
        r_late, _ = spearmanr(late_stage[signal], late_stage['Abs_Advantage'], nan_policy='omit')
        results.append({'Signal': signal, 'Stage': 'Early (Turn < 4)', 'Spearman_Rho': r_early})
        results.append({'Signal': signal, 'Stage': 'Late (Turn >= 4)', 'Spearman_Rho': r_late})
        
    df_res = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_res, x='Signal', y='Spearman_Rho', hue='Stage', palette=['#3498db', '#e74c3c'])
    plt.title('Predictive Power of Cognitive Signals on |Advantage| over Time', fontsize=14, pad=15)
    plt.xlabel('Internal Cognitive Signals', fontsize=12)
    plt.ylabel('Spearman Correlation ($\\rho$)', fontsize=12)
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_analysis1_temporal_signals.png"), dpi=300)
    plt.close()

def analysis_2_persona_difficulty_tolerance(df, output_dir="results", prefix=None):
    # (保留原有代码逻辑)
    failures = df[df['Advantage'] < 0].copy()
    successes = df[df['Advantage'] > 0].copy()
    failures['Penalty_Amplitude'] = failures['Advantage'].abs()
    successes['Reward_Amplitude'] = successes['Advantage'].abs()
    
    fail_agg = failures.groupby('Persona').agg(V_bar=('V_bar_persona', 'mean'), Mean_Penalty=('Penalty_Amplitude', 'mean')).reset_index()
    succ_agg = successes.groupby('Persona').agg(V_bar=('V_bar_persona', 'mean'), Mean_Reward=('Reward_Amplitude', 'mean')).reset_index()
    df_res = pd.merge(fail_agg, succ_agg, on=['Persona', 'V_bar'], how='outer')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.regplot(ax=axes[0], data=df_res, x='V_bar', y='Mean_Penalty', scatter_kws={'s': 100, 'color': '#e74c3c'}, line_kws={'color': 'gray', 'linestyle': '--'})
    axes[0].set_title('Cost of Failure vs. Environment Difficulty')
    sns.regplot(ax=axes[1], data=df_res, x='V_bar', y='Mean_Reward', scatter_kws={'s': 100, 'color': '#2ecc71'}, line_kws={'color': 'gray', 'linestyle': '--'})
    axes[1].set_title('Reward of Success vs. Environment Difficulty')
    for i, row in df_res.iterrows():
        axes[0].text(row['V_bar']+0.01, row['Mean_Penalty'], row['Persona'], fontsize=9, alpha=0.7)
        axes[1].text(row['V_bar']+0.01, row['Mean_Reward'], row['Persona'], fontsize=9, alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_analysis2_persona_tolerance.png"), dpi=300)
    plt.close()

def analysis_3_action_advantage_heatmap(df_action, output_dir="results", prefix=None):
    """
    【全新设计】分析3：使用特定动作级数据表绘制策略因果图谱
    展示在特定的防守状态下，采取特定的策略动作能带来多少独立优势 (Advantage)。
    """
    print("\n--- Running Analysis 3 (Action-Level): Strategy Transition vs. Action Advantage ---")
    df_valid = df_action[(df_action['Prev_User_Category'] != 'Other') & (df_action['Sys_Category'] != 'Other')].copy()
    
    # 【核心修正】：用 Advantage 取代 Progress
    transition_pivot = df_valid.pivot_table(
        values='Advantage', 
        index='Prev_User_Category', 
        columns='Sys_Category', 
        aggfunc='mean'
    ).fillna(0)
    
    csv_path = os.path.join(output_dir, f"{prefix}_analysis3_action_advantage.csv" if prefix else "analysis3_action_advantage.csv")
    transition_pivot.to_csv(csv_path)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_pivot, annot=True, cmap='RdYlGn', center=0, fmt=".3f", 
                linewidths=.5, cbar_kws={'label': 'Mean Action Advantage ($A_t$)'})
    
    plt.title('Causal Map: Action Strategy Impact on Expected Advantage', fontsize=16, pad=15)
    plt.xlabel('Current Specific System Action ($a_t$ category)', fontsize=12)
    plt.ylabel('Previous User State ($S_{user, t-1}$)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_analysis3_action_advantage.png" if prefix else "analysis3_action_advantage.png"), dpi=300)
    plt.close()
    print(f"  -> Saved to {prefix}_analysis3_action_advantage.png")

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# =====================================================================
# 模块 4：分析 4 - 全因素关联度与时序课程奖励对比 (+ 提取决策树)
# =====================================================================

def analysis_4_comprehensive_factor_importance(df_action, persona_dict, output_dir="results", prefix=None, model_name=None):
    print(f"\n--- Running Analysis 4: Comprehensive Factor Importance & Temporal Reward ({prefix}) ---")
    
    df_valid = df_action.dropna(subset=['H_state', 'H_action', 'Advantage', 'Delta_H_state']).copy()
    if len(df_valid) < 20:
        print("[Warning] Not enough data for Analysis 4.")
        return

    df_valid['Big_Five_Label'] = df_valid['User_ID'].apply(
        lambda uid: robust_persona_mapping(persona_dict.get(uid, {}).get("Big-Five Personality", "Unknown")))
    df_valid['Decision_Making_Label'] = df_valid['User_ID'].apply(
        lambda uid: robust_persona_mapping(persona_dict.get(uid, {}).get("Decision-Making Style", "Unknown")))

    def get_specific_v_bar(row, persona_col):
        exp = row.get('Experiment', 'Unknown')
        persona = row.get(persona_col, 'Unknown')
        model_dict = PERSONA_BASELINE_SR_MAP.get(exp, PERSONA_BASELINE_SR_MAP["Qwen3-8B-Thinking"])
        return model_dict.get(persona, 0.35)

    df_valid['V_bar_BigFive'] = df_valid.apply(lambda row: get_specific_v_bar(row, 'Big_Five_Label'), axis=1)
    df_valid['V_bar_DecisionMaking'] = df_valid.apply(lambda row: get_specific_v_bar(row, 'Decision_Making_Label'), axis=1)

    max_turn = df_valid['Turn'].max() if df_valid['Turn'].max() > 0 else 10
    df_valid['Temporal_Weight'] = 0.5 + (df_valid['Turn'] / max_turn)
    df_valid['Temporal_Advantage'] = df_valid['Advantage'] * df_valid['Temporal_Weight']

    features_num = [
        'Turn', 'H_state', 'H_action', 
        'Delta_H_state', 'Delta_H_action', 'Dissonance', 
        'V_bar_BigFive', 'V_bar_DecisionMaking'
    ]
    features_cat = ['Sys_Strategy', 'Sys_Category', 'Prev_User_Strategy', 'Prev_User_Category']
    
    X = pd.get_dummies(df_valid[features_num + features_cat], columns=features_cat)
    
    def aggregate_importances(importances, feature_names):
        agg_dict = {
            'Dialogue Turn (Time)': 0, 'State Entropy (H_s)': 0, 'Action Entropy (H_a)': 0, 
            'State Entropy Delta (\u0394H_s)': 0, 'Action Entropy Delta (\u0394H_a)': 0, 
            'Cognitive Dissonance': 0, 'Big-Five Difficulty': 0, 'Decision-Making Difficulty': 0,
            'System Strategy (Specific & Category)': 0, 'User Resistance (Specific & Category)': 0
        }
        for name, imp in zip(feature_names, importances):
            if name == 'Turn': agg_dict['Dialogue Turn (Time)'] += imp
            elif name == 'H_state': agg_dict['State Entropy (H_s)'] += imp
            elif name == 'H_action': agg_dict['Action Entropy (H_a)'] += imp
            elif name == 'Delta_H_state': agg_dict['State Entropy Delta (\u0394H_s)'] += imp
            elif name == 'Delta_H_action': agg_dict['Action Entropy Delta (\u0394H_a)'] += imp
            elif name == 'Dissonance': agg_dict['Cognitive Dissonance'] += imp
            elif name == 'V_bar_BigFive': agg_dict['Big-Five Difficulty'] += imp
            elif name == 'V_bar_DecisionMaking': agg_dict['Decision-Making Difficulty'] += imp
            elif name.startswith('Sys_Strategy') or name.startswith('Sys_Category'):
                agg_dict['System Strategy (Specific & Category)'] += imp
            elif name.startswith('Prev_User_Strategy') or name.startswith('Prev_User_Category'):
                agg_dict['User Resistance (Specific & Category)'] += imp
        return agg_dict

    # 训练随机森林模型
    rf_orig = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_orig.fit(X, df_valid['Advantage'])
    imp_orig = aggregate_importances(rf_orig.feature_importances_, X.columns)

    rf_temp = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    rf_temp.fit(X, df_valid['Temporal_Advantage'])
    imp_temp = aggregate_importances(rf_temp.feature_importances_, X.columns)

    # =========================================================================
    # [新增] 提取并可视化随机森林中的一棵代表性决策树 (Sample Decision Tree)
    # =========================================================================
    print("  -> Generating Sample Decision Tree Visualization...")
    tree_estimator = rf_temp.estimators_[0] # 提取时序加权森林中的第1棵树
    
    plt.figure(figsize=(24, 12))
    # 限制 max_depth=3 是为了保证图片肉眼可读。如果想看全貌，可以把 max_depth 设为 None
    plot_tree(tree_estimator, feature_names=X.columns, filled=True, rounded=True, max_depth=3, fontsize=10, proportion=True)
    plt.title(f'Sample Decision Tree Logic from Random Forest ({prefix} - Temporal Curriculum)\n(Showing Top 3 Levels for Readability)', fontsize=18, pad=20)
    
    tree_path = os.path.join(output_dir, f"{prefix}_analysis4_sample_decision_tree.png" if prefix else "analysis4_sample_decision_tree.png")
    plt.tight_layout()
    plt.savefig(tree_path, dpi=300)
    plt.close()
    print(f"  -> Saved Sample Decision Tree to {tree_path}")
    # =========================================================================

    # 组装数据并绘制重要性条形图
    df_imp = pd.DataFrame([
        {'Factor': k, 'Importance': v, 'Reward_Type': 'Original Advantage'} for k, v in imp_orig.items()
    ] + [
        {'Factor': k, 'Importance': v, 'Reward_Type': 'Temporal Curriculum Advantage'} for k, v in imp_temp.items()
    ])

    csv_path = os.path.join(output_dir, f"{prefix}_analysis4_factor_importance.csv" if prefix else "analysis4_factor_importance.csv")
    df_imp.to_csv(csv_path, index=False)

    plt.figure(figsize=(12, 8))
    order_list = df_imp[df_imp['Reward_Type'] == 'Original Advantage'].sort_values(by='Importance', ascending=False)['Factor'].tolist()
    sns.barplot(data=df_imp, x='Importance', y='Factor', hue='Reward_Type', order=order_list, palette=['#95a5a6', '#e74c3c'])
    
    plt.title(f'{model_name}: Comprehensive Factor Importance on Action Advantage \n(Original vs. Temporal Curriculum Weighted)', fontsize=15, pad=15)
    plt.xlabel('Relative Contribution to Advantage (Random Forest Feature Importance)', fontsize=12)
    plt.ylabel('Interaction Factors', fontsize=12)
    plt.legend(title='Reward Design Mode', loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_analysis4_factor_importance.png" if prefix else "analysis4_factor_importance.png"), dpi=300)
    plt.close()
    print(f"  -> Saved to {prefix}_analysis4_factor_importance.png")


from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# =====================================================================
# 模块 5：分析 5 - 线性回归相关性与方向性分析 (保留全量特征并导出CSV)
# =====================================================================
def analysis_5_linear_regression_correlation(df_action, persona_dict, output_dir="results", prefix=None, model_name=None):
    print(f"\n--- Running Analysis 5: Linear Regression Directional Impact ({prefix}) ---")
    
    df_valid = df_action.dropna(subset=['H_state', 'H_action', 'Advantage', 'Delta_H_state']).copy()
    if len(df_valid) < 20:
        print("[Warning] Not enough data for Analysis 5.")
        return

    df_valid['Big_Five_Label'] = df_valid['User_ID'].apply(
        lambda uid: robust_persona_mapping(persona_dict.get(uid, {}).get("Big-Five Personality", "Unknown")))
    df_valid['Decision_Making_Label'] = df_valid['User_ID'].apply(
        lambda uid: robust_persona_mapping(persona_dict.get(uid, {}).get("Decision-Making Style", "Unknown")))

    def get_specific_v_bar(row, persona_col):
        exp = row.get('Experiment', 'Unknown')
        persona = row.get(persona_col, 'Unknown')
        model_dict = PERSONA_BASELINE_SR_MAP.get(exp, PERSONA_BASELINE_SR_MAP["Qwen3-8B-Thinking"])
        return model_dict.get(persona, 0.35)

    df_valid['V_bar_BigFive'] = df_valid.apply(lambda row: get_specific_v_bar(row, 'Big_Five_Label'), axis=1)
    df_valid['V_bar_DecisionMaking'] = df_valid.apply(lambda row: get_specific_v_bar(row, 'Decision_Making_Label'), axis=1)

    max_turn = df_valid['Turn'].max() if df_valid['Turn'].max() > 0 else 10
    df_valid['Temporal_Weight'] = 0.5 + (df_valid['Turn'] / max_turn)
    df_valid['Temporal_Advantage'] = df_valid['Advantage'] * df_valid['Temporal_Weight']

    features_num = [
        'Turn', 'H_state', 'H_action', 
        'Delta_H_state', 'Delta_H_action', 
        'V_bar_BigFive', 'V_bar_DecisionMaking'
    ]
    features_cat = ['Sys_Category', 'Prev_User_Category']
    
    scaler = StandardScaler()
    df_valid[features_num] = scaler.fit_transform(df_valid[features_num])
    X = pd.get_dummies(df_valid[features_num + features_cat], columns=features_cat)
    
    model_orig = Ridge(alpha=1.0, random_state=42)
    model_orig.fit(X, df_valid['Advantage'])
    
    model_temp = Ridge(alpha=1.0, random_state=42)
    model_temp.fit(X, df_valid['Temporal_Advantage'])

    coef_orig = model_orig.coef_
    coef_temp = model_temp.coef_
    
    clean_names = []
    for name in X.columns:
        name = name.replace('Sys_Category_', 'Sys: ')
        name = name.replace('Prev_User_Category_', 'User: ')
        clean_names.append(name)

    df_coef = pd.DataFrame({
        'Feature': clean_names * 2,
        'Coefficient': list(coef_orig) + list(coef_temp),
        'Reward_Type': ['Original Advantage'] * len(clean_names) + ['Temporal Curriculum Advantage'] * len(clean_names)
    })

    # =========================================================================
    # [修改] 不过度过滤，保留所有特征 (>= 0.0)
    # [修改] 将结果保存至 CSV 文件
    # =========================================================================
    df_coef = df_coef[df_coef['Coefficient'].abs() >= 0.0].copy()
    
    csv_path = os.path.join(output_dir, f"{prefix}_analysis5_linear_coefficients.csv" if prefix else "analysis5_linear_coefficients.csv")
    df_coef.to_csv(csv_path, index=False)
    print(f"  -> Saved Coefficients CSV to {csv_path}")

    # 绘制图形
    plt.figure(figsize=(12, 10))
    sort_order = df_coef[df_coef['Reward_Type'] == 'Original Advantage'].sort_values(by='Coefficient', ascending=False)['Feature'].tolist()
    
    sns.barplot(data=df_coef, x='Coefficient', y='Feature', hue='Reward_Type', order=sort_order, palette=['#95a5a6', '#e74c3c'])
    plt.axvline(0, color='black', linewidth=1.5)
    
    plt.title(f'{model_name} - Linear Regression Coefficients on Action Advantage\n(Positive = Increases Win Rate, Negative = Decreases Win Rate)', fontsize=15, pad=15)
    plt.xlabel('Regression Coefficient (Directional Impact)', fontsize=12)
    plt.ylabel('Interaction Factors', fontsize=12)
    plt.legend(title='Reward Design Mode', loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_analysis5_linear_coefficients.png" if prefix else "analysis5_linear_coefficients.png"), dpi=300)
    plt.close()
    print(f"  -> Saved to {prefix}_analysis5_linear_coefficients.png")

# =====================================================================
# [修改版] 模块 4：分析 6 - 全局熵值时序演变趋势 (仅均值)
# 去除了方差的计算与绘制，专注于宏观的探索/利用趋势
# =====================================================================
def analysis_6_global_entropy_trend(df_action, output_dir="results", prefix=None):
    """
    分析6：可视化所有轨迹的状态熵与动作熵随对话轮次的变化趋势。
    【核心修正】：基于 (User_ID, Turn, 状态熵, 动作熵, Outcome) 进行严格去重，
    彻底消除由于多条干预分支轨迹共享相同初始前缀而导致的“重复计算(Double Counting)”问题。
    """
    print(f"\n--- Running Analysis 6: Global Mean Entropy Trend (Deduplicated) ({prefix}) ---")
    
    # 1. 过滤掉没有熵值的无效数据
    df_valid = df_action.dropna(subset=['H_state', 'H_action', 'Action_Q', 'Turn']).copy()
    if len(df_valid) == 0:
        print("[Warning] Not enough data for Analysis 6.")
        return

    # 2. 将数值型的胜负结果映射为类别标签
    df_valid['Outcome'] = df_valid['Action_Q'].apply(lambda x: 'Success' if x >= 0.5 else 'Fail')

    # 3. 【核心修正】：针对“共享前缀”的去重逻辑
    # 同一个 User_ID 在同一个 Turn 的 H_state 和 H_action 如果完全一样，说明是共享的前缀轨迹。
    # 加入 Outcome 共同去重，意味着：该前缀如果最终导向了成功，在成功池子里只算一次；
    # 如果同时也导向了失败（比如另一条分支失败了），在失败池子里也只算一次。
    # 彻底消除分支数量带来的权重偏误！
    if 'User_ID' in df_valid.columns:
        df_unique = df_valid.drop_duplicates(subset=['User_ID', 'Turn', 'H_state', 'H_action', 'Outcome'])
    else:
        df_unique = df_valid.drop_duplicates(subset=['Turn', 'H_state', 'H_action', 'Outcome'])

    # 4. 按 Turn 和 Outcome 分组，计算均值 (Mean)
    agg_df = df_unique.groupby(['Turn', 'Outcome']).agg(
        Mean_H_state=('H_state', 'mean'),
        Mean_H_action=('H_action', 'mean'),
        Count=('H_state', 'count') # 记录一下去重后的样本量，防长尾噪音
    ).reset_index()
    
    # 【细节优化】：过滤掉极度稀疏的后期轮次（例如由于偶然原因打到12轮，样本量不足3），防止折线图末端乱飘
    agg_df = agg_df[agg_df['Count'] >= 3]

    # 保存聚合数据供查阅
    csv_path = os.path.join(output_dir, f"{prefix}_analysis6_entropy_trend.csv" if prefix else "analysis6_entropy_trend.csv")
    agg_df.to_csv(csv_path, index=False)

    # 5. 绘制折线图
    plt.figure(figsize=(10, 6))
    
    # 定义样式和颜色
    colors = {'State': '#1f77b4', 'Action': '#ff7f0e'} # 蓝色代表状态熵，橙色代表动作熵
    styles = {'Success': '-', 'Fail': '--'}           # 实线代表成功，虚线代表失败
    marker_style = 'o'                                # 使用圆圈作为节点标记

    for outcome in ['Success', 'Fail']:
        data = agg_df[agg_df['Outcome'] == outcome].sort_values(by='Turn')
        if data.empty: continue
        
        # 绘制 State Entropy 均值
        plt.plot(data['Turn'], data['Mean_H_state'], 
                 label=f'{outcome} State Entropy', 
                 color=colors['State'], linestyle=styles[outcome], 
                 marker=marker_style, linewidth=2.5, markersize=7)
        
        # 绘制 Action Entropy 均值
        plt.plot(data['Turn'], data['Mean_H_action'], 
                 label=f'{outcome} Action Entropy', 
                 color=colors['Action'], linestyle=styles[outcome], 
                 marker=marker_style, linewidth=2.5, markersize=7)

    plt.title(f'Global Mean Entropy Trend (Prefix Deduplicated) - {prefix}', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Dialogue Turn', fontsize=13)
    plt.ylabel('Mean Entropy (H)', fontsize=13)
    
    # 强制让 X 轴只显示整数轮次
    plt.xticks(range(int(agg_df['Turn'].min()), int(agg_df['Turn'].max()) + 1))
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=11, framealpha=0.9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{prefix}_analysis6_entropy_trend.png" if prefix else "analysis6_entropy_trend.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"  -> Saved to {plot_path}")

# =====================================================================
# 模块 5：主程序入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', required=True, help="List of EXP result json files")
    parser.add_argument("--names", nargs='+', required=True, help="List of Model Names corresponding to files")
    parser.add_argument("--output_dir", default="eval_results", help="Directory to save analysis results")
    args = parser.parse_args()

    if len(args.files) != len(args.names):
        print("[Error] Number of files and names must match.")
        return
    EXPERIMENTS_CONFIG = [{"name": n, "path": p} for p, n in zip(args.files, args.names)]
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 同时生成 Node-Level 和 Action-Level 两个数据集
    df_rl_node, df_rl_action = build_rl_datasets(EXPERIMENTS_CONFIG)
    
    if df_rl_node.empty:
        print("[Error] No data parsed.")
        return

    # 保存基础数据表
    df_rl_node.to_csv(os.path.join(args.output_dir, "rl_node_dataset_v3.csv"), index=False)
    df_rl_action.to_csv(os.path.join(args.output_dir, "rl_action_dataset_v3.csv"), index=False)
    print(f"\n✅ 成功！提取节点级记录: {len(df_rl_node)} 条，动作级拆解记录: {len(df_rl_action)} 条。")

    persona_data = json.load(open("/root/EvolvingAgent-master/EvolvingAgentTest_wym/user_personas.json", 'r', encoding='utf-8'))

    # 循环每个模型进行出图
    for model in EXPERIMENTS_CONFIG:
        exp_name = model["name"]
        print(f"\n{'='*50}\n--- Starting RL Analysis For Model: {exp_name} ---\n{'='*50}")
        
        # 切片出当前模型的数据
        node_df = df_rl_node[df_rl_node["Experiment"] == exp_name]
        action_df = df_rl_action[df_rl_action["Experiment"] == exp_name]

        # 遍历两个分析维度：五大人格 vs 决策风格
        for dimension in ["Big-Five Personality", "Decision-Making Style"]:
            dim_prefix = dimension.split(' ')[0] # 'Big-Five' or 'Decision-Making'
            print(f"\n[Info] 分析维度 ---> {dimension}")
            
            # 为节点表挂载人格属性 (用于分析1和2)
            node_processed = load_and_preprocess_data(node_df, persona_dict=persona_data, target_dimension=dimension)
            
            # 为动作表挂载人格属性 (用于最新的分析3)
            action_processed = load_and_preprocess_data(action_df, persona_dict=persona_data, target_dimension=dimension)
            
            file_prefix = f"{exp_name}_{dim_prefix}"
            
            # 执行分析 1 和 2 (使用 Node-Level 数据)
            analysis_1_cognitive_signals_over_time(node_processed, args.output_dir, prefix=file_prefix)
            analysis_2_persona_difficulty_tolerance(node_processed, args.output_dir, prefix=file_prefix)
            
            # 执行修复割裂后的分析 3 (使用 Action-Level 数据)
            analysis_3_action_advantage_heatmap(action_processed, args.output_dir, prefix=file_prefix)

        analysis_4_comprehensive_factor_importance(
            df_action=action_df, 
            persona_dict=persona_data, 
            output_dir=args.output_dir, 
            prefix=f"{exp_name}_Unified",
            model_name=exp_name
        )
        analysis_5_linear_regression_correlation(
                df_action=action_df, persona_dict=persona_data, output_dir=args.output_dir, prefix=f"{exp_name}_Unified",
                model_name=exp_name
            )
        analysis_6_global_entropy_trend(action_df, args.output_dir, prefix=f"{exp_name}_Unified")
if __name__ == "__main__":
    main()