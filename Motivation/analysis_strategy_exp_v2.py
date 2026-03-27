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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
import random
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
    df = csv_data.copy()
    df['Persona'] = df['User_ID'].apply(lambda uid: persona_dict.get(uid, {}).get(target_dimension, "Unknown"))
    df['Persona'] = df['Persona'].apply(robust_persona_mapping)
    df['V_bar_persona'] = df.apply(get_v_bar, axis=1)
    
    if 'Advantage' in df.columns: df['Abs_Advantage'] = df['Advantage'].abs()
    if 'Progress' in df.columns: df['Abs_Progress'] = df['Progress'].abs()
    return df

# =====================================================================
# 模块 2：[极度强健版] Token 解析与认知耦合熵计算
# =====================================================================

def segment_metrics_new_format(metrics):
    """
    【完美修复版】基于绝对字符坐标轴的 Token 过滤算法。
    彻底解决大模型忘写闭合标签、以及 Tokenizer 切碎标签导致的空值 BUG。
    """
    tokens_text = "".join([m['token'] for m in metrics])
    
    s1 = tokens_text.find("<state_analysis>")
    e1 = tokens_text.find("</state_analysis>")
    s2 = tokens_text.find("<strategy>")
    e2 = tokens_text.find("</strategy>")
    
    # 智能纠错：如果模型忘了输出闭合标签，用下一个开标签或文本结尾兜底
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
        
        # 判断当前 Token 是否完全或部分落入合法的内容区间内
        in_state = (s1 != -1 and e1 != -1 and token_end > s1 + len("<state_analysis>") and token_start < e1)
        in_strat = (s2 != -1 and e2 != -1 and token_end > s2 + len("<strategy>") and token_start < e2)
        
        if in_state or in_strat:
            valid_tokens.append(m)
            
        current_pos += token_len
        
    return valid_tokens

def get_hc_from_turn(turn, traj_id, user_metrics):
    if not user_metrics: return None
    m_traj = user_metrics.get(traj_id)
    if not m_traj: return None
    
    r = turn.get('round')
    m_round = next((m for m in m_traj.get('metrics', []) if m.get('round') == r), None)
    if m_round and 'metrics' in m_round:
        valid_tokens = segment_metrics_new_format(m_round['metrics'])
        ents = [tok.get('entropy', 0.0) for tok in valid_tokens]
        if ents: return float(np.mean(ents))
    return None

# =====================================================================
# 模块 3：强化学习指标计算与数据提取
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
                delta_hs = hs - prev_hs if prev_hs is not None else 0.0
                delta_ha = ha - prev_ha if prev_ha is not None else 0.0
                turn_features[r] = {
                    'H_state': hs, 'H_action': ha, 'Dissonance': hs - ha,
                    'Delta_H_state': delta_hs, 'Delta_H_action': delta_ha, 'S_Trend': delta_hs - delta_ha,
                    'Sys_Strategy': sys_strategy, 'Sys_Category': SYS_STRATEGY_MAP.get(sys_strategy, "Other"),
                    'Prev_User_Strategy': last_user_strategy, 'Prev_User_Category': USER_STRATEGY_MAP.get(last_user_strategy, "Other")
                }
                prev_hs, prev_ha = hs, ha
    return turn_features

def process_single_user(user_item, exp_name):
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
        if t_branch is not None: branches_by_turn[t_branch].append(float(is_traj_success(b)))
            
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

def process_single_user_action_level(user_item, exp_name, user_metrics=None):
    user_id = user_item.get('user_id', 'unknown')
    trajectories = user_item.get('trajectories', [])
    root_traj = next((t for t in trajectories if t.get('id') == 'root'), None)
    if not root_traj: return []
        
    state_context = {}
    last_user_strategy = "None"
    last_hs, last_ha, last_hc = None, None, None
    
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
                'Prev_H_action': last_ha,
                'Prev_H_coupled': last_hc
            }
            hs_val, ha_val = turn.get('hs'), turn.get('ha')
            last_hs = float(hs_val) if hs_val is not None else None
            last_ha = float(ha_val) if ha_val is not None else None
            last_hc = get_hc_from_turn(turn, 'root', user_metrics)

    branch_trajs = [t for t in trajectories if t.get('id') != 'root']
    actions_by_turn = defaultdict(list)
    
    for turn in root_traj.get('turns', []):
        if turn.get('role') == 'Persuader':
            r = turn.get('round')
            hs_val, ha_val = turn.get('hs'), turn.get('ha')
            actions_by_turn[r].append({
                'traj_type': 'root',
                'Sys_Strategy': turn.get('strategy_name', "Unknown"),
                'Outcome_Q': float(is_traj_success(root_traj)),
                'H_state': float(hs_val) if hs_val is not None else None,
                'H_action': float(ha_val) if ha_val is not None else None,
                'H_coupled': get_hc_from_turn(turn, 'root', user_metrics)
            })
            
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
                        'H_action': float(ha_val) if ha_val is not None else None,
                        'H_coupled': get_hc_from_turn(turn, b.get('id'), user_metrics)
                    })
                    break

    action_records = []
    for t, actions in actions_by_turn.items():
        if t not in state_context: continue
        if len(actions) <= 1: continue 
            
        v_state = np.mean([a['Outcome_Q'] for a in actions])
        for a in actions:
            hs, ha, hc = a['H_state'], a['H_action'], a['H_coupled']
            prev_hs, prev_ha, prev_hc = state_context[t]['Prev_H_state'], state_context[t]['Prev_H_action'], state_context[t]['Prev_H_coupled']
            
            record = {
                'Experiment': exp_name, 'User_ID': user_id, 'Turn': t,
                'Traj_Type': a['traj_type'], 'V_state': v_state, 
                'Action_Q': a['Outcome_Q'], 'Advantage': a['Outcome_Q'] - v_state, 
                'H_state': hs, 'H_action': ha, 'H_coupled': hc,
                'Delta_H_state': (hs - prev_hs) if (hs is not None and prev_hs is not None) else 0.0,
                'Delta_H_action': (ha - prev_ha) if (ha is not None and prev_ha is not None) else 0.0,
                'Delta_H_coupled': (hc - prev_hc) if (hc is not None and prev_hc is not None) else None,
                'Dissonance': (hs - ha) if (hs is not None and ha is not None) else None,
                'Sys_Strategy': a['Sys_Strategy'], 
                'Sys_Category': SYS_STRATEGY_MAP.get(a['Sys_Strategy'], "Other"),
                'Prev_User_Strategy': state_context[t]['Prev_User_Strategy'],
                'Prev_User_Category': state_context[t]['Prev_User_Category']
            }
            action_records.append(record)
            
    return action_records

def build_rl_datasets(experiments_config):
    all_node_data, all_action_data = [], []
    for config in experiments_config:
        exp_name, filepath = config.get('name'), config.get('path')
        metrics_filepath = filepath.replace('.json', '_metrics.json')
        
        if not os.path.exists(filepath): continue
        print(f"[Info] 解析 RL 数据: {exp_name}")
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
            
        metrics_lookup = {}
        if os.path.exists(metrics_filepath):
            print(f"     [Info] Found corresponding metrics file: {metrics_filepath}")
            with open(metrics_filepath, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            for m_user in metrics_data:
                metrics_lookup[m_user.get('user_id')] = {t['id']: t for t in m_user.get('trajectories', [])}
        else:
            print(f"     [Warning] No metrics file found. H_coupled will be NaN.")

        for user_item in data:
            uid = user_item.get('user_id', 'unknown')
            all_node_data.extend(process_single_user(user_item, exp_name))
            all_action_data.extend(process_single_user_action_level(user_item, exp_name, metrics_lookup.get(uid, {})))
                
    return pd.DataFrame(all_node_data), pd.DataFrame(all_action_data)

# =====================================================================
# 模块 4：四大核心分析画图函数 (原 Analysis 1-5 保留)
# =====================================================================

def analysis_1_cognitive_signals_over_time(df, output_dir="results", prefix=None):
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
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_analysis1_temporal_signals.png"), dpi=300)
    plt.close()

def analysis_2_persona_difficulty_tolerance(df, output_dir="results", prefix=None):
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
    df_valid = df_action[(df_action['Prev_User_Category'] != 'Other') & (df_action['Sys_Category'] != 'Other')].copy()
    transition_pivot = df_valid.pivot_table(values='Advantage', index='Prev_User_Category', columns='Sys_Category', aggfunc='mean').fillna(0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_pivot, annot=True, cmap='RdYlGn', center=0, fmt=".3f", linewidths=.5, cbar_kws={'label': 'Mean Action Advantage ($A_t$)'})
    plt.title('Causal Map: Action Strategy Impact on Expected Advantage', fontsize=16, pad=15)
    plt.xlabel('Current Specific System Action ($a_t$ category)', fontsize=12)
    plt.ylabel('Previous User State ($S_{user, t-1}$)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_analysis3_action_advantage.png"), dpi=300)
    plt.close()

def analysis_4_comprehensive_factor_importance(df_action, persona_dict, output_dir="results", prefix=None, model_name=None):
    df_valid = df_action.dropna(subset=['H_state', 'H_action', 'Advantage', 'Delta_H_state']).copy()
    if len(df_valid) < 20: return
    df_valid['Big_Five_Label'] = df_valid['User_ID'].apply(lambda uid: robust_persona_mapping(persona_dict.get(uid, {}).get("Big-Five Personality", "Unknown")))
    df_valid['Decision_Making_Label'] = df_valid['User_ID'].apply(lambda uid: robust_persona_mapping(persona_dict.get(uid, {}).get("Decision-Making Style", "Unknown")))

    df_valid['V_bar_BigFive'] = df_valid.apply(lambda row: get_v_bar({'Experiment': row['Experiment'], 'Persona': row['Big_Five_Label']}), axis=1)
    df_valid['V_bar_DecisionMaking'] = df_valid.apply(lambda row: get_v_bar({'Experiment': row['Experiment'], 'Persona': row['Decision_Making_Label']}), axis=1)

    max_turn = df_valid['Turn'].max() if df_valid['Turn'].max() > 0 else 10
    df_valid['Temporal_Weight'] = 0.5 + (df_valid['Turn'] / max_turn)
    df_valid['Temporal_Advantage'] = df_valid['Advantage'] * df_valid['Temporal_Weight']

    features_num = ['Turn', 'H_state', 'H_action', 'Delta_H_state', 'Delta_H_action', 'Dissonance', 'V_bar_BigFive', 'V_bar_DecisionMaking']
    features_cat = ['Sys_Strategy', 'Sys_Category', 'Prev_User_Strategy', 'Prev_User_Category']
    X = pd.get_dummies(df_valid[features_num + features_cat], columns=features_cat)
    
    rf_orig = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10).fit(X, df_valid['Advantage'])
    rf_temp = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10).fit(X, df_valid['Temporal_Advantage'])

    tree_estimator = rf_temp.estimators_[0]
    plt.figure(figsize=(24, 12))
    plot_tree(tree_estimator, feature_names=X.columns, filled=True, rounded=True, max_depth=3, fontsize=10, proportion=True)
    plt.title(f'Sample Decision Tree Logic from Random Forest ({prefix} - Temporal Curriculum)', fontsize=18, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_analysis4_sample_decision_tree.png"), dpi=300)
    plt.close()

def analysis_5_linear_regression_correlation(df_action, persona_dict, output_dir="results", prefix=None, model_name=None):
    df_valid = df_action.dropna(subset=['H_state', 'H_action', 'Advantage', 'Delta_H_state']).copy()
    if len(df_valid) < 20: return
    df_valid['Big_Five_Label'] = df_valid['User_ID'].apply(lambda uid: robust_persona_mapping(persona_dict.get(uid, {}).get("Big-Five Personality", "Unknown")))
    df_valid['Decision_Making_Label'] = df_valid['User_ID'].apply(lambda uid: robust_persona_mapping(persona_dict.get(uid, {}).get("Decision-Making Style", "Unknown")))

    df_valid['V_bar_BigFive'] = df_valid.apply(lambda row: get_v_bar({'Experiment': row['Experiment'], 'Persona': row['Big_Five_Label']}), axis=1)
    df_valid['V_bar_DecisionMaking'] = df_valid.apply(lambda row: get_v_bar({'Experiment': row['Experiment'], 'Persona': row['Decision_Making_Label']}), axis=1)

    max_turn = df_valid['Turn'].max() if df_valid['Turn'].max() > 0 else 10
    df_valid['Temporal_Weight'] = 0.5 + (df_valid['Turn'] / max_turn)
    df_valid['Temporal_Advantage'] = df_valid['Advantage'] * df_valid['Temporal_Weight']

    features_num = ['Turn', 'H_state', 'H_action', 'Delta_H_state', 'Delta_H_action', 'V_bar_BigFive', 'V_bar_DecisionMaking']
    features_cat = ['Sys_Category', 'Prev_User_Category']
    
    scaler = StandardScaler()
    df_valid[features_num] = scaler.fit_transform(df_valid[features_num])
    X = pd.get_dummies(df_valid[features_num + features_cat], columns=features_cat)
    
    model_orig = Ridge(alpha=1.0, random_state=42).fit(X, df_valid['Advantage'])
    model_temp = Ridge(alpha=1.0, random_state=42).fit(X, df_valid['Temporal_Advantage'])

    clean_names = [name.replace('Sys_Category_', 'Sys: ').replace('Prev_User_Category_', 'User: ') for name in X.columns]
    df_coef = pd.DataFrame({
        'Feature': clean_names * 2,
        'Coefficient': list(model_orig.coef_) + list(model_temp.coef_),
        'Reward_Type': ['Original Advantage'] * len(clean_names) + ['Temporal Curriculum Advantage'] * len(clean_names)
    })

    df_coef = df_coef[df_coef['Coefficient'].abs() >= 0.0].copy()
    df_coef.to_csv(os.path.join(output_dir, f"{prefix}_analysis5_linear_coefficients.csv"), index=False)

# =====================================================================
# 模块 5：[需求 1 满足] 全局熵值时序演变 (三线同框：Hs vs Ha vs Hc)
# =====================================================================

def analysis_6_global_entropy_trend(df_action, output_dir="results", prefix=None):
    """
    【升级版】将耦合趋势(Hc)与动作熵(Ha)、状态熵(Hs)绘制在同一张全局均值图中。
    完美继承前缀去重功能，确保同源平行分支的历史记录不被重复计算！
    """
    print(f"\n--- Running Analysis 6: Global Mean Entropy Trend (Hs vs Ha vs Hc) ({prefix}) ---")
    
    df_valid = df_action.dropna(subset=['H_state', 'H_action', 'Action_Q', 'Turn']).copy()
    if len(df_valid) == 0: return

    df_valid['Outcome'] = df_valid['Action_Q'].apply(lambda x: 'Success' if x >= 0.5 else 'Fail')
    
    # 极度严谨的前缀去重
    if 'User_ID' in df_valid.columns:
        df_unique = df_valid.drop_duplicates(subset=['User_ID', 'Turn', 'H_state', 'H_action', 'Outcome'])
    else:
        df_unique = df_valid.drop_duplicates(subset=['Turn', 'H_state', 'H_action', 'Outcome'])

    agg_dict = {'Mean_H_state': ('H_state', 'mean'), 'Mean_H_action': ('H_action', 'mean')}
    if 'H_coupled' in df_unique.columns and df_unique['H_coupled'].notna().any():
        agg_dict['Mean_H_coupled'] = ('H_coupled', 'mean')
        
    agg_df = df_unique.groupby(['Turn', 'Outcome']).agg(**agg_dict).reset_index()
    agg_df.to_csv(os.path.join(output_dir, f"{prefix}_analysis6_entropy_trend.csv"), index=False)

    # 绘制三线合一宏观图
    plt.figure(figsize=(12, 7))
    colors = {'State': '#2ecc71', 'Action': '#3498db', 'Coupled': '#9b59b6'} 
    styles = {'Success': '-', 'Fail': '--'}
    marker_style = 'o'

    for outcome in ['Success', 'Fail']:
        data = agg_df[agg_df['Outcome'] == outcome].sort_values(by='Turn')
        if data.empty: continue
        
        plt.plot(data['Turn'], data['Mean_H_state'], label=f'{outcome} $H_s$ (State)', color=colors['State'], linestyle=styles[outcome], marker=marker_style, linewidth=2.5)
        plt.plot(data['Turn'], data['Mean_H_action'], label=f'{outcome} $H_a$ (Action)', color=colors['Action'], linestyle=styles[outcome], marker=marker_style, linewidth=2.5)
        
        # 额外绘制紫色的 H_coupled
        if 'Mean_H_coupled' in data.columns:
            plt.plot(data['Turn'], data['Mean_H_coupled'], label=f'{outcome} $H_c$ (Coupled)', color=colors['Coupled'], linestyle=styles[outcome], marker='D', linewidth=2.5)

    plt.title(f'Global Mean Entropy Trend ($H_s$ vs $H_a$ vs $H_c$) - {prefix}', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Dialogue Turn', fontsize=13)
    plt.ylabel('Mean Entropy (H)', fontsize=13)
    plt.xticks(range(int(agg_df['Turn'].min()), int(agg_df['Turn'].max()) + 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', fontsize=11, framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{prefix}_analysis6_entropy_trend.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  -> Saved to {prefix}_analysis6_entropy_trend.png")

# =====================================================================
# 模块 6：[需求 2 满足] 个体平行宇宙轨迹追踪 (三轨展示，不缩减！)
# =====================================================================

def get_full_turns(traj, root_traj, alpha=0.35, metrics_traj=None, root_metrics_traj=None):
    turns = []
    t_branch = traj.get('branch_at_turn', float('inf')) if traj.get('id') != 'root' else float('inf')
    
    def get_hc_for_turn(m_traj, r):
        if m_traj:
            m_round = next((m for m in m_traj.get('metrics', []) if m.get('round') == r), None)
            if m_round and 'metrics' in m_round:
                # 使用刚刚修正过的绝对安全过滤算法
                valid_tokens = segment_metrics_new_format(m_round['metrics'])
                ents = [tok.get('entropy', 0.0) for tok in valid_tokens]
                if ents: return float(np.mean(ents))
        return None

    current_round = 1
    for t in root_traj.get('turns', []):
        if t.get('round') is not None: current_round = t.get('round')
        if current_round < t_branch and t.get('role') == 'Persuader':
            hs, ha = t.get('hs'), t.get('ha')
            hc = get_hc_for_turn(root_metrics_traj, current_round)
            if hs is not None and ha is not None:
                turns.append({'Turn': current_round, 'Hs': float(hs), 'Ha': float(ha), 'Hc': hc})
                
    current_round = 1 
    for t in traj.get('turns', []):
        if t.get('round') is not None: current_round = t.get('round')
        if current_round >= t_branch and t.get('role') == 'Persuader':
            hs, ha = t.get('hs'), t.get('ha')
            hc = get_hc_for_turn(metrics_traj, current_round)
            if hs is not None and ha is not None:
                turns.append({'Turn': current_round, 'Hs': float(hs), 'Ha': float(ha), 'Hc': hc})
                
    df = pd.DataFrame(turns)
    if not df.empty:
        df['Hs_EMA'] = df['Hs'].ewm(alpha=alpha, adjust=False).mean()
        df['Ha_EMA'] = df['Ha'].ewm(alpha=alpha, adjust=False).mean()
        if 'Hc' in df.columns and df['Hc'].notna().any():
            df['Hc_EMA'] = df['Hc'].ewm(alpha=alpha, adjust=False).mean()
        else:
            df['Hc_EMA'] = np.nan
    return df

def analysis_7_individual_parallel_universe(filepath, metrics_filepath, output_dir, prefix):
    """
    【升级且不缩减行数】随机抽取 5 名用户，将成败分支平行的历史画在严谨的 3 行图表中！
    Row 1: 状态熵 | Row 2: 动作熵 | Row 3: 耦合熵
    """
    print("  -> Running Analysis 7: Individual Parallel Universe Tracking (Strictly 3 Rows!)")
    if not os.path.exists(filepath): return
        
    with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        
    metrics_lookup = {}
    if os.path.exists(metrics_filepath):
        with open(metrics_filepath, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
        for m_user in metrics_data:
            metrics_lookup[m_user.get('user_id')] = {t['id']: t for t in m_user.get('trajectories', [])}
        
    pairs = []
    for user in data:
        uid = user.get('user_id', 'Unknown')
        trajs = user.get('trajectories', [])
        root_traj = next((t for t in trajs if t.get('id') == 'root'), None)
        if not root_traj: continue
        
        user_metrics = metrics_lookup.get(uid, {})
        root_metrics = user_metrics.get('root')
        root_success = is_traj_success(root_traj)
        
        valid_pair = None
        for b in trajs:
            if b.get('id') == 'root': continue
            b_success = is_traj_success(b)
            if b_success != root_success:
                t_branch = b.get('branch_at_turn')
                b_metrics = user_metrics.get(b.get('id'))
                
                valid_pair = {
                    'User_ID': str(uid)[:8], 't_branch': t_branch,
                    'Succ_DF': get_full_turns(root_traj if root_success else b, root_traj, metrics_traj=root_metrics if root_success else b_metrics, root_metrics_traj=root_metrics),
                    'Fail_DF': get_full_turns(b if root_success else root_traj, root_traj, metrics_traj=b_metrics if root_success else root_metrics, root_metrics_traj=root_metrics)
                }
                break 
                
        if valid_pair: pairs.append(valid_pair)
            
    random.seed(42)
    if len(pairs) > 5: pairs = random.sample(pairs, 5)
    elif len(pairs) == 0: return
        
    fig, axes = plt.subplots(3, len(pairs), figsize=(5 * len(pairs), 15), sharex=False, sharey='row')
    if len(pairs) == 1: axes = np.array([axes]).T 
    
    for i, pair in enumerate(pairs):
        user_id, t_branch, succ_df, fail_df = pair['User_ID'], pair['t_branch'], pair['Succ_DF'], pair['Fail_DF']
        
        # Row 0: Hs
        ax_hs = axes[0, i]
        ax_hs.plot(succ_df['Turn'], succ_df['Hs'], color='#2ecc71', alpha=0.3, linestyle='-', marker='.')
        ax_hs.plot(fail_df['Turn'], fail_df['Hs'], color='#e74c3c', alpha=0.3, linestyle='-', marker='.')
        ax_hs.plot(succ_df['Turn'], succ_df['Hs_EMA'], label='Success Traj (EMA)', color='#2ecc71', marker='o', linewidth=3, markersize=8)
        ax_hs.plot(fail_df['Turn'], fail_df['Hs_EMA'], label='Fail Traj (EMA)', color='#e74c3c', marker='X', linestyle='--', linewidth=3, markersize=8)
        ax_hs.axvline(t_branch, color='gray', linestyle=':', label=f'Divergence Node (T={t_branch})')
        ax_hs.set_title(f"User ID: {user_id}", fontsize=15, fontweight='bold', pad=10)
        if i == 0: ax_hs.set_ylabel("State Entropy ($H_s$)", fontsize=14)
        ax_hs.grid(True, linestyle='--', alpha=0.5)
        if i == 0: ax_hs.legend(loc='upper left', fontsize=10)
        
        # Row 1: Ha
        ax_ha = axes[1, i]
        ax_ha.plot(succ_df['Turn'], succ_df['Ha'], color='#3498db', alpha=0.3, linestyle='-', marker='.')
        ax_ha.plot(fail_df['Turn'], fail_df['Ha'], color='#e74c3c', alpha=0.3, linestyle='-', marker='.')
        ax_ha.plot(succ_df['Turn'], succ_df['Ha_EMA'], color='#3498db', marker='o', linewidth=3, markersize=8)
        ax_ha.plot(fail_df['Turn'], fail_df['Ha_EMA'], color='#e74c3c', marker='X', linestyle='--', linewidth=3, markersize=8)
        ax_ha.axvline(t_branch, color='gray', linestyle=':')
        if i == 0: ax_ha.set_ylabel("Action Entropy ($H_a$)", fontsize=14)
        ax_ha.grid(True, linestyle='--', alpha=0.5)

        # Row 2: Hc (Coupled Entropy)
        ax_hc = axes[2, i]
        if 'Hc' in succ_df.columns and succ_df['Hc'].notna().any():
            ax_hc.plot(succ_df['Turn'], succ_df['Hc'], color='#9b59b6', alpha=0.3, linestyle='-', marker='.')
            ax_hc.plot(fail_df['Turn'], fail_df['Hc'], color='#e67e22', alpha=0.3, linestyle='-', marker='.')
            ax_hc.plot(succ_df['Turn'], succ_df['Hc_EMA'], label='Success Traj ($H_c$ EMA)', color='#9b59b6', marker='o', linewidth=3, markersize=8)
            ax_hc.plot(fail_df['Turn'], fail_df['Hc_EMA'], label='Fail Traj ($H_c$ EMA)', color='#e67e22', marker='X', linestyle='--', linewidth=3, markersize=8)
        
        ax_hc.axvline(t_branch, color='gray', linestyle=':')
        ax_hc.set_xlabel("Dialogue Turn", fontsize=12)
        if i == 0: ax_hc.set_ylabel("Coupled Entropy ($H_c$)", fontsize=14)
        ax_hc.grid(True, linestyle='--', alpha=0.5)
        if i == 0: ax_hc.legend(loc='upper left', fontsize=10)
        
        max_turn = max(succ_df['Turn'].max() if not succ_df.empty else 0, fail_df['Turn'].max() if not fail_df.empty else 0)
        ax_hs.set_xticks(range(1, int(max_turn)+2))
        ax_ha.set_xticks(range(1, int(max_turn)+2))
        ax_hc.set_xticks(range(1, int(max_turn)+2))

    plt.suptitle(f"Analysis 7: Parallel Universe Tracking (State vs Action vs Coupled) with EMA - {prefix}", fontsize=22, y=1.02)
    plt.tight_layout()
    output_filepath = os.path.join(output_dir, f"{prefix}_analysis7_parallel_universe.png")
    plt.savefig(output_filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     ✅ Saved Individual User Trajectory Tracking (3 Rows) to: {output_filepath}")

# =====================================================================
# 模块 7：主程序入口
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

    df_rl_node, df_rl_action = build_rl_datasets(EXPERIMENTS_CONFIG)
    
    if df_rl_node.empty:
        print("[Error] No data parsed.")
        return

    df_rl_node.to_csv(os.path.join(args.output_dir, "rl_node_dataset_v3.csv"), index=False)
    df_rl_action.to_csv(os.path.join(args.output_dir, "rl_action_dataset_v3.csv"), index=False)
    print(f"\n✅ 成功！提取节点级记录: {len(df_rl_node)} 条，动作级拆解记录: {len(df_rl_action)} 条。")

    try:
        persona_data = json.load(open("/root/EvolvingAgent-master/EvolvingAgentTest_wym/user_personas.json", 'r', encoding='utf-8'))
    except FileNotFoundError:
        print("[Warning] Persona file not found. Persona-related analysis might fail or show 'Unknown'.")
        persona_data = {}

    for config in EXPERIMENTS_CONFIG:
        exp_name = config["name"]
        filepath = config["path"]
        metrics_filepath = filepath.replace('.json', '_metrics.json')
        print(f"\n{'='*50}\n--- Starting RL Analysis For Model: {exp_name} ---\n{'='*50}")
        
        node_df = df_rl_node[df_rl_node["Experiment"] == exp_name]
        action_df = df_rl_action[df_rl_action["Experiment"] == exp_name]

        for dimension in ["Big-Five Personality", "Decision-Making Style"]:
            dim_prefix = dimension.split(' ')[0] 
            print(f"\n[Info] 分析维度 ---> {dimension}")
            
            node_processed = load_and_preprocess_data(node_df, persona_dict=persona_data, target_dimension=dimension)
            action_processed = load_and_preprocess_data(action_df, persona_dict=persona_data, target_dimension=dimension)
            
            file_prefix = f"{exp_name}_{dim_prefix}"
            
            analysis_1_cognitive_signals_over_time(node_processed, args.output_dir, prefix=file_prefix)
            analysis_2_persona_difficulty_tolerance(node_processed, args.output_dir, prefix=file_prefix)
            analysis_3_action_advantage_heatmap(action_processed, args.output_dir, prefix=file_prefix)

        analysis_4_comprehensive_factor_importance(df_action=action_df, persona_dict=persona_data, output_dir=args.output_dir, prefix=f"{exp_name}_Unified", model_name=exp_name)
        analysis_5_linear_regression_correlation(df_action=action_df, persona_dict=persona_data, output_dir=args.output_dir, prefix=f"{exp_name}_Unified", model_name=exp_name)
        
        # 核心绘制：三线同框宏观全局趋势图
        analysis_6_global_entropy_trend(action_df, args.output_dir, prefix=f"{exp_name}_Unified")
        
        # 核心绘制：不缩减！三行展示的个体平行宇宙追踪图
        analysis_7_individual_parallel_universe(filepath, metrics_filepath, args.output_dir, prefix=f"{exp_name}_Unified")

if __name__ == "__main__":
    main()