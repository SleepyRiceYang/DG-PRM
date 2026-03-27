import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 1. 启发式奖励公式的核心组件
# =====================================================================
def semantic_gating(prev_user_cat, sys_cat):
    """
    第一层：策略上下文安全门控 (Semantic Gating)
    """
    if prev_user_cat == 'Defensive' and sys_cat == 'Action Facilitation':
        return -1 # 用户防御时强行逼单 -> 死局
    if prev_user_cat == 'Offensive' and sys_cat in ['Core Persuasion', 'Action Facilitation']:
        return -1 # 用户攻击时强硬说服 -> 死局
    return 1

def calculate_heuristic(row, params):
    """
    计算单步的三维启发式信用得分 (修正版：引入环境基线 Outcome)
    """
    turn = row['Turn']
    h_state = row['H_state']
    h_action = row['H_action']
    user_cat = row['Prev_User_Category']
    sys_cat = row['Sys_Category']
    node_type = row['Node_Type'] # 【新增】必须知道这条轨迹最终是成是败！
    
    # 解析超参数
    theta_s, theta_a = params['theta_s'], params['theta_a']
    r_heavy_pos = params['r_heavy_pos']   # 重磅奖励 (神仙手)
    r_light_pos = params['r_light_pos']   # 轻微奖励 (平庸成功)
    p_light_neg = params['p_light_neg']   # 轻微惩罚 (值得探索的失败)
    p_heavy_neg = params['p_heavy_neg']   # 重度惩罚 (盲目自信的失败)
    lambda_fatal = params['lambda_fatal'] # 语义致命拦截
    
    # 1. 门控判定
    mask = semantic_gating(user_cat, sys_cat)
    if mask == -1:
        return lambda_fatal # 烂招直接返回极负的分数

    # 2. 结合【最终结果】与【认知象限】进行判定
    if 'Failed' in node_type:
        # 失败轨迹基调：惩罚
        if h_state < theta_s and h_action >= theta_a:
            # Q2 (高挽救率)：知道自己不懂，还在试探。值得宽容 -> 轻微惩罚
            base_reward = p_light_neg 
        else:
            # Q1/Q4 (低挽救率)：烂泥扶不上墙，或者盲目自信 -> 重度惩罚
            base_reward = p_heavy_neg 
    else:
        # 成功轨迹基调：奖励
        if h_state < theta_s and h_action < theta_a:
            # Q3 (高脆弱率/神仙手)：看清局势，果断出手 -> 重磅奖励
            base_reward = r_heavy_pos 
        else:
            # 其他 (低脆弱率)：平庸的成功，赢了但也随时可被替换 -> 轻微奖励
            base_reward = r_light_pos

    # 3. 时序动态加权
    max_turns = 10.0
    temporal_weight = 0.5 + (turn / max_turns)
    
    return base_reward * temporal_weight

# =====================================================================
# 2. 网格搜索 (针对单个模型/实验独立搜索)
# =====================================================================

def perform_grid_search_for_exp(exp_name, exp_df):
    """
    为特定的模型独立搜索最优超参数
    """
    print(f"\n---> 开始为 [{exp_name}] 搜索最优启发式参数...")
    
    search_space = {
        'theta_s': [0.1, 0.2, 0.3, 0.4, 0.5],
        'theta_a': [0.1, 0.2, 0.3, 0.4, 0.5],
        'r_heavy_pos': [0.8, 1.0, 1.2, 1.5],       # Q3 奖励
        'r_light_pos': [0.2, 0.4, 0.5, 0.8],       # 其他成功奖励
        'p_light_neg': [-0.2, -0.4, -0.5, -0.8],    # Q2 惩罚
        'p_heavy_neg': [-0.8, -1.0, -1.2, -1.5],    # 其他失败惩罚
        'lambda_fatal': [-1.5, -2.0]
    }
    
    keys, values = zip(*search_space.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_corr = -1.0
    best_params = None
    best_p_value = 1.0
    
    y_true = exp_df['Credit_Score'].values
    total_iters = len(param_combinations)
    
    for i, params in enumerate(param_combinations):
        y_heuristic = exp_df.apply(lambda row: calculate_heuristic(row, params), axis=1).values
        # 如果启发式分数全是一样的(方差为0)，spearmanr会报警，做个容错
        if len(np.unique(y_heuristic)) <= 1:
            continue
            
        corr, p_val = spearmanr(y_true, y_heuristic)
        
        if corr > best_corr and p_val < 0.05:
            best_corr = corr
            best_params = params
            best_p_value = p_val
            
    if best_params:
        print(f"  [成功] 找到最优 Spearman 相关系数: {best_corr:.4f} (p-value: {best_p_value:.4e})")
        print("  最优参数组合:")
        for k, v in best_params.items():
            print(f"    - {k}: {v}")
    else:
        print(f"  [警告] 未能为 [{exp_name}] 找到统计显著的正相关参数。")
        
    return best_params, best_corr

# =====================================================================
# 3. 可视化并生成报告 (每个模型独立一张图)
# =====================================================================

def plot_validation_results(exp_name, exp_df, best_params, output_dir="eval_results"):
    if best_params is None:
        return
        
    exp_df = exp_df.copy()
    exp_df['Heuristic_Score'] = exp_df.apply(lambda row: calculate_heuristic(row, best_params), axis=1)
    
    plt.figure(figsize=(9, 7))
    sns.regplot(
        data=exp_df, 
        x='Credit_Score', 
        y='Heuristic_Score', 
        scatter_kws={'alpha': 0.6, 'edgecolor': 'w', 's': 60, 'color': 'teal'},
        line_kws={'color': 'tomato', 'linewidth': 2, 'linestyle': '--'}
    )
    
    corr_p, _ = pearsonr(exp_df['Credit_Score'], exp_df['Heuristic_Score'])
    corr_s, _ = spearmanr(exp_df['Credit_Score'], exp_df['Heuristic_Score'])
    
    plt.title(f'[{exp_name}] Heuristic Reward vs. Ground Truth Score\n(Spearman $\\rho$={corr_s:.2f}, Pearson $r$={corr_p:.2f})', fontsize=14)
    plt.xlabel('Ground Truth Credit Score (MC Rescue/Vulnerability)', fontsize=12)
    plt.ylabel('Proposed Heuristic Score', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 强制固定坐标轴范围，方便不同模型之间的视觉对比
    plt.xlim(-1.1, 1.1)
    # y 轴可以根据 Heuristic_Score 的极端值自适应，或者也限制在 -2 到 2
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{exp_name}_heuristic_validation.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  --> [{exp_name}] 的验证散点图已保存至: {save_path}")

# =====================================================================
# 4. 核心调用接口 (被 base_analysis.py 调用)
# =====================================================================

def run_validation(df, output_dir="eval_results"):
    """
    供 base_analysis.py 调用的入口函数。
    会按 Experiment (模型名称) 进行分组，分开进行验证。
    """
    print("\n========== [Phase 3] 开始启发式奖励公式的独立验证与搜索 ==========")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    req_cols = ['Experiment', 'Turn', 'H_state', 'H_action', 'Prev_User_Category', 'Sys_Category', 'Credit_Score']
    
    missing_cols = [col for col in req_cols if col not in df.columns]
    if missing_cols:
        print(f"[Error] 数据缺失关键列: {missing_cols}，验证中止。")
        return
        
    valid_df = df.dropna(subset=req_cols).copy()
    
    # 提取所有包含的实验模型名称
    experiments = valid_df['Experiment'].unique()
    print(f"检测到 {len(experiments)} 个独立的模型数据，将分别进行超参数搜索。")
    
    all_results = {}
    
    # 循环遍历每一个模型，分开进行评估
    for exp in experiments:
        exp_df = valid_df[valid_df['Experiment'] == exp].copy()
        
        if len(exp_df) < 10:
            print(f"\n[Error] 模型 [{exp}] 的有效数据量过少({len(exp_df)}条)，跳过验证。")
            continue
            
        best_params, best_corr = perform_grid_search_for_exp(exp, exp_df)
        plot_validation_results(exp, exp_df, best_params, output_dir)
        
        all_results[exp] = {
            'correlation': best_corr,
            'params': best_params
        }
        
    print("\n========== [Phase 3 总结] ==========")
    for exp, res in all_results.items():
        corr = res['correlation']
        if corr > 0.4:
            print(f"✅ [{exp}]: 验证成功 (Spearman = {corr:.2f})")
        else:
            print(f"⚠️ [{exp}]: 相关性偏弱 (Spearman = {corr:.2f})")