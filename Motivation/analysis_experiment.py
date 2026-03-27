import os
import json
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from analysis_utils import validate_heuristic

def build_reward_prediction_model(df, output_dir="plots"):
    """
    建立数学模型：通过熵、策略属性、轮次来预测启发式信用分数 (Credit Score)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if df.empty or 'Credit_Score' not in df.columns:
        print("[Modeling] 缺少数据或 Credit_Score 未计算，跳过模型训练。")
        return

    # 为了防止不同实验的数据混杂导致模型困惑，我们为每个 Experiment 单独训练一个模型
    experiments = df['Experiment'].unique()
    
    for exp in experiments:
        print(f"\n========== 训练奖励预测模型: {exp} ==========")
        exp_data = df[df['Experiment'] == exp].copy()
        
        # 1. 特征选择与数据预处理
        # 提取相关列
        features_num = ['Turn', 'H_state', 'H_action']
        features_cat = ['Sys_Category', 'Prev_User_Category']
        
        # 去除缺失值
        ml_data = exp_data[features_num + features_cat + ['Credit_Score']].dropna()
        
        if len(ml_data) < 20:
            print(f"[{exp}] 数据量过少 ({len(ml_data)} 条)，跳过模型训练。")
            continue
            
        # 对类别特征进行 One-Hot 编码 (独热编码)
        X = pd.get_dummies(ml_data[features_num + features_cat], columns=features_cat, drop_first=False)
        y = ml_data['Credit_Score']
        
        # 2. 划分训练集和测试集 (80% 训练，20% 测试)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. 初始化并训练随机森林回归模型
        # 选择 RandomForest 因为它对超参数不敏感，且能有效捕获非线性交互(如象限特征)
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # 4. 模型评估
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"[{exp}] 模型评估结果:")
        print(f"  - 均方误差 (MSE): {mse:.4f}")
        print(f"  - R^2 Score: {r2:.4f} (越接近1说明模型解释力越强)")
        
        # =====================================================================
        # 图 7. 特征重要性条形图 (Feature Importance)
        # 目的: 告诉我们到底是熵重要，还是策略上下文重要，还是轮次重要？
        # =====================================================================
        importances = model.feature_importances_
        feature_names = X.columns
        
        # 将重要性排序
        imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False).head(15) # 只看前15个最重要的特征
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=imp_df, x='Importance', y='Feature', palette='viridis')
        plt.title(f'[{exp}] Feature Importance for Credit Score Prediction', fontsize=14)
        plt.xlabel('Relative Importance', fontsize=12)
        plt.ylabel('Features (Entropy, Turn, Strategies)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{exp}_feature_importance.png'), dpi=300)
        plt.close()
        
        # =====================================================================
        # 图 8. 真实值 vs 预测值散点图 (Actual vs Predicted Credit Score)
        # 目的: 直观展示模型是否准确捕捉了奖励得分的分布趋势
        # =====================================================================
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', s=60, color='dodgerblue')
        
        # 绘制完美的 y=x 对角线
        plt.plot([-1, 1], [-1, 1], 'r--', lw=2)
        
        plt.title(f'[{exp}] Actual vs Predicted Credit Score\n($R^2={r2:.2f}, MSE={mse:.2f}$)', fontsize=14)
        plt.xlabel('Actual Credit Score (From Branching Data)', fontsize=12)
        plt.ylabel('Predicted Credit Score (From ML Model)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{exp}_actual_vs_predicted.png'), dpi=300)
        plt.close()
        
    print("\n[Modeling Complete] 奖励预测模型训练与分析图表已生成。")

# ==================== 1. 基础函数与分类字典 ====================

def is_traj_success(trajectory):
    """
    判断单条轨迹是否成功 (参考 eval_offline_branch_v2.py)
    """
    if not trajectory:
        return False
    # 1. 优先检查顶层 'success' 字段
    if trajectory.get('success'):
        return True
    
    # 2. 如果没有顶层字段，检查 'turns' 中是否有任何一轮的 reward >= 1.0
    turns = trajectory.get('turns', [])
    if not turns:
        return False
    for turn in turns:
        if turn.get('reward', 0.0) >= 1.0:
            return True
            
    if turns[-1].get('reward', 0.0) >= 1.0:
        return True

    return False

# 系统（Persuader）策略属性大类映射
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

# 用户（Persuadee）抵抗策略属性大类映射
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

# ==================== 2. 数据加载与解析 ====================

def process_experiments(experiments_config):
    """
    根据配置列表加载指定的实验文件，提取熵值、策略与分叉结果
    :param experiments_config: list of dict, e.g., [{'name': 'exp1', 'path': 'file1.json'}, ...]
    """
    all_records = []
    
    for config in experiments_config:
        exp_name = config.get('name')
        filepath = config.get('path')
        
        if not os.path.exists(filepath):
            print(f"[Warning] 文件不存在，跳过: {filepath}")
            continue
            
        print(f"[Info] 正在处理实验: {exp_name} | 文件: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for user_item in data:
            user_id = user_item.get('user_id', 'unknown')
            trajectories = user_item.get('trajectories', [])
            
            # 找到初始轨迹 (root)
            root_traj = next((t for t in trajectories if t.get('id') == 'root'), None)
            if not root_traj:
                continue
                
            root_success = is_traj_success(root_traj)
            
            # 解析 root 轨迹中的轮次级数据 (熵值和策略)
            # 将 sys_turn 的熵和策略，与 user_turn(t-1) 的策略对齐
            root_turn_data = {}
            turns = root_traj.get('turns', [])
            
            last_user_strategy = "None"
            for turn in turns:
                r = turn.get('round')
                role = turn.get('role')
                
                if role == 'Persuadee':
                    last_user_strategy = turn.get('user_strategy', "None")
                elif role == 'Persuader':
                    hs = turn.get('hs')
                    ha = turn.get('ha')
                    sys_strategy = turn.get('strategy_name', "Unknown")
                    
                    # 只有包含有效熵值的轮次才记录 (通常避开无熵值的第一轮)
                    if hs is not None and ha is not None:
                        root_turn_data[r] = {
                            'hs': float(hs),
                            'ha': float(ha),
                            'sys_strategy': sys_strategy,
                            'sys_category': SYS_STRATEGY_MAP.get(sys_strategy, "Other"),
                            'user_strategy_t_minus_1': last_user_strategy,
                            'user_category_t_minus_1': USER_STRATEGY_MAP.get(last_user_strategy, "Other")
                        }
            
            # 解析分支轨迹 (branch)，计算每个分支节点的成功率
            branch_trajs = [t for t in trajectories if t.get('id') != 'root']
            turn_outcomes = defaultdict(list)
            for b in branch_trajs:
                b_turn = b.get('branch_at_turn')
                if b_turn is not None:
                    turn_outcomes[b_turn].append(is_traj_success(b))
            
            # 计算 Criticality 并整合记录
            for t_branch, outcomes in turn_outcomes.items():
                if t_branch not in root_turn_data:
                    continue # 如果 root 轨迹中该轮次无熵值数据，则跳过
                    
                succ_rate = np.mean(outcomes)
                
                # 定义 Criticality (关键性得分)
                if not root_success:
                    criticality = succ_rate # 原始失败 -> 改变后成功的概率 = 挽救潜力 (Rescue Potential)
                    node_type = 'Failed Root (Rescue Potential)'
                else:
                    criticality = 1.0 - succ_rate # 原始成功 -> 改变后失败的概率 = 脆弱性 (Vulnerability)
                    node_type = 'Successful Root (Vulnerability)'
                
                record = {
                    'Experiment': exp_name,
                    'User_ID': user_id,
                    'Turn': t_branch,
                    'Node_Type': node_type,
                    'Criticality': criticality,
                    'H_state': root_turn_data[t_branch]['hs'],
                    'H_action': root_turn_data[t_branch]['ha'],
                    'Sys_Strategy': root_turn_data[t_branch]['sys_strategy'],
                    'Sys_Category': root_turn_data[t_branch]['sys_category'],
                    'Prev_User_Strategy': root_turn_data[t_branch]['user_strategy_t_minus_1'],
                    'Prev_User_Category': root_turn_data[t_branch]['user_category_t_minus_1']
                }
                all_records.append(record)
                
    return pd.DataFrame(all_records)

#   3. 绘制关键节点象限分析散点图
def plot_rescue_potential_quadrants(df, output_dir="plots", outlier_bounds=(0.05, 0.95)):
    """
    图 1：绘制关键节点象限分析散点图
    新增特性：为每个模型绘制独立十字原点时，支持通过分位数参数排除异常极端的熵值。
    :param df: 包含实验数据的 DataFrame
    :param output_dir: 图表输出目录
    :param outlier_bounds: tuple, (下限分位数, 上限分位数)，默认过滤掉最低5%和最高5%的异常值
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    failed_roots = df[df['Node_Type'] == 'Failed Root (Rescue Potential)']
    if failed_roots.empty:
        print("[Plot 1] 无 Failed Root 数据，跳过象限图绘制。")
        return

    plt.figure(figsize=(14, 9))
    
    # 绘制散点本身时不进行过滤（保留完整数据视图）
    sns.scatterplot(
        data=failed_roots, 
        x='H_state', y='H_action', 
        size='Criticality', hue='Criticality', style='Experiment',
        palette='viridis', sizes=(50, 500), alpha=0.75, edgecolor='w', linewidth=0.5
    )
    
    experiments = failed_roots['Experiment'].unique()
    crosshair_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    lower_q, upper_q = outlier_bounds
    
    for i, exp in enumerate(experiments):
        exp_data = failed_roots[failed_roots['Experiment'] == exp]
        if exp_data.empty:
            continue
            
        # ==================== 异常值排除逻辑 ====================
        hs_series = exp_data['H_state']
        ha_series = exp_data['H_action']
        
        # 计算该模型当前特征的合理分位数边界
        hs_lower, hs_upper = hs_series.quantile(lower_q), hs_series.quantile(upper_q)
        ha_lower, ha_upper = ha_series.quantile(lower_q), ha_series.quantile(upper_q)
        
        # 提取位于合理区间内的“健康数据”
        filtered_hs = hs_series[(hs_series >= hs_lower) & (hs_series <= hs_upper)]
        filtered_ha = ha_series[(ha_series >= ha_lower) & (ha_series <= ha_upper)]
        
        # 基于过滤后的数据计算原点（基准线）
        median_hs = filtered_hs.median()
        median_ha = filtered_ha.median()
        # ========================================================
        
        color = crosshair_colors[i % len(crosshair_colors)]
        
        # 绘制基准十字线
        plt.axvline(median_hs, color=color, linestyle='--', alpha=0.6, 
                    label=f'{exp} Origin $H_{{state}}$')
        plt.axhline(median_ha, color=color, linestyle=':', alpha=0.6, 
                    label=f'{exp} Origin $H_{{action}}$')
        
        # 在十字交叉点标注文字
        plt.text(median_hs, median_ha, f' {exp} Origin', color=color, 
                 alpha=0.9, fontsize=11, fontweight='bold', va='bottom', ha='left')

    plt.title(f'Pivotal Points Quadrant Analysis: Rescue Potential\n(Origin calculated within {lower_q*100:.0f}%-{upper_q*100:.0f}% percentile)', fontsize=16, pad=15)
    plt.xlabel('State Entropy ($H_{state}$)', fontsize=14)
    plt.ylabel('Action Entropy ($H_{action}$)', fontsize=14)
    
    # 将图例移出绘图区，防止重叠
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rescue_potential_quadrant.png'), dpi=300)
    plt.close()

def plot_strategy_interaction_heatmap(df, output_dir="plots"):
    """
    图 2：绘制策略交互上下文热力图
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    failed_roots = df[df['Node_Type'] == 'Failed Root (Rescue Potential)']
    if failed_roots.empty:
        print("[Plot 2] 无 Failed Root 数据，跳过热力图绘制。")
        return

    pivot_data = failed_roots.pivot_table(
        values='Criticality', 
        index='Prev_User_Category', 
        columns='Sys_Category', 
        aggfunc='mean'
    ).fillna(0)
    
    plt.figure(figsize=(11, 7))
    sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5, vmin=0, vmax=1.0)
    plt.title('Average Rescue Potential by Strategy Interaction Context', fontsize=16, pad=15)
    plt.xlabel('System Strategy Category (Current Turn)', fontsize=12)
    plt.ylabel('User Resistance Category (Previous Turn)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_interaction_heatmap.png'), dpi=300)
    plt.close()


def plot_strategy_vulnerability_comparison(df, output_dir="plots"):
    """
    图 3：绘制成功轨迹中的策略脆弱性对比图
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    succ_roots = df[df['Node_Type'] == 'Successful Root (Vulnerability)']
    if succ_roots.empty:
        print("[Plot 3] 无 Successful Root 数据，跳过脆弱性柱状图绘制。")
        return

    plt.figure(figsize=(11, 7))
    order = succ_roots.groupby('Sys_Category')['Criticality'].mean().sort_values(ascending=False).index
    sns.barplot(
        data=succ_roots, 
        x='Sys_Category', y='Criticality', hue='Experiment', 
        order=order, capsize=.1, palette='mako'
    )
    plt.title('Strategy Vulnerability in Successful Trajectories by Experiment', fontsize=16, pad=15)
    plt.xlabel('System Strategy Category', fontsize=14)
    plt.ylabel('Vulnerability Rate', fontsize=14)
    plt.xticks(rotation=15)
    plt.legend(title='Experiment')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vulnerability_by_strategy_exp_comparison.png'), dpi=300)
    plt.close()


def explore_credit_assignment(df, output_dir="plots"):
    """
    探索性分析：计算启发式信用分数 (Credit Score) 并分析其与轮次、熵值、策略的关系。
    Credit Score 定义:
      - 失败轨迹: 惩罚 = - 挽救率 (Rescue Potential)
      - 成功轨迹: 奖励 = + 脆弱率 (Vulnerability)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if df.empty:
        print("[Credit Analysis] DataFrame is empty. Skip.")
        return

    # 1. 计算 Heuristic Credit Score
    # 假设 df 中 'Criticality' 已经计算好：
    # Failed Root 的 Criticality = 挽救率
    # Successful Root 的 Criticality = 脆弱率
    df['Credit_Score'] = df.apply(
        lambda row: -row['Criticality'] if 'Failed' in row['Node_Type'] else row['Criticality'], 
        axis=1
    )

    # =====================================================================
    # 图 4. 信用分数随轮次的时序分布 (Credit Score by Turn)
    # 目的: 探索是前期还是后期的决策更容易引发巨大的惩罚/奖励
    # =====================================================================
    plt.figure(figsize=(12, 6))
    
    # 绘制小提琴图，展示每个轮次上信用分数的分布和密度
    sns.violinplot(
        data=df, x='Turn', y='Credit_Score', hue='Experiment', 
        split=True, inner='quartile', palette='muted'
    )
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    plt.title('Distribution of Credit Score over Dialogue Turns', fontsize=16, pad=15)
    plt.xlabel('Dialogue Turn', fontsize=14)
    plt.ylabel('Heuristic Credit Score\n(-1=Critical Blunder, +1=Brilliant Move)', fontsize=14)
    plt.legend(title='Experiment', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'credit_score_by_turn.png'), dpi=300)
    plt.close()

    # =====================================================================
    # 图 5. 认知熵空间的信用投影 (Credit Score Mapping in Entropy Space)
    # 目的: 查看在 H_state 和 H_action 的什么组合下，最容易出现大奖或大罚
    # =====================================================================
    # 为了清晰展示，我们使用 hexbin 或散点图的平滑插值，这里采用带颜色映射的散点图
    plt.figure(figsize=(14, 9))
    
    # 将图分为左右两个子图，按 Experiment 拆分（若只有一个实验则占满）
    experiments = df['Experiment'].unique()
    num_exp = len(experiments)
    
    for i, exp in enumerate(experiments):
        plt.subplot(1, num_exp, i + 1)
        exp_data = df[df['Experiment'] == exp]
        
        # 使用 RdYlGn (红-黄-绿) 色谱：红色代表严厉惩罚(负)，绿色代表丰厚奖励(正)
        scatter = plt.scatter(
            exp_data['H_state'], exp_data['H_action'], 
            c=exp_data['Credit_Score'], cmap='RdYlGn', 
            s=80, alpha=0.8, edgecolor='w', vmin=-1.0, vmax=1.0
        )
        
        plt.title(f'{exp}\nCredit Mapping', fontsize=14)
        plt.xlabel('State Entropy ($H_{state}$)', fontsize=12)
        if i == 0:
            plt.ylabel('Action Entropy ($H_{action}$)', fontsize=12)
            
        plt.colorbar(scatter, label='Credit Score')
        
        # 同样可以加上中位数十字虚线作为参考
        plt.axvline(exp_data['H_state'].median(), color='gray', linestyle='--')
        plt.axhline(exp_data['H_action'].median(), color='gray', linestyle='--')

    plt.suptitle('Credit Score Projection in Cognitive Entropy Space', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'credit_score_entropy_mapping.png'), dpi=300)
    plt.close()

    # =====================================================================
    # 图 6. 策略交互上下文的信用期望热力图 (Expected Credit by Strategy)
    # 目的: 指导模型在特定用户抵抗下，到底该选什么策略 (负数表示绝对别选，正数表示强烈建议)
    # =====================================================================
    plt.figure(figsize=(11, 7))
    
    # 计算均值，这里包含了成功(正)和失败(负)的综合期望
    credit_pivot = df.pivot_table(
        values='Credit_Score', 
        index='Prev_User_Category', 
        columns='Sys_Category', 
        aggfunc='mean'
    ).fillna(0)
    
    # 依然使用 RdYlGn，中心值为 0
    sns.heatmap(credit_pivot, annot=True, cmap='RdYlGn', center=0, fmt=".2f", linewidths=.5, vmin=-0.8, vmax=0.8)
    
    plt.title('Expected Credit Score by Strategy Interaction Context', fontsize=16, pad=15)
    plt.xlabel('System Strategy Category (Current Turn)', fontsize=12)
    plt.ylabel('User Resistance Category (Previous Turn)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'credit_expected_strategy_heatmap.png'), dpi=300)
    plt.close()
    
    print("\n[Analysis Complete] 信用分数(Credit Score)探索图表已生成完毕。")

# ==================== 4. 主程序入口 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', required=True, help="List of EXP result json files")
    parser.add_argument("--names", nargs='+', required=True, help="List of Model Names corresponding to files")
    parser.add_argument("--output_dir", default="eval_results", help="Directory to save analysis results")
    args = parser.parse_args()

    if len(args.files) != len(args.names):
        print("[Error] Number of files and names must match.")
        return
    
    EXPERIMENTS_CONFIG = []

    for file_path, model_name in zip(args.files, args.names):
        EXPERIMENTS_CONFIG.append({"name": model_name, "path": file_path})
    os.makedirs(args.output_dir, exist_ok=True)

    output_directory = args.output_dir
    
    print("=== 开始解析实验 JSON 数据 ===")
    df_results = process_experiments(EXPERIMENTS_CONFIG)
    
    if not df_results.empty:
        print(f"\n[Success] 共解析出 {len(df_results)} 个有效的分叉节点数据记录。")
        
        print("正在生成图 1: 关键节点象限分析图...")
        plot_rescue_potential_quadrants(df_results, output_dir=output_directory)
        
        print("正在生成图 2: 策略交互上下文热力图...")
        plot_strategy_interaction_heatmap(df_results, output_dir=output_directory)
        
        print("正在生成图 3: 策略脆弱性对比柱状图...")
        plot_strategy_vulnerability_comparison(df_results, output_dir=output_directory)
        
        print("正在进行动态奖励（Credit Score）探索性分析与作图...")
        explore_credit_assignment(df_results, output_dir=output_directory)
        
        # 3. 训练奖励预测数学模型
        print("正在构建并训练基于特征的数学预测模型...")
        build_reward_prediction_model(df_results, output_dir=output_directory)
        print(f"\n所有图表已成功保存至 '{output_directory}' 目录中。")
        
        csv_path = os.path.join(output_directory, "entropy_strategy_criticality_analysis.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"数据明细已保存至: {csv_path}")

        # ==============================================================
        # 新增：直接把 df_results 丢给验证模块，免去读写文件的麻烦
        # ==============================================================
        validate_heuristic.run_validation(df_results, output_dir=output_directory)
    else:
        print("\n[Error] 未解析到任何有效数据。")

if __name__ == "__main__":
    main()