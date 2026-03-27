import json
import numpy as np
import os, sys
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from abc import ABC, abstractmethod
from collections import defaultdict
from tqdm import tqdm
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

# ==================== 路径与导入设置 ====================
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
grandparent_dir = os.path.dirname(os.path.dirname(parent_dir))
sys.path.insert(0, grandparent_dir)

# 设置绘图风格
sns.set_theme(style="whitegrid")

PERSONA_PATH = "/root/EvolvingAgent-master/EvolvingAgentTest_wym/user_personas.json"

def is_traj_success(trajectory):
    """
    判断单条轨迹是否成功。
    
    逻辑：
    1. 优先检查顶层 'success' 字段。
    2. 如果没有，检查 'turns' 中是否有任何一轮的 reward >= 1.0。
    """
    if not trajectory:
        return False
    if trajectory.get('success'):
        return True
    
    turns = trajectory.get('turns', [])
    if not turns:
        return False
    for turn in turns:
        if turn.get('reward', 0.0) >= 1.0:
            return True
        
    if turns[-1].get('reward', 0.0) >= 1.0:
        return True

    return False

# ==================== 1. 数据加载与预处理 ====================
class DataLoader:
    def __init__(self, exp_file_path):
        self.data = DataLoader._load_data(exp_file_path)
        self.stats = self._calibrate_statistics()
    @staticmethod
    def _load_data(path):
        if not os.path.exists(path): # 使用json加载数据
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _calibrate_statistics(self):
        """
        计算全局统计特征
        """
        all_hs = []
        all_ha = []
        
        for ep in self.data:
            root_traj = next((t for t in ep['trajectories'] if t['id'] == 'root'), None)
            if not root_traj: continue
            
            for turn in root_traj['turns']:
                if turn['role'] == 'Persuader':
                    if turn.get('hs') is not None: all_hs.append(turn['hs'])
                    if turn.get('ha') is not None: all_ha.append(turn['ha'])
        
        if not all_hs: all_hs = [0.0]
        if not all_ha: all_ha = [0.0]

        return {
            "hs_mean": np.mean(all_hs), "hs_std": np.std(all_hs),
            "hs_p75": np.percentile(all_hs, 75),
            "ha_mean": np.mean(all_ha), "ha_std": np.std(all_ha),
            "ha_p25": np.percentile(all_ha, 25)
        }

class PersonaSensitivityAnalyzer:
    def __init__(self, exp_files, persona_file, model_name=None):
        self.exp_data = self._load_exp_data(exp_files)
        self.persona_map = self._load_persona_map(persona_file)
        self.model_name = model_name
        
    def _load_exp_data(self, data_file):
        with open(data_file, 'r') as f:
            data = json.load(f) 
        return data
    
    def _load_persona_map(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _get_persona_attrs(self, user_id):
        # 获取用户的两种人格属性
        info = self.persona_map.get(user_id, {})
        big_five = info.get('Big-Five Personality').strip().title()
        decision = info.get('Decision-Making Style').strip().title()
        return big_five, decision

    def analyze(self):
        """
        核心逻辑：遍历所有轨迹，计算 Rescue 和 Vulnerability 分数
        """
        records = []

        for ep in self.exp_data:
            user_id = ep.get('user_id')
            big_five, decision = self._get_persona_attrs(user_id)
            
            trajs = ep.get('trajectories', [])
            root_traj = next((t for t in trajs if t['id'] == 'root'), None)
            if not root_traj: continue
            
            # 使用全局定义的 is_traj_success 函数
            is_root_succ = is_traj_success(root_traj)
            
            # 按轮次聚合分叉结果
            branches_by_turn = defaultdict(list)
            for t in trajs:
                if t['id'] != 'root':
                    # 注意：有些数据可能没有 branch_at_turn，需要 get 默认值
                    b_turn = t.get('branch_at_turn')
                    if b_turn:
                        branches_by_turn[b_turn].append(t)
            
            # 计算每一轮的得分
            for turn in range(1, 11): # 假设最多10轮
                branches = branches_by_turn.get(turn, [])
                if not branches: continue
                
                # --- [修正点] ---
                # 正确的语法：遍历 branches，对每个 b 判断是否成功
                win_rate = sum(1 for b in branches if is_traj_success(b)) / len(branches)
                # ----------------
                
                if not is_root_succ:
                    # 场景 A: 挽救 (原本失败，看能不能救)
                    score = win_rate
                    scenario = 'Rescue Potential'
                else:
                    # 场景 B: 脆弱性 (原本成功，看会不会翻车)
                    score = 1.0 - win_rate
                    scenario = 'Vulnerability'
                
                records.append({
                    "Scenario": scenario,
                    "Big_Five": big_five,
                    "Decision": decision,
                    "Turn": turn,
                    "Score": score
                })
        
        return pd.DataFrame(records)

    def plot_heatmap(self, df, persona_col, output_dir):
        """绘制热力图"""
        scenarios = ['Rescue Potential', 'Vulnerability']
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        for i, sc in enumerate(scenarios):
            # 筛选数据
            sub_df = df[df['Scenario'] == sc]
            if sub_df.empty: continue
            
            # 聚合：计算 (Persona, Turn) 的平均分
            pivot = sub_df.pivot_table(
                index=persona_col, 
                columns='Turn', 
                values='Score', 
                aggfunc='mean'
            ).fillna(0)
            
            # [修改] 排序逻辑：按索引（人格名称）首字母顺序排列
            # 原代码（按均值排序）：
            # pivot['mean'] = pivot.mean(axis=1)
            # pivot = pivot.sort_values('mean', ascending=False).drop(columns=['mean'])
            
            # 新代码（按字母排序）：
            pivot = pivot.sort_index(ascending=True)
            
            # 绘图
            sns.heatmap(pivot, ax=axes[i], cmap="YlGnBu", annot=True, fmt=".2f", cbar=True)
            axes[i].set_title(f"{sc} by {persona_col}")
            axes[i].set_xlabel("Dialogue Turn")
            axes[i].set_ylabel(persona_col)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{self.model_name}_persona_sensitivity_{persona_col}.png")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    def calculate_statistical_significance(self, df, persona_col):
        """
        计算人格类型对关键性得分的统计显著性 (ANOVA)
        
        Output:
        打印每一轮的 F-statistic 和 p-value。
        如果 p < 0.05，说明在该轮次，不同人格的表现有显著差异（即：人格确实影响了结果）。
        """
        print(f"\n=== ANOVA Test for {persona_col} ===")
        
        scenarios = ['Rescue Potential', 'Vulnerability']
        
        for sc in scenarios:
            print(f"\n--- Scenario: {sc} ---")
            # 筛选场景
            sub_df = df[df['Scenario'] == sc]
            if sub_df.empty: continue
            
            # 按轮次进行检验
            for turn in sorted(sub_df['Turn'].unique()):
                turn_data = sub_df[sub_df['Turn'] == turn]
                
                # 将数据按人格分组，提取 Score 列表
                # groups 结构: [ [0.1, 0.2...], [0.8, 0.9...], ... ]
                # 每个子列表对应一种人格在该轮次下的所有得分
                groups = [data['Score'].values for name, data in turn_data.groupby(persona_col)]
                
                # 至少要有两组数据才能做对比
                if len(groups) < 2:
                    continue
                
                # 执行 ANOVA
                # f_stat: 组间差异与组内差异的比值。越大说明人格的影响越大。
                # p_val: 显著性。越小越好。
                f_stat, p_val = stats.f_oneway(*groups)
                
                # 标记显著性
                significance = ""
                if p_val < 0.001: significance = "***"
                elif p_val < 0.01: significance = "**"
                elif p_val < 0.05: significance = "*"
                
                print(f"Turn {turn:2d} | F={f_stat:.2f} | p={p_val:.4f} {significance}")

# ==================== 关键节点分析模块 ====================
class PivotalAnalyzer:
    def __init__(self, data):
        self.data = data
        self.persona_data = self._load_persona_data(PERSONA_PATH)
        self.pivot_df = self._calculate_pivot_scores()
    
    def _load_persona_data(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Persona file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_persona(self, user_id):
        """
        根据 user_id 从人格数据库中严格提取两种人格信息
        Returns: (big_five, decision_style)
        """
        # 1. 严格校验用户是否存在
        if user_id not in self.persona_data:
            # print(f"[Warning] User ID '{user_id}' not found in persona database!")
            return "Unknown", "Unknown"
        
        user_info = self.persona_data[user_id]
        
        # 2. 提取属性
        big_five = user_info.get('Big-Five Personality')
        decision_style = user_info.get('Decision-Making Style')
        
        # 3. 严格校验属性完整性
        if not big_five or not decision_style:
            # print(f"[Warning] Incomplete persona attributes for user '{user_id}'.")
             return "Unknown", "Unknown"
        
        # 4. 鲁棒性处理：去除首尾空格，首字母大写 (Title Case)
        big_five_norm = big_five.strip().title()
        decision_style_norm = decision_style.strip().title()
        
        return big_five_norm, decision_style_norm

    def _calculate_pivot_scores(self):
        records = []
        
        for ep in self.data:
            user_id = ep.get('user_id', 'unknown')
            
            # [修改] 调用新逻辑，获取两种人格
            big_five, decision_style = self._extract_persona(user_id)
            
            trajs = ep.get('trajectories', [])
            
            # 1. 找到 Root
            root_traj = next((t for t in trajs if t['id'] == 'root'), None)
            if not root_traj: continue
            
            is_root_succ = is_traj_success(root_traj)
            
            # 2. 聚合分叉数据
            branches_by_turn = defaultdict(list)
            for t in trajs:
                if t['id'] == 'root': continue
                b_turn = t.get('branch_at_turn')
                if b_turn: branches_by_turn[b_turn].append(t)
            
            # 3. 遍历每一轮
            # 注意：如果不只是分叉点才算pivot，需要遍历所有轮次
            # 但这里我们只分析有数据的点
            for turn_num, branches in branches_by_turn.items():
                total_branches = len(branches)
                success_branches = sum(1 for b in branches if is_traj_success(b))
                fail_branches = total_branches - success_branches
                
                win_rate = success_branches / total_branches if total_branches > 0 else 0
                
                if not is_root_succ:
                    phi = win_rate
                    scenario = "Rescue"
                else:
                    phi = 1.0 - win_rate
                    scenario = "Stability"
                
                # [关键修复] 确保所有需要的列都在这里
                records.append({
                    "User": user_id,
                    "Big_Five": big_five,          
                    "Decision_Style": decision_style, 
                    "Turn": turn_num,
                    "Scenario": scenario,
                    "Phi_Score": phi,
                    # 下面这三列是报错缺失的，必须加上
                    "Total_Branches": total_branches,
                    "Success_Branches": success_branches,
                    "Fail_Branches": fail_branches
                })
        
        # 如果没有数据，返回空的带有列名的 DataFrame，防止后续报错
        if not records:
            return pd.DataFrame(columns=[
                "User", "Big_Five", "Decision_Style", "Turn", "Scenario", 
                "Phi_Score", "Total_Branches", "Success_Branches", "Fail_Branches"
            ])
            
        return pd.DataFrame(records)

    def get_aggregated_stats(self, group_by="Turn", scenario="Rescue"):
        """
        聚合统计：按轮次或人格聚合
        scenario: 'Rescue' or 'Stability'
        """
        if self.pivot_df.empty: return None
        
        df = self.pivot_df[self.pivot_df['Scenario'] == scenario]
        if df.empty: return None
        
        # 聚合计算
        agg = df.groupby(group_by).agg({
            "Total_Branches": "sum",
            "Success_Branches": "sum",
            "Fail_Branches": "sum"
        }).reset_index()
        
        if scenario == "Rescue":
            agg["Rate"] = agg["Success_Branches"] / agg["Total_Branches"]
            agg["Count_Main"] = agg["Success_Branches"] 
        else:
            agg["Rate"] = agg["Fail_Branches"] / agg["Total_Branches"]
            agg["Count_Main"] = agg["Fail_Branches"] 
            
        return agg

    def get_correlation_data(self):
        """返回用于画相关性图的数据 (过滤掉无数据点)"""
        # 如果 dataframe 为空或缺少列，返回空
        if self.pivot_df.empty: return self.pivot_df
        # 确保 Num_Branches 存在，或者用 Total_Branches 替代
        if 'Total_Branches' in self.pivot_df.columns:
             return self.pivot_df[self.pivot_df['Total_Branches'] > 0]
        return self.pivot_df

# ==================== 2. 触发策略定义 ====================
class TriggerPolicy(ABC):
    def __init__(self, stats, budget_n=3, time_decay=False, enable_cutoff=True, **kwargs):
        self.stats = stats
        self.budget_n = budget_n
        self.time_decay = time_decay 
        self.enable_cutoff = enable_cutoff # [新增] 控制是否开启晚期硬截断
        self.kwargs = kwargs
        self.name = "Base"

    def apply_time_decay(self, score, turn_idx):
        if not self.time_decay: return score
        
        # 基础时间衰减公式: 1 / log(t + e)
        # Turn 1 (idx 0) -> 1.0, Turn 10 -> 0.4
        weight = 1.0 / np.log(turn_idx + np.e)
        
        # [修改] 根据 enable_cutoff 决定是否在 Turn 8 以后截断
        if self.enable_cutoff and turn_idx > 7: 
            weight *= 0.0
            
        return score * weight

    @abstractmethod
    def calculate_score(self, turn_info, history_metrics):
        """
        计算当前轮次的分叉优先级分数。分数越高，越容易被选中。
        """
        pass

# --- 具体策略 ---
class AllTriggerPolicy(TriggerPolicy):
    def __init__(self, stats):
        # 将 budget_n 设为极大值 (99)，关闭衰减和截断以获取纯粹的数据上限
        super().__init__(stats, budget_n=99, time_decay=False, enable_cutoff=False)
        self.name = "exp(All-Trigger)"
        
    def calculate_score(self, turn_info, history_metrics):
        # 所有轮次给最高分，确保全部被选中
        return 1.0

class RandomPolicy(TriggerPolicy):
    def __init__(self, stats, budget_n=3, time_decay=False, enable_cutoff=True):
        super().__init__(stats, budget_n, time_decay, enable_cutoff)
        self.name = "Random"
    def calculate_score(self, turn_info, history_metrics):
        return np.random.random()

class ActionEntropyPolicy(TriggerPolicy):
    def __init__(self, stats, budget_n=3, time_decay=False, enable_cutoff=True):
        super().__init__(stats, budget_n, time_decay, enable_cutoff)
        self.name = "ActionEntropy" + ("(T)" if time_decay else "") 
        if self.time_decay: 
            self.name += ("(C)" if enable_cutoff else "")
    def calculate_score(self, turn_info, history_metrics):
        score = turn_info.get('ha', 0.0)
        return self.apply_time_decay(score, turn_info['round']-1)

class StateEntropyPolicy(TriggerPolicy):
    def __init__(self, stats, budget_n=3, time_decay=False, enable_cutoff=True):
        super().__init__(stats, budget_n, time_decay, enable_cutoff)
        self.name = "StateEntropy" + ("(T)" if time_decay else "") 
        if self.time_decay: 
            self.name += ("(C)" if enable_cutoff else "")
    def calculate_score(self, turn_info, history_metrics):
        score = turn_info.get('hs', 0.0)
        return self.apply_time_decay(score, turn_info['round']-1)

class DissonancePolicy(TriggerPolicy):
    def __init__(self, stats, budget_n=3, time_decay=False, enable_cutoff=True):
        super().__init__(stats, budget_n, time_decay, enable_cutoff)
        self.name = "Dissonance" + ("(T)" if time_decay else "")
        if self.time_decay: 
            self.name += ("(C)" if enable_cutoff else "")
    def calculate_score(self, turn_info, history_metrics):
        hs = turn_info.get('hs', 0.0)
        ha = turn_info.get('ha', 0.0)
        score = hs - ha
        return self.apply_time_decay(score, turn_info['round']-1)

class TrendPolicy(TriggerPolicy):
    def __init__(self, stats, budget_n=3, time_decay=False, enable_cutoff=True):
        super().__init__(stats, budget_n, time_decay, enable_cutoff)
        self.name = "Trend" + ("(T)" if time_decay else "") 
        if self.time_decay: 
            self.name += ("(C)" if enable_cutoff else "")

    def calculate_score(self, turn_info, history_metrics):
        if len(history_metrics) < 1: return 0.0
        
        last_metric = history_metrics[-1]
        delta_hs = turn_info['hs'] - last_metric['hs']
        delta_ha = turn_info['ha'] - last_metric['ha']
        
        # 捕捉：状态变难 (Delta Hs > 0) 且 动作变自信 (Delta Ha < 0)
        score = delta_hs - delta_ha
        return self.apply_time_decay(score, turn_info['round']-1)

# ==================== 3. 模拟评估引擎 ====================
class OfflineEvaluator:
    def __init__(self, data_loader):
        self.data = data_loader.data
        self.stats = data_loader.stats
        
        # [修复] 初始化 score_logs，用于存储策略打分日志供可视化使用
        self.score_logs = defaultdict(list)
        
        # 初始化其他统计容器
        self.timing_stats = defaultdict(lambda: defaultdict(int)) 
        self.user_stats = defaultdict(dict)

    def evaluate(self, policy: TriggerPolicy):
        results = {
            "total_episodes": 0, "org_fail_count": 0, "rescued_count": 0,
            "total_triggers": 0, "successful_triggers": 0, 
            "rescue_turns_sum": 0,     # 成功挽救的轨迹总轮数累加
            "wasted_triggers": 0       # 选中的分叉点无效的数量
        }
        
        # 记录每个 Episode 选中的分叉点及其结果
        # 结构: {episode_index: [{'round': int, 'success': bool}, ...]}
        selected_points_log = {} 
        
        # 记录当前策略的 Timing 统计 (临时变量，最后存入 self.timing_stats)
        policy_timing = defaultdict(lambda: {'total': 0, 'success': 0, 'fail': 0})

        for ep_idx, ep in enumerate(self.data):
            results["total_episodes"] += 1
            
            # 1. 检查原始轨迹 (Root) 是否成功
            root_traj = next((t for t in ep['trajectories'] if t['id'] == 'root'), None)
            if not root_traj: continue
            
            # 兼容多种成功标记方式
            is_org_success = root_traj.get('success', False) or \
                             (root_traj['turns'] and root_traj['turns'][-1].get('reward', 0) >= 1.0)
            
            if not is_org_success:
                results["org_fail_count"] += 1
            
            # 2. 收集该 Episode 所有轮次的候选分
            candidates = [] 
            history_metrics = [] # 用于计算 Trend
            
            sys_turns = [t for t in root_traj['turns'] if t['role'] == 'Persuader']
            
            for turn in sys_turns:
                # 跳过无效轮次
                raw_hs = turn.get('hs')
                raw_ha = turn.get('ha')
                if raw_hs is None: continue

                turn_info = {
                    'hs': raw_hs, 'ha': raw_ha, 'round': turn.get('round', 0)
                }
                
                # A. 计算策略打分
                score = policy.calculate_score(turn_info, history_metrics)
                history_metrics.append(turn_info)
                
                # [修复后此处不会报错] 记录分数用于分布图
                self.score_logs[policy.name].append({'round': turn_info['round'], 'score': score})
                
                # B. 查表 (Oracle Check)
                can_rescue = False
                success_traj_len = float('inf')
                
                for t in ep['trajectories']:
                    if t.get('branch_at_turn') == turn_info['round']:
                        if t.get('success') or (t['turns'] and t['turns'][-1].get('reward', 0) >= 1.0):
                            can_rescue = True
                            t_len = t['turns'][-1].get('round', len(t['turns']) // 2)
                            if t_len < success_traj_len:
                                success_traj_len = t_len
                
                candidates.append({
                    'round': turn_info['round'],
                    'score': score,
                    'can_rescue': can_rescue,
                    'success_len': success_traj_len if can_rescue else 0
                })

            # 3. Top-N 选择
            candidates.sort(key=lambda x: x['score'], reverse=True)
            selected_turns = candidates[:policy.budget_n]
            
            # 记录选中的轮次以及结果 (用于气泡图)
            selected_points_log[ep_idx] = [
                {'round': c['round'], 'success': c['can_rescue']} 
                for c in selected_turns
            ]
            
            # 4. 结算统计
            episode_rescued = False
            best_rescue_len = float('inf')
            
            for cand in selected_turns:
                results["total_triggers"] += 1
                r = cand['round']
                policy_timing[r]['total'] += 1
                
                if cand['can_rescue']:
                    results["successful_triggers"] += 1
                    policy_timing[r]['success'] += 1
                    episode_rescued = True
                    if cand['success_len'] < best_rescue_len:
                        best_rescue_len = cand['success_len']
                else:
                    results["wasted_triggers"] += 1
                    policy_timing[r]['fail'] += 1
            
            if episode_rescued and not is_org_success:
                results["rescued_count"] += 1
                results["rescue_turns_sum"] += best_rescue_len
                
        # 将 log 加入返回结果
        results['selected_points_log'] = selected_points_log
        
        # 更新 timing stats
        self.timing_stats[policy.name] = policy_timing
        
        return results

# ==================== 4. 可视化模块 ====================
class Visualizer:
    def __init__(self, output_dir, model_name):
        self.output_dir = output_dir
        self.model_name = model_name
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_dual_pivotal_analysis(self, analyzer):
        """
        绘制两张图：
        1. Rescue Potential (Failure -> Success)
        2. Stability Analysis (Success -> Failure)
        """
        print("Generating Pivot Analysis Plots...")
        
        # 准备数据
        rescue_df = analyzer.get_aggregated_stats(group_by="Turn", scenario="Rescue")
        stability_df = analyzer.get_aggregated_stats(group_by="Turn", scenario="Stability")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
        plt.subplots_adjust(wspace=0.25)
        
        # --- Helper Function for Plotting ---
        def plot_mix_chart(ax, df, title, y_label_line, bar_color, line_color, is_rescue=True):
            if df is None or df.empty:
                ax.text(0.5, 0.5, "No Data Available", ha='center')
                return
            
            rounds = df['Turn']
            total = df['Total_Branches']
            main_count = df['Count_Main'] # Rescue时是Success数，Stability时是Fail数
            rate = df['Rate']
            
            # 双轴
            ax_count = ax.twinx()
            
            # 绘制柱状图 (Counts)
            # 背景灰条：Total Branches
            p1 = ax_count.bar(rounds, total, color='#ecf0f1', label='Total Branches', alpha=0.6, width=0.8, zorder=1)
            # 前景彩条：Flip Branches
            label_bar = 'Success Branches' if is_rescue else 'Failed Branches'
            p2 = ax_count.bar(rounds, main_count, color=bar_color, label=label_bar, alpha=0.8, width=0.8, zorder=2)
            
            # 绘制折线图 (Rate)
            # 注意：将折线绘制在 ax 上 (左轴)，这样通常折线在上层
            p3, = ax.plot(rounds, rate, color=line_color, marker='o' if is_rescue else 'x', 
                          linewidth=3, markersize=8, label=y_label_line, zorder=3)
            
            # 设置坐标轴
            ax.set_xlabel("Dialogue Round (Intervention Point)", fontsize=12)
            ax.set_ylabel(y_label_line, color=line_color, fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis='y', labelcolor=line_color)
            ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
            
            ax_count.set_ylabel("Absolute Counts", color='gray', fontsize=12)
            ax_count.set_ylim(0, max(total)*1.15)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xticks(rounds)
            
            # 合并图例
            # 技巧：手动收集 handles
            handles = [p1, p2, p3]
            labels = [h.get_label() for h in handles]
            ax.legend(handles, labels, loc='upper left')

        # --- Plot 1: Rescue Potential ---
        plot_mix_chart(
            ax1, rescue_df, 
            title="Rescue Potential: Where can we fix failures?", 
            y_label_line="Rescue Rate", 
            bar_color='#5dade2', # Blueish
            line_color='#2874a6',
            is_rescue=True
        )

        # --- Plot 2: Stability Analysis ---
        plot_mix_chart(
            ax2, stability_df, 
            title="Failure Timing: Where is success fragile?", 
            y_label_line="Failure Rate", 
            bar_color='#e74c3c', # Reddish
            line_color='#922b21',
            is_rescue=False
        )
        
        save_path = os.path.join(self.output_dir, f"{self.model_name}_pivotal_turn_analysis.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    def plot_persona_impact(self, analyzer):
        """
        绘制人格维度的热力图 (分别绘制 Big-Five 和 Decision-Style)
        """
        print("Generating Persona Analysis Plots...")
        
        # 只分析 Rescue 场景
        df = analyzer.pivot_df[analyzer.pivot_df['Scenario'] == 'Rescue']
        
        if df.empty:
            print("No Rescue data for persona analysis.")
            return

        # 定义要分析的维度列表
        persona_types = [
            ("Big_Five", "Big-Five Personality"), 
            ("Decision_Style", "Decision-Making Style")
        ]

        for col_name, title_name in persona_types:
            # 聚合
            pivot_table = df.pivot_table(
                index=col_name, 
                columns='Turn', 
                values='Phi_Score', 
                aggfunc='mean'
            )
            
            # 排序
            if not pivot_table.empty:
                pivot_table['mean'] = pivot_table.mean(axis=1)
                pivot_table = pivot_table.sort_values('mean', ascending=False)
                pivot_table = pivot_table.drop(columns=['mean'])
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(pivot_table, cmap="YlGnBu", annot=True, fmt=".2f", 
                            cbar_kws={'label': 'Avg Rescue Rate'})
                
                plt.title(f"[{self.model_name}] Critical Turns by {title_name}", fontsize=14, fontweight='bold')
                plt.xlabel("Dialogue Round")
                plt.ylabel(title_name)
                
                # 保存不同的文件名
                fname = f"{self.model_name}_persona_heatmap_{col_name.lower()}.png"
                save_path = os.path.join(self.output_dir, fname)
                plt.savefig(save_path, bbox_inches='tight')
                print(f"Saved: {save_path}")
                plt.close()

    def plot_timing_analysis(self, timing_data, policy_name):
        rounds = sorted(timing_data.keys())
        if not rounds: return

        total = [timing_data[r]['total'] for r in rounds]
        success = [timing_data[r]['success'] for r in rounds]
        fail = [timing_data[r]['fail'] for r in rounds]
        
        succ_rate = [s/t if t>0 else 0 for s,t in zip(success, total)]
        fail_rate = [f/t if t>0 else 0 for f,t in zip(fail, total)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
        # 左图: Success Timing
        ax1_twin = ax1.twinx()
        ax1.bar(rounds, total, color='#f0f0f0', label='Total Branches')
        ax1.bar(rounds, success, color='#7fb07f', label='Success Branches')
        ax1_twin.plot(rounds, succ_rate, color='darkgreen', marker='o', lw=2, label='Success Rate')
        
        ax1.set_title(f"[{self.model_name}] Success Timing: {policy_name}")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Counts")
        ax1_twin.set_ylabel("Rate", color='darkgreen')
        ax1.legend(loc='upper left')
        
        # 右图: Failure Timing
        ax2_twin = ax2.twinx()
        ax2.bar(rounds, total, color='#f0f0f0', label='Total Branches')
        ax2.bar(rounds, fail, color='#f08080', label='Fail Branches')
        ax2_twin.plot(rounds, fail_rate, color='darkred', marker='x', lw=2, label='Fail Rate')
        
        ax2.set_title(f"[{self.model_name}] Failure Timing: {policy_name}")
        ax2.set_xlabel("Round")
        ax2_twin.set_ylabel("Rate", color='darkred')
        ax2.legend(loc='upper left')
        
        plt.tight_layout()
        filename = f"{self.model_name}_timing_{policy_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def plot_global_performance(self, summary_df):
        if summary_df.empty: return
        
        methods = summary_df['Method']
        # 注意：这里需要确保 summary_df 里有 Root SR 数据，或者我们假设是 0 
        # 为了更准确，我们可以传入 root sr
        root_sr_val = 0.0 # 默认，或者从数据中获取
        if 'Root SR' in summary_df.columns:
             root_sr = summary_df['Root SR'].apply(lambda x: float(str(x).strip('%'))/100)
        else:
             root_sr = [0.0] * len(methods)

        rescue = summary_df['Rescue Rate'].apply(lambda x: float(str(x).strip('%'))/100)
        # 简单计算 Oracle = Root + (1-Root)*Rescue
        oracle = [r + (1-r)*res for r, res in zip(root_sr, rescue)]

        x = np.arange(len(methods))
        width = 0.25
        
        plt.figure(figsize=(14, 7))
        plt.bar(x - width, root_sr, width, label='Root SR', color='#bdc3c7')
        plt.bar(x, oracle, width, label='Oracle SR', color='#2ecc71')
        plt.bar(x + width, rescue, width, label='Rescue Rate', color='#3498db')
        
        plt.xticks(x, methods, rotation=30, ha='right')
        plt.title(f"[{self.model_name}] Global Performance Comparison")
        plt.ylabel("Rate")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        filename = f"{self.model_name}_global_performance.png"
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def plot_branching_distribution_comparison(self, all_policy_logs):
        """
        绘制不同策略的分叉分布气泡图
        X轴: Round, Y轴: Policy
        Size: Frequency (选择频率)
        Color: Success Rate (该节点分叉的成功率)
        """
        if not all_policy_logs: return
        
        # 1. 整理数据
        plot_data = []
        all_rounds = set()
        
        for policy_name, logs in all_policy_logs.items():
            # 统计结构: {round: {'total_selected': 0, 'success_count': 0}}
            stats_per_round = defaultdict(lambda: {'total': 0, 'succ': 0})
            total_branches_for_policy = 0
            
            for ep_idx, points in logs.items():
                for p in points:
                    r = p['round']
                    is_succ = p['success']
                    
                    stats_per_round[r]['total'] += 1
                    if is_succ:
                        stats_per_round[r]['succ'] += 1
                    
                    total_branches_for_policy += 1
                    all_rounds.add(r)
            
            # 计算指标
            for r, stat in stats_per_round.items():
                # Frequency: 该轮次被选中的次数 / 该策略总分叉次数
                freq = stat['total'] / total_branches_for_policy if total_branches_for_policy > 0 else 0
                
                # Success Rate: 该轮次分叉成功的次数 / 该轮次被选中的次数
                succ_rate = stat['succ'] / stat['total'] if stat['total'] > 0 else 0
                
                plot_data.append({
                    "Policy": policy_name,
                    "Round": r,
                    "Frequency": freq,
                    "Success Rate": succ_rate,
                    "Count Label": f"{stat['succ']}/{stat['total']}" # 可选：用于标注文本
                })
                
        df = pd.DataFrame(plot_data)
        if df.empty: return

        # 2. 绘图
        plt.figure(figsize=(16, 9))
        
        # 使用 RdYlGn (红-黄-绿) 色谱：红代表低成功率，绿代表高成功率
        # sizes 控制气泡最小和最大尺寸
        scatter = sns.scatterplot(
            data=df, 
            x="Round", 
            y="Policy", 
            size="Frequency", 
            sizes=(100, 1000), 
            hue="Success Rate",
            palette="RdYlGn", 
            hue_norm=(0, 1), # 固定颜色范围 0% - 100%
            edgecolor="black",
            alpha=0.8
        )
        
        # 3. 美化图表
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 确保 X 轴显示所有出现的轮次整数
        if all_rounds:
            plt.xticks(sorted(list(all_rounds)))
        
        plt.xlabel("Dialogue Round", fontsize=12)
        plt.ylabel("Strategy Policy", fontsize=12)
        plt.title(f"[{self.model_name}] Branching Analysis: Size=Frequency, Color=Success Rate", fontsize=14, weight='bold')
        
        # 调整 Legend 位置
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        
        # (可选) 在气泡上标注成功率数值，如果气泡够大
        # for line in range(0, df.shape[0]):
        #      plt.text(df.Round[line], df.Policy[line], f"{df['Success Rate'][line]:.2f}", 
        #               horizontalalignment='center', size='small', color='black', weight='semibold')

        plt.tight_layout()
        
        filename = f"{self.model_name}_branching_bubble_color.png"
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path)
        plt.close()
        
        print(f"Saved branching bubble plot to {save_path}")

    def plot_pivot_analysis(self, pivot_df):
        """绘制关键节点分析图表"""
        print(f"Generating Pivot Analysis plots...")
        
        # 1. Pivot Score vs. Entropy (Scatter Plot)
        # 分为两组：Rescue Potential 和 Vulnerability
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        types = ["Rescue Potential", "Vulnerability"]
        
        for i, p_type in enumerate(types):
            ax = axes[i]
            data = pivot_df[pivot_df['Type'] == p_type]
            
            if data.empty:
                ax.text(0.5, 0.5, "No Data", ha='center')
                continue
                
            # 散点图: X=Dissonance, Y=Phi
            sns.scatterplot(data=data, x="Dissonance", y="Phi", ax=ax, alpha=0.6, color='blue')
            # 加上回归线
            sns.regplot(data=data, x="Dissonance", y="Phi", ax=ax, scatter=False, color='red')
            
            ax.set_title(f"{p_type} vs. Cognitive Dissonance")
            ax.set_xlabel("Dissonance (Hs - Ha)")
            ax.set_ylabel(f"Pivot Score ({p_type})")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{self.model_name}_pivot_correlation.png"))
        plt.close()

        # 2. Pivot Score Heatmap (User vs Turn)
        # 筛选出 Top-20 用户（按平均 Phi 排序）以防图太大
        # 只画 Rescue Potential (这是主要关注点)
        rescue_data = pivot_df[pivot_df['Type'] == "Rescue Potential"]
        if not rescue_data.empty:
            pivot_matrix = rescue_data.pivot(index="User", columns="Turn", values="Phi")
            
            # 按行的平均值排序
            pivot_matrix['mean'] = pivot_matrix.mean(axis=1)
            pivot_matrix = pivot_matrix.sort_values('mean', ascending=False).drop(columns=['mean']).head(20)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_matrix, cmap="YlOrRd", annot=True, fmt=".1f", cbar_kws={'label': 'Rescue Potential'})
            plt.title(f"[{self.model_name}] Rescue Potential Heatmap (Top 20 Failed Users)")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{self.model_name}_pivot_heatmap_rescue.png"))
            plt.close()


def plot_pivotal_turn_analysis(loader, output_dir, model_name):
    """
    绘制基于轮次的关键节点分析图 (Success vs Failure Split)
    """
    print("Generating Pivotal Turn Analysis Plots...")
    
    # 1. 数据桶初始化
    # {turn: {'total': 0, 'flip': 0}}
    rescue_stats = defaultdict(lambda: {'total': 0, 'success': 0})      # Base: Fail -> Branch: Success
    vulnerability_stats = defaultdict(lambda: {'total': 0, 'fail': 0}) # Base: Success -> Branch: Fail
    
    # 2. 遍历数据
    for ep in loader.data:
        trajectories = ep.get('trajectories', [])
        root_traj = next((t for t in trajectories if t['id'] == 'root'), None)
        if not root_traj: continue
        
        is_root_succ = is_traj_success(root_traj)
        
        # 遍历该 Episode 的所有分叉
        for traj in trajectories:
            if traj['id'] == 'root': continue
            
            b_turn = traj.get('branch_at_turn')
            if not b_turn: continue
            
            is_branch_succ = is_traj_success(traj)
            
            if not is_root_succ:
                # [Case A] 原始失败 -> 看能否救回
                rescue_stats[b_turn]['total'] += 1
                if is_branch_succ:
                    rescue_stats[b_turn]['success'] += 1
            else:
                # [Case B] 原始成功 -> 看是否翻车
                vulnerability_stats[b_turn]['total'] += 1
                if not is_branch_succ:
                    vulnerability_stats[b_turn]['fail'] += 1

    # 3. 准备绘图数据
    def prepare_plot_data(stats_dict):
        rounds = sorted(stats_dict.keys())
        if not rounds: return None
        totals = [stats_dict[r]['total'] for r in rounds]
        flips = [stats_dict[r].get('success', 0) or stats_dict[r].get('fail', 0) for r in rounds]
        rates = [f/t if t>0 else 0 for f, t in zip(flips, totals)]
        return rounds, totals, flips, rates

    rescue_data = prepare_plot_data(rescue_stats)
    vuln_data = prepare_plot_data(vulnerability_stats)
    
    # 4. 开始绘图 (双子图)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    plt.subplots_adjust(wspace=0.3)
    
    # --- 左图：Rescue Analysis (Origin: Fail) ---
    if rescue_data:
        rounds, totals, succs, rates = rescue_data
        
        # 双轴
        ax1_twin = ax1.twinx()
        
        # 柱状图 (绝对数量)
        ax1_twin.bar(rounds, totals, color='#ecf0f1', label='Total Attempts', alpha=0.6, width=0.8)
        ax1_twin.bar(rounds, succs, color='#2ecc71', label='Rescued (Success)', alpha=0.8, width=0.8)
        
        # 折线图 (成功率)
        line, = ax1.plot(rounds, rates, color='#27ae60', marker='o', linewidth=3, label='Rescue Rate')
        
        # 设置样式
        ax1.set_title("Rescue Potential: Where can we fix failures?", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Dialogue Round", fontsize=12)
        ax1.set_ylabel("Rescue Rate", color='#27ae60', fontsize=12)
        ax1.set_ylim(0, 1.05)
        ax1_twin.set_ylabel("Number of Branches", color='gray', fontsize=12)
        
        # 图例
        lines = [line]
        # 获取 bar 的 handle 需要一点技巧，或者简单的手动加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ecf0f1', edgecolor='gray', label='Total Branches'),
            Patch(facecolor='#2ecc71', label='Successful Branches'),
            line
        ]
        ax1.legend(handles=legend_elements, loc='upper left')
        ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax1.set_xticks(rounds)

    # --- 右图：Vulnerability Analysis (Origin: Success) ---
    if vuln_data:
        rounds, totals, fails, rates = vuln_data
        
        ax2_twin = ax2.twinx()
        
        # 柱状图
        ax2_twin.bar(rounds, totals, color='#ecf0f1', label='Total Attempts', alpha=0.6, width=0.8)
        ax2_twin.bar(rounds, fails, color='#e74c3c', label='Collapsed (Fail)', alpha=0.8, width=0.8)
        
        # 折线图
        line, = ax2.plot(rounds, rates, color='#c0392b', marker='x', linewidth=3, label='Failure Rate')
        
        # 设置样式
        ax2.set_title("Stability Analysis: Where is success fragile?", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Dialogue Round", fontsize=12)
        ax2.set_ylabel("Breakdown Rate (Success -> Fail)", color='#c0392b', fontsize=12)
        ax2.set_ylim(0, 1.05)
        ax2_twin.set_ylabel("Number of Branches", color='gray', fontsize=12)
        
        legend_elements = [
            Patch(facecolor='#ecf0f1', edgecolor='gray', label='Total Branches'),
            Patch(facecolor='#e74c3c', label='Failed Branches'),
            line
        ]
        ax2.legend(handles=legend_elements, loc='upper left')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
        ax2.set_xticks(rounds)

    # 保存
    save_path = os.path.join(output_dir, f"{model_name}_pivotal_turn_distribution.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()

# ==================== 5. 主程序 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', required=True, help="List of EXP result json files")
    parser.add_argument("--names", nargs='+', required=True, help="List of Model Names corresponding to files")
    parser.add_argument("--output_dir", default="eval_results", help="Directory to save analysis results")
    parser.add_argument("--budget", type=int, default=3)
    args = parser.parse_args()

    if len(args.files) != len(args.names):
        print("[Error] Number of files and names must match.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 遍历每个文件（模型）进行独立评估
    for file_path, model_name in zip(args.files, args.names):
        print(f"\n{'='*20} Processing: {model_name} {'='*20}")
        
        loader = DataLoader(file_path)
        evaluator = OfflineEvaluator(loader)
        visualizer = Visualizer(args.output_dir, model_name)
        
        ## 绘制整体分析图
        print(f"\n{'='*20} Analyzing Pivotal Points {'='*20}")
        pivot_analyzer = PivotalAnalyzer(loader.data)
        visualizer = Visualizer(args.output_dir, model_name)
        visualizer.plot_dual_pivotal_analysis(pivot_analyzer)
        # visualizer.plot_persona_impact(pivot_analyzer)

        ## 分析人格因素影响
        persona_analyzer = PersonaSensitivityAnalyzer(file_path, PERSONA_PATH, model_name=model_name)
        df = persona_analyzer.analyze()
        persona_analyzer.plot_heatmap(df, "Big_Five", args.output_dir)
        persona_analyzer.plot_heatmap(df, "Decision", args.output_dir)
        df.to_csv(os.path.join(args.output_dir, f"{model_name}_persona_analysis.csv"), index=False)
        
        persona_analyzer.calculate_statistical_significance(df, "Big_Five")
        persona_analyzer.calculate_statistical_significance(df, "Decision")
        continue

        # 定义策略组
        policies = [
            # RandomPolicy(loader.stats, budget_n=args.budget),
            # 基础版
            AllTriggerPolicy(loader.stats), # [新增基线] 代表 expmod
            ActionEntropyPolicy(loader.stats, budget_n=args.budget, time_decay=False),
            ActionEntropyPolicy(loader.stats, budget_n=args.budget, time_decay=True, enable_cutoff=False),
            ActionEntropyPolicy(loader.stats, budget_n=args.budget, time_decay=True, enable_cutoff=True),
            
            StateEntropyPolicy(loader.stats, budget_n=args.budget, time_decay=False),
            StateEntropyPolicy(loader.stats, budget_n=args.budget, time_decay=True, enable_cutoff=False),
            StateEntropyPolicy(loader.stats, budget_n=args.budget, time_decay=True, enable_cutoff=True),
            
            DissonancePolicy(loader.stats, budget_n=args.budget, time_decay=False),
            DissonancePolicy(loader.stats, budget_n=args.budget, time_decay=True, enable_cutoff=False),
            DissonancePolicy(loader.stats, budget_n=args.budget, time_decay=True, enable_cutoff=True),
            # 趋势版
            TrendPolicy(loader.stats, budget_n=args.budget, time_decay=False),
            TrendPolicy(loader.stats, budget_n=args.budget, time_decay=True, enable_cutoff=False),
            TrendPolicy(loader.stats, budget_n=args.budget, time_decay=True),
        ]
        
        summary = []
        
        # [新增] 用于收集所有策略的选点日志，用于画气泡图
        all_policy_logs = {} 
        
        # 获取 Root SR 用于绘图
        total_ep = 0
        root_succ = 0
        for ep in loader.data:
            total_ep += 1
            root_traj = next((t for t in ep['trajectories'] if t['id'] == 'root'), None)
            if root_traj and (root_traj.get('success', False) or (root_traj['turns'][-1].get('reward', 0) >= 1.0)):
                root_succ += 1
        root_sr_val = root_succ / total_ep if total_ep > 0 else 0
        
        for pol in policies:
            # Random 简单跑一次
            res = evaluator.evaluate(pol)
            
            # [新增] 收集日志
            all_policy_logs[pol.name] = res['selected_points_log']
            
            # 计算指标
            rescue_rate = res['rescued_count'] / (res['org_fail_count'] + 1e-6)
            
            if res['rescued_count'] > 0:
                avg_rescue_turns = res['rescue_turns_sum'] / res['rescued_count']
            else:
                avg_rescue_turns = 0.0 

            if avg_rescue_turns > 0:
                ces_score = rescue_rate * (10.0 / avg_rescue_turns)
            else:
                ces_score = 0.0

            summary.append({
                "Method": pol.name,
                "Root SR": f"{root_sr_val*100:.1f}%",
                "Rescue Rate": f"{rescue_rate * 100:.1f}%",
                "Avg Rescue Turns": f"{avg_rescue_turns:.2f}", 
                "CES (Eff-Score)": f"{ces_score:.2f}",        
                "Avg Triggers": f"{res['total_triggers'] / res['total_episodes']:.2f}",
                "Precision": f"{res['successful_triggers'] / (res['total_triggers']+1e-6) * 100:.1f}%"
            })
            
            # 绘制 Timing 图
            visualizer.plot_timing_analysis(evaluator.timing_stats[pol.name], pol.name)
            
        # 汇总与绘图
        print("\n=== Final Leaderboard ===")
        df = pd.DataFrame(summary)
        print(f"\n--- Leaderboard: {model_name} ---")
        df_sorted = df.sort_values(by="CES (Eff-Score)", ascending=False, key=lambda col: col.astype(float))
        print(df_sorted.to_markdown())
        
        visualizer.plot_global_performance(df)
        visualizer.plot_branching_distribution_comparison(all_policy_logs)
        
        # 保存 CSV
        df.to_csv(os.path.join(args.output_dir, f"{model_name}_summary.csv"), index=False)

    print(f"\nAll evaluations done. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()