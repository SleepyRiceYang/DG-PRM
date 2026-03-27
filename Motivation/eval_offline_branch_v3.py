import json
import numpy as np
import os, sys
import pandas as pd
from abc import ABC, abstractmethod
from collections import defaultdict
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# ==================== 路径与导入设置 ====================
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
grandparent_dir = os.path.dirname(os.path.dirname(parent_dir))
sys.path.insert(0, grandparent_dir)

# ==================== 1. 数据加载与预处理 ====================
class DataLoader:
    def __init__(self, exp_file_path):
        self.data = self._load_data(exp_file_path)
        self.stats = self._calibrate_statistics()
        
    def _load_data(self, path):
        if not os.path.exists(path):
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

"""
用于分析用户画像
"""
class PersonaAnalyzer:
    def __init__(self, persona_file):
        self.persona_map = self._load_personas(persona_file)
        
    def _load_personas(self, path):
        if not os.path.exists(path):
            print(f"[Warning] Persona file not found: {path}")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def analyze(self, policy_name, episode_details):
        """
        输入: 策略名，该策略下的 episode 详细结果
        输出: 按人格分组的统计数据
        """
        # 结构: {Dimension: {Category: {'fail': 0, 'rescued': 0}}}
        stats = defaultdict(lambda: defaultdict(lambda: {'fail': 0, 'rescued': 0}))
        
        for item in episode_details:
            user_id = item['user_id']
            # 如果原始就成功了，不纳入"挽救率"的分母
            if item['is_org_success']: continue
            
            is_rescued = item['is_rescued']
            
            # 获取该用户的画像
            profile = self.persona_map.get(user_id, {})
            
            # 1. Big-Five
            big5 = profile.get('Big-Five Personality', 'Unknown').strip()
            # 2. Decision Style
            style = profile.get('Decision-Making Style', 'Unknown').strip()
            
            # 统计 Big-Five
            stats['Big-Five'][big5]['fail'] += 1
            if is_rescued: stats['Big-Five'][big5]['rescued'] += 1
            
            # 统计 Style
            stats['Decision-Style'][style]['fail'] += 1
            if is_rescued: stats['Decision-Style'][style]['rescued'] += 1
            
        return stats

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

    def plot_persona_heatmap(self, big5_data, style_data):
        """
        绘制人格 x 策略 的挽救率热力图
        big5_data: DataFrame [Strategy, Persona, RescueRate]
        """
        if big5_data.empty and style_data.empty: return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # 1. Big-Five Heatmap
        if not big5_data.empty:
            # Pivot table: Index=Persona, Columns=Strategy, Values=Rate
            df_pivot = big5_data.pivot(index="Persona", columns="Strategy", values="RescueRate")
            sns.heatmap(df_pivot, annot=True, fmt=".1%", cmap="YlGnBu", ax=ax1, cbar_kws={'label': 'Rescue Rate'})
            ax1.set_title(f"[{self.model_name}] Rescue Rate by Big-Five Personality", fontsize=14, weight='bold')
            ax1.set_xlabel("Strategy", fontsize=12)
            ax1.set_ylabel("Personality", fontsize=12)
            # 旋转X轴标签防止重叠
            plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

        # 2. Decision Style Heatmap
        if not style_data.empty:
            df_pivot = style_data.pivot(index="Persona", columns="Strategy", values="RescueRate")
            sns.heatmap(df_pivot, annot=True, fmt=".1%", cmap="YlGnBu", ax=ax2, cbar_kws={'label': 'Rescue Rate'})
            ax2.set_title(f"[{self.model_name}] Rescue Rate by Decision-Making Style", fontsize=14, weight='bold')
            ax2.set_xlabel("Strategy", fontsize=12)
            ax2.set_ylabel("Style", fontsize=12)
            plt.setp(ax2.get_xticklabels(), rotation=30, ha='right')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, f"{self.model_name}_persona_rescue_heatmap.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved persona heatmap to {save_path}")
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
        
        # [新增] 绘制分叉分布气泡图
        visualizer.plot_branching_distribution_comparison(all_policy_logs)
        
        # 保存 CSV
        df.to_csv(os.path.join(args.output_dir, f"{model_name}_summary.csv"), index=False)

    print(f"\nAll evaluations done. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()