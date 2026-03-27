import json
import os
from collections import defaultdict
import numpy as np


def is_traj_success(traj):
    """统一成功判定：success 字段 / 最后一轮 reward / 任一轮 reward >= 1.0"""
    if traj.get('success'):
        return True
    turns = traj.get('turns', [])
    if not turns:
        return False
    last_turn = turns[-1]
    if last_turn.get('reward', 0.0) >= 1.0:
        return True
    for t in turns:
        if t.get('reward', 0.0) >= 1.0:
            return True
    return False


class UniqueSegmentAnalyzer:
    """基于单个 rollout 文件的 unique segment 分析（root: 从 1 轮开始；branch: 从 branch_at_turn 开始）。

    输出：
    - reports/dynamics/unique_segments/unique_trends.json
    - reports/dynamics/unique_segments/strategy_branch_performance.json
    - reports/dynamics/unique_segments/branch_decisions.json
    - figures/dynamics/unique_segments/unique_metrics_dashboard.png
    """

    METRICS = ['hs', 'ha', 'delta_hs', 'delta_ha', 'z_hs', 'z_ha']

    def __init__(self, rollout_path, output_dir, mode_name="default", remark="unique_v1", visualizer=None):
        self.rollout_path = rollout_path
        self.output_dir = output_dir
        self.mode_name = mode_name
        self.remark = remark
        self.visualizer = visualizer

        self.report_dir = os.path.join(self.output_dir, 'reports/dynamics/unique_segments')
        os.makedirs(self.report_dir, exist_ok=True)

        self.fig_dir = os.path.join(self.output_dir, 'figures/dynamics/unique_segments')
        os.makedirs(self.fig_dir, exist_ok=True)

    @staticmethod
    def _safe_calc_z(val, history):
        if len(history) < 2:
            return 0.0
        return float((val - np.mean(history)) / (np.std(history) + 1e-6))

    def _calculate_unique_path_stats(self, turns, start_round):
        """对单条轨迹计算 unique 段上的指标序列。"""
        unique_stats = []
        hs_hist, ha_hist = [], []
        for turn in turns:
            hs = turn.get('hs')
            ha = turn.get('ha')
            r = turn.get('round')
            if hs is None or ha is None or r is None:
                continue
            d_hs = hs - hs_hist[-1] if hs_hist else 0.0
            d_ha = ha - ha_hist[-1] if ha_hist else 0.0
            z_hs = self._safe_calc_z(hs, hs_hist)
            z_ha = self._safe_calc_z(ha, ha_hist)
            hs_hist.append(hs)
            ha_hist.append(ha)
            if r >= start_round:
                unique_stats.append({
                    "round": r,
                    "hs": float(hs),
                    "ha": float(ha),
                    "delta_hs": float(d_hs),
                    "delta_ha": float(d_ha),
                    "z_hs": float(z_hs),
                    "z_ha": float(z_ha),
                })
        return unique_stats

    def _aggregate_stats(self, collector):
        agg = {}
        for status in ['success', 'fail']:
            agg[status] = {}
            for m in self.METRICS:
                rounds = sorted(collector[status][m].keys())
                if not rounds:
                    agg[status][m] = {"rounds": [], "mean": [], "std": []}
                    continue
                agg[status][m] = {
                    "rounds": rounds,
                    "mean": [float(np.mean(collector[status][m][r])) for r in rounds],
                    "std": [float(np.std(collector[status][m][r])) for r in rounds],
                }
        return agg

    def run(self):
        with open(self.rollout_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metrics_collector = {
            'success': {m: defaultdict(list) for m in self.METRICS},
            'fail': {m: defaultdict(list) for m in self.METRICS},
        }
        strategy_stats = defaultdict(lambda: {'count': 0, 'success': 0})
        branch_decisions = []

        total_episodes = len(data)
        root_success_count = 0
        oracle_success_count = 0
        rescue_count = 0

        for episode in data:
            user_id = episode.get('user_id')
            trajs = {t['id']: t for t in episode.get('trajectories', [])}
            root_traj = trajs.get('root')
            if not root_traj:
                continue
            is_root_succ = is_traj_success(root_traj)
            if is_root_succ:
                root_success_count += 1

            any_branch_succ = False

            for t_id, traj in trajs.items():
                is_this_succ = is_traj_success(traj)
                if t_id != 'root' and is_this_succ:
                    any_branch_succ = True
                status = 'success' if is_this_succ else 'fail'
                start_round = 1 if t_id == 'root' else traj.get('branch_at_turn', 1)
                path_metrics = self._calculate_unique_path_stats(traj.get('turns', []), start_round)
                for m_data in path_metrics:
                    r = m_data['round']
                    for m in self.METRICS:
                        metrics_collector[status][m][r].append(m_data[m])

                if t_id != 'root':
                    branch_round = traj.get('branch_at_turn')
                    branch_turn = next((tn for tn in traj.get('turns', []) if tn.get('round') == branch_round and tn.get('role') == 'Persuader'), None)
                    strat = branch_turn.get('strategy_name') if branch_turn else None
                    if strat:
                        strategy_stats[strat]['count'] += 1
                        if is_this_succ:
                            strategy_stats[strat]['success'] += 1
                    branch_decisions.append({
                        "user_id": user_id,
                        "branch_id": t_id,
                        "round": branch_round,
                        "is_success": bool(is_this_succ),
                        "strategy": strat,
                        "reasoning": branch_turn.get('strategy_reason') if branch_turn else None,
                        "state_analysis": branch_turn.get('state_analysis') if branch_turn else None,
                        "entropy_at_branch": {
                            "hs": branch_turn.get('hs') if branch_turn else None,
                            "ha": branch_turn.get('ha') if branch_turn else None,
                        },
                    })

            if is_root_succ or any_branch_succ:
                oracle_success_count += 1
            if (not is_root_succ) and any_branch_succ:
                rescue_count += 1

        aggregated_trends = self._aggregate_stats(metrics_collector)
        meta = {
            "total_episodes": total_episodes,
            "root_success_rate": float(root_success_count / total_episodes) if total_episodes else 0.0,
            "oracle_success_rate": float(oracle_success_count / total_episodes) if total_episodes else 0.0,
            "rescue_rate": float(rescue_count / (total_episodes - root_success_count + 1e-6)) if total_episodes else 0.0,
            "mode": self.mode_name,
            "remark": self.remark,
        }

        # 写入 reports
        with open(os.path.join(self.report_dir, 'unique_trends.json'), 'w', encoding='utf-8') as f:
            json.dump({"meta": meta, "trends": aggregated_trends}, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.report_dir, 'strategy_branch_performance.json'), 'w', encoding='utf-8') as f:
            json.dump(strategy_stats, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.report_dir, 'branch_decisions.json'), 'w', encoding='utf-8') as f:
            json.dump(branch_decisions, f, indent=2, ensure_ascii=False)

        # 绘制 unique metrics dashboard
        if self.visualizer is not None:
            self.visualizer.plot_unique_metrics_dashboard(aggregated_trends, self.fig_dir, self.mode_name)

        return {
            "meta": meta,
            "trends": aggregated_trends,
            "strategy_performance": strategy_stats,
            "branch_decisions": branch_decisions,
        }
