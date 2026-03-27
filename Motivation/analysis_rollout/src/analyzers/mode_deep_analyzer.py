import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from .unique_segment_analyzer import is_traj_success


class ModeDeepAnalyzer:
    """多文件 / 多 mode 深度对比分析。

    功能：
    - 合并所有文件的 unique 段指标，绘制全局 Success vs Fail trends（combine_metrics_dashboard）。
    - 统计各分叉轮次的分支成功/失败数量与成功率（timing_success_fail_analysis）。
    - 比较不同 mode 的：
      - root / oracle / rescue rate（用户级）；
      - root vs branch 轨迹成功率（轨迹级）；
      - 每个用户在不同 mode 下的分支成功比例。
    - 输出详细 reports（JSON + CSV）。
    """

    METRICS = ['hs', 'ha', 'delta_hs', 'delta_ha', 'z_hs', 'z_ha']

    def __init__(self, base_dir, output_name="analysis_v1", visualizer=None):
        self.base_dir = Path(base_dir)
        # 统一：所有输出先进入 base_dir/output_name，再在其中建立 deep_analysis 子目录
        self.output_root = self.base_dir / output_name
        self.deep_dir = self.output_root / "deep_analysis"
        os.makedirs(self.deep_dir, exist_ok=True)

        self.visualizer = visualizer

        # 汇总容器
        self.global_collector = {
            'success': {m: defaultdict(list) for m in self.METRICS},
            'fail': {m: defaultdict(list) for m in self.METRICS},
        }
        self.timing_stats = defaultdict(lambda: {'success': 0, 'fail': 0, 'total': 0})
        self.mode_user_stats = defaultdict(lambda: defaultdict(lambda: {
            'root_succ': False,
            'branch_succ_count': 0,
            'num_branches': 0,
        }))
        self.mode_traj_stats = defaultdict(lambda: {
            'root_total': 0,
            'root_succ': 0,
            'branch_total': 0,
            'branch_succ': 0,
        })
        self.files = []  # [{'path': Path, 'mode': str}]

    def add_file(self, filename, mode_name=None, auto_name=True):
        """注册一个 JSON 结果文件。

        filename: 相对于 base_dir 的文件名或绝对路径。
        mode_name: 模式名称，如果 auto_name=True 则根据文件名解析。
        """
        fpath = Path(filename)
        if not fpath.is_absolute():
            fpath = self.base_dir / fpath
        if not fpath.exists():
            return False
        if auto_name and mode_name is None:
            mode_name = self._parse_mode_name(fpath.name)
        elif mode_name is None:
            mode_name = fpath.stem
        self.files.append({'path': fpath, 'mode': mode_name})
        return True

    @staticmethod
    def _parse_mode_name(filename: str) -> str:
        name = filename.replace("results_", "").split("_t_")[0]
        return "".join(word.capitalize() for word in name.split("_"))

    @staticmethod
    def _safe_calc_z(val, history):
        if len(history) < 2:
            return 0.0
        return float((val - np.mean(history)) / (np.std(history) + 1e-6))

    def _process_single_file(self, path: Path, mode: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Error] Failed to read {path}: {e}")
            return

        for episode in data:
            user_id = episode.get('user_id', 'unknown')
            trajs = {t['id']: t for t in episode.get('trajectories', [])}
            root_traj = trajs.get('root')
            if not root_traj:
                continue

            is_root_succ = is_traj_success(root_traj)
            self.mode_traj_stats[mode]['root_total'] += 1
            if is_root_succ:
                self.mode_traj_stats[mode]['root_succ'] += 1
            self.mode_user_stats[mode][user_id]['root_succ'] = is_root_succ

            for t_id, traj in trajs.items():
                is_this_succ = is_traj_success(traj)
                status = 'success' if is_this_succ else 'fail'
                start_round = 1 if t_id == 'root' else traj.get('branch_at_turn', 1)

                if t_id != 'root':
                    self.mode_traj_stats[mode]['branch_total'] += 1
                    if is_this_succ:
                        self.mode_traj_stats[mode]['branch_succ'] += 1
                    self.mode_user_stats[mode][user_id]['num_branches'] += 1
                    if is_this_succ:
                        self.mode_user_stats[mode][user_id]['branch_succ_count'] += 1

                    b_round = traj.get('branch_at_turn')
                    if b_round is not None:
                        self.timing_stats[b_round]['total'] += 1
                        if is_this_succ:
                            self.timing_stats[b_round]['success'] += 1
                        else:
                            self.timing_stats[b_round]['fail'] += 1

                # unique segment 指标汇总
                hs_hist, ha_hist = [], []
                for turn in traj.get('turns', []):
                    hs, ha, r = turn.get('hs'), turn.get('ha'), turn.get('round')
                    if hs is None or ha is None or r is None:
                        continue
                    d_hs = hs - hs_hist[-1] if hs_hist else 0.0
                    d_ha = ha - ha_hist[-1] if ha_hist else 0.0
                    z_hs = self._safe_calc_z(hs, hs_hist)
                    z_ha = self._safe_calc_z(ha, ha_hist)
                    hs_hist.append(hs)
                    ha_hist.append(ha)
                    if r >= start_round:
                        v = {
                            'hs': float(hs),
                            'ha': float(ha),
                            'delta_hs': float(d_hs),
                            'delta_ha': float(d_ha),
                            'z_hs': float(z_hs),
                            'z_ha': float(z_ha),
                        }
                        for m in self.METRICS:
                            self.global_collector[status][m][r].append(v[m])

    def _aggregate_deep_metrics(self):
        agg = {}
        for status in ['success', 'fail']:
            agg[status] = {}
            for m in self.METRICS:
                rounds = sorted(self.global_collector[status][m].keys())
                if not rounds:
                    agg[status][m] = {"rounds": [], "mean": [], "std": []}
                    continue
                agg[status][m] = {
                    "rounds": rounds,
                    "mean": [float(np.mean(self.global_collector[status][m][r])) for r in rounds],
                    "std": [float(np.std(self.global_collector[status][m][r])) for r in rounds],
                }
        return agg

    def _serialize_defaultdict(self, d):
        if isinstance(d, defaultdict):
            return {k: self._serialize_defaultdict(v) for k, v in d.items()}
        if isinstance(d, dict):
            return {k: self._serialize_defaultdict(v) for k, v in d.items()}
        return d

    def run(self):
        if not self.files:
            print("[Error] No files registered for ModeDeepAnalyzer.")
            return None

        print(f"[*] Starting deep mode analysis on {len(self.files)} files...")
        for cfg in self.files:
            self._process_single_file(cfg['path'], cfg['mode'])

        agg_trends = self._aggregate_deep_metrics()

        # 绘制综合指标 dashboard
        if self.visualizer is not None:
            self.visualizer.plot_unique_metrics_dashboard(
                agg_trends,
                fig_dir=os.path.join(self.deep_dir, 'figures'),
                title_prefix="Global Metric Trends: Success vs Failure (Combined)"
            )
            self.visualizer.plot_timing_success_fail(
                self.timing_stats,
                os.path.join(self.deep_dir, 'figures')
            )
            self.visualizer.plot_mode_comprehensive_performance(
                self.mode_user_stats,
                self.mode_traj_stats,
                os.path.join(self.deep_dir, 'figures')
            )

        # 输出 reports
        reports_dir = self.deep_dir / 'reports'
        os.makedirs(reports_dir, exist_ok=True)

        with open(reports_dir / 'combined_unique_trends.json', 'w', encoding='utf-8') as f:
            json.dump(agg_trends, f, indent=2, ensure_ascii=False)

        # timing stats
        with open(reports_dir / 'branch_timing_stats.json', 'w', encoding='utf-8') as f:
            json.dump(self._serialize_defaultdict(self.timing_stats), f, indent=2, ensure_ascii=False)

        # mode-level stats
        with open(reports_dir / 'mode_traj_stats.json', 'w', encoding='utf-8') as f:
            json.dump(self._serialize_defaultdict(self.mode_traj_stats), f, indent=2, ensure_ascii=False)
        with open(reports_dir / 'mode_user_stats.json', 'w', encoding='utf-8') as f:
            json.dump(self._serialize_defaultdict(self.mode_user_stats), f, indent=2, ensure_ascii=False)

        print(f"[✔] Mode deep analysis completed. Reports and figures saved to: {self.output_root}")

        return {
            'trends': agg_trends,
            'timing_stats': self.timing_stats,
            'mode_traj_stats': self.mode_traj_stats,
            'mode_user_stats': self.mode_user_stats,
        }
