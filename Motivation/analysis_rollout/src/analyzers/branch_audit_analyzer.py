import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from .unique_segment_analyzer import is_traj_success


class BranchAuditAnalyzer:
    """分支审计与长度统计分析器。

    并入 main.py 统一输出:
    - reports/branch_audit/branch_strategy_diff_details.csv
    - reports/branch_audit/branch_success_length_summary.json

    轨迹长度定义: 最后一轮 turn.round 作为长度 (与旧脚本保持一致)。
    """

    def __init__(self, rollout_path: str, output_dir: str):
        self.rollout_path = rollout_path
        self.output_dir = output_dir
        self.report_dir = os.path.join(self.output_dir, "reports", "branch_audit")
        os.makedirs(self.report_dir, exist_ok=True)

    @staticmethod
    def _traj_length(traj) -> int:
        turns = traj.get("turns", [])
        if not turns:
            return 0
        return int(turns[-1].get("round", 0))

    @staticmethod
    def _get_turn_at_round(traj, round_num):
        for turn in traj.get("turns", []):
            if turn.get("role") == "Persuader" and turn.get("round") == round_num:
                return turn
        return None

    @staticmethod
    def _get_strategy_at_round(traj, round_num):
        turn = BranchAuditAnalyzer._get_turn_at_round(traj, round_num)
        if not turn:
            return "N/A"
        return turn.get("strategy_name") or turn.get("strategy") or "Unknown"

    def run(self):
        with open(self.rollout_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        detail_rows = []
        # user -> stats
        user_stats = defaultdict(lambda: {
            "total_trajs": 0,
            "success_trajs": 0,
            "success_branch_lens": [],
        })

        global_total_trajs = 0
        global_success_trajs = 0
        global_success_branch_lens = []

        for episode in data:
            user_id = episode.get("user_id", "Unknown")
            trajs = episode.get("trajectories", [])
            traj_by_id = {t["id"]: t for t in trajs}

            root = traj_by_id.get("root")
            if not root:
                continue

            # root 轨迹
            root_succ = is_traj_success(root)
            root_len = self._traj_length(root)

            user_stats[user_id]["total_trajs"] += 1
            if root_succ:
                user_stats[user_id]["success_trajs"] += 1

            detail_rows.append({
                "User ID": user_id,
                "Type": "Root",
                "Traj ID": "root",
                "Branch Round": "-",
                "Old Strategy": "-",
                "New Strategy": "-",
                "Success": bool(root_succ),
                "Turns": root_len,
            })

            # branch 轨迹
            for t_id, traj in traj_by_id.items():
                if t_id == "root":
                    continue
                b_succ = is_traj_success(traj)
                b_len = self._traj_length(traj)
                b_round = traj.get("branch_at_turn", -1)

                user_stats[user_id]["total_trajs"] += 1
                if b_succ:
                    user_stats[user_id]["success_trajs"] += 1
                    user_stats[user_id]["success_branch_lens"].append(b_len)
                    global_success_branch_lens.append(b_len)

                old_strat = self._get_strategy_at_round(root, b_round)
                new_strat = self._get_strategy_at_round(traj, b_round)

                detail_rows.append({
                    "User ID": user_id,
                    "Type": "Branch",
                    "Traj ID": t_id,
                    "Branch Round": b_round,
                    "Old Strategy": old_strat,
                    "New Strategy": new_strat,
                    "Success": bool(b_succ),
                    "Turns": b_len,
                })

        # 汇总用户级与全局级
        for u, st in user_stats.items():
            global_total_trajs += st["total_trajs"]
            global_success_trajs += st["success_trajs"]

        df_details = pd.DataFrame(detail_rows)
        csv_path = os.path.join(self.report_dir, "branch_strategy_diff_details.csv")
        df_details.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 汇总 JSON
        user_summary = {}
        for u, st in user_stats.items():
            total = st["total_trajs"]
            succ = st["success_trajs"]
            lens = st["success_branch_lens"]
            user_summary[u] = {
                "total_trajectories": int(total),
                "success_trajectories": int(succ),
                "trajectory_success_rate": float(succ / total) if total > 0 else 0.0,
                "avg_success_branch_length": float(np.mean(lens)) if lens else 0.0,
            }

        overall = {
            "global_trajectory_success_rate": float(global_success_trajs / global_total_trajs) if global_total_trajs > 0 else 0.0,
            "global_total_trajectories": int(global_total_trajs),
            "global_success_trajectories": int(global_success_trajs),
            "global_avg_success_branch_length": float(np.mean(global_success_branch_lens)) if global_success_branch_lens else 0.0,
        }

        summary_path = os.path.join(self.report_dir, "branch_success_length_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"overall": overall, "user_stats": user_summary}, f, indent=2, ensure_ascii=False)

        return {
            "details_path": csv_path,
            "summary_path": summary_path,
            "overall": overall,
        }
