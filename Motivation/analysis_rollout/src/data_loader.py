import json
import re
import os
import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, rollout_file, persona_file, metrics_file=None):
        self.rollout_file = rollout_file
        self.persona_file = persona_file
        self.metrics_file = metrics_file

    @staticmethod
    def _normalize_strategy_name(strategy_name):
        """统一清洗策略名，避免尾部引号/空白导致同一策略被拆成多个标签。"""
        if strategy_name is None:
            return "None"
        s = str(strategy_name)
        s = s.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
        s = s.strip()
        # 去除两端多余引号（可连续）
        s = re.sub(r'^[\"\']+|[\"\']+$', '', s)
        # 折叠内部多余空白
        s = re.sub(r'\s+', ' ', s).strip()
        return s if s else "None"

    def load_personas(self):
        with open(self.persona_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        records = []
        for uid, info in data.items():
            records.append({
                "User": uid,
                "BigFive": str(info.get("Big-Five Personality", "Unknown")).strip().title(),
                "Style": str(info.get("Decision-Making Style", "Unknown").strip().title()),
            })
        return pd.DataFrame(records).set_index("User")

    def _infer_metrics_file(self):
        if self.metrics_file:
            return self.metrics_file if os.path.exists(self.metrics_file) else None
        if self.rollout_file.endswith('.json'):
            candidate = self.rollout_file[:-5] + '_metrics.json'
            if os.path.exists(candidate):
                return candidate
        return None

    def _load_metrics_lookup(self):
        """加载 *_metrics.json 并建立 {user::traj_id: per-turn metrics} 索引。"""
        metrics_file = self._infer_metrics_file()
        if not metrics_file:
            print("    [Heatmap] metrics file not found; token-level heatmap may be degraded.")
            return {}, {'metrics_file': None, 'loaded_trajs': 0}

        with open(metrics_file, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = list(raw.values())

        lookup = {}
        loaded = 0
        for user_entry in raw:
            user_id = user_entry.get('user_id')
            for traj in user_entry.get('trajectories', []):
                tid = traj.get('id')
                if tid is None:
                    continue
                unique_id = f"{user_id}::{tid}"
                per_turn = traj.get('metrics', []) or []
                turn_map = {}
                for itm in per_turn:
                    # 常见格式: {'turn': 3, 'metrics': [...]} 或 {'round':3,'metrics':[...]} 或直接 token list
                    if isinstance(itm, dict) and ('turn' in itm or 'round' in itm):
                        t = itm.get('turn', itm.get('round'))
                        val = itm.get('metrics', itm.get('tokens', itm.get('token_metrics', [])))
                        turn_map[t] = val
                    elif isinstance(itm, dict) and 'token' in itm:
                        # 如果是扁平 token 列表，默认挂到 unknown turn，后续再尝试回退
                        turn_map.setdefault(None, []).append(itm)
                lookup[unique_id] = turn_map
                loaded += 1

        print(f"    [Heatmap] Loaded metrics file: {metrics_file}")
        print(f"    [Heatmap] Metrics trajectories indexed: {loaded}")
        return lookup, {'metrics_file': metrics_file, 'loaded_trajs': loaded}

    def load_trajectories(self):
        """
        加载轨迹并建立 Parent-Child 索引用于因果分析
        核心修复：解决多用户 ID 冲突问题 (所有 Root ID 都是 'root')
        """
        with open(self.rollout_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # 兼容列表或字典格式
        if isinstance(raw_data, dict):
            raw_data = list(raw_data.values())

        flat_turns = []
        traj_meta = []

        # 建立快速查找字典: {unique_traj_id: traj_data}
        traj_lookup = {}

        # 加载 token-level metrics 真实数据源（*_metrics.json）
        metrics_lookup, metrics_meta = self._load_metrics_lookup()

        for user_entry in raw_data:
            user_id = user_entry['user_id']

            for traj in user_entry['trajectories']:
                # --- [Fix Start] ID Unification ---
                # 原始 ID (例如 "root" 或 "root_b0") 在不同用户间会重复
                # 我们在内存中构建全局唯一的 ID: "user_id::original_id"
                original_id = traj['id']
                unique_id = f"{user_id}::{original_id}"

                # 更新当前轨迹对象的 ID
                traj['id'] = unique_id
                traj['user_id'] = user_id  # 确保有 user_id 字段

                # 同时也需要更新 parent_id 指向，否则找不到父节点
                if traj.get('parent_id'):
                    original_parent = traj['parent_id']
                    traj['parent_id'] = f"{user_id}::{original_parent}"
                # --- [Fix End] ---

                # 挂载来自 *_metrics.json 的每轮 token metrics（若存在）
                traj['turn_metrics'] = metrics_lookup.get(unique_id, {})
                traj_lookup[unique_id] = traj

        # 遍历处理 (现在 traj_lookup 包含所有用户的轨迹，不会覆盖)
        print(f"    [Debug] Total unique trajectories loaded: {len(traj_lookup)}")

        for t_id, traj in traj_lookup.items():
            user_id = traj['user_id']
            # 判断 Root: 只要 parent_id 为 None 或者 原始ID包含 root (加上前缀后的判断)
            # 更稳健的方式是看 parent_id 是否存在
            is_root = (traj.get('parent_id') is None)
            success = traj.get('success', False)

            # 元数据：记录分支信息
            traj_meta.append({
                "User": user_id,
                "TrajID": t_id,
                "ParentID": traj.get('parent_id'),
                "BranchTurn": traj.get('branch_at_turn', -1),
                "IsRoot": is_root,
                "Success": success,
                "Length": len(traj['turns']) // 2  # Approx rounds
            })

            # 轮次数据
            for turn in traj['turns']:
                if turn['role'] == 'Persuader':
                    flat_turns.append({
                        "User": user_id,
                        "TrajID": t_id,
                        "Turn": turn['round'],
                        "Role": turn['role'],
                        "Strategy": self._normalize_strategy_name(turn.get('strategy_name', 'None')),
                        "Hs": turn.get('hs', 0.0),
                        "Ha": turn.get('ha', 0.0),
                        "Reward": turn.get('reward', 0.0),
                        "Outcome": "Success" if success else "Fail"
                    })

        df_turns = pd.DataFrame(flat_turns)
        df_meta = pd.DataFrame(traj_meta)

        self.metrics_meta = metrics_meta
        return df_turns, df_meta, traj_lookup
