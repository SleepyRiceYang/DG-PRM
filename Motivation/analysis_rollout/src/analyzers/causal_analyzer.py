import pandas as pd
import numpy as np

class CausalAnalyzer:
    def __init__(self, df_turns, df_meta, traj_lookup):
        self.df_turns = df_turns
        self.df_meta = df_meta
        self.traj_lookup = traj_lookup

    MAX_ROUND = 10  # At most 10 rounds; rounds beyond are excluded from stats

    def identify_critical_turns(self):
        """3.1 Identify at which round branching leads to rescue success (raw list)."""
        critical_points = []
        for idx, row in self.df_meta.iterrows():
            if row['IsRoot'] or not row['Success']:
                continue
            round_h = row['BranchTurn'] + 1
            if round_h > self.MAX_ROUND:
                continue
            critical_points.append({
                "BranchTurn": row['BranchTurn'],
                "Round": round_h,
                "User": row['User']
            })
        return pd.DataFrame(critical_points)

    def get_critical_turns_summary(self):
        """
        Per-round summary for plotting: total branches and success branches (Round 1..MAX_ROUND only).
        Returns DataFrame with columns: Round, TotalBranches, SuccessBranches, SuccessRatio.
        """
        branches = self.df_meta[~self.df_meta['IsRoot']].copy()
        if branches.empty:
            return pd.DataFrame(columns=['Round', 'TotalBranches', 'SuccessBranches', 'SuccessRatio'])
        branches['Round'] = branches['BranchTurn'] + 1
        branches = branches[branches['Round'] <= self.MAX_ROUND]
        agg = branches.groupby('Round').agg(
            TotalBranches=('TrajID', 'count'),
            SuccessBranches=('Success', 'sum')
        ).reset_index()
        agg['SuccessBranches'] = agg['SuccessBranches'].astype(int)
        # Ensure all rounds 1..MAX_ROUND exist (fill missing with 0)
        full = pd.DataFrame({'Round': range(1, self.MAX_ROUND + 1)})
        agg = full.merge(agg, on='Round', how='left').fillna(0)
        agg['TotalBranches'] = agg['TotalBranches'].astype(int)
        agg['SuccessBranches'] = agg['SuccessBranches'].astype(int)
        agg['SuccessRatio'] = np.where(
            agg['TotalBranches'] > 0,
            agg['SuccessBranches'].astype(float) / agg['TotalBranches'],
            0.0
        )
        return agg

    def analyze_entropy_relief(self):
        """3.3 计算熵值缓解 (Entropy Relief)
        对比：Root 在 Turn T 采取策略 A -> Turn T+1 的 Hs_root
              Branch 在 Turn T 采取策略 B -> Turn T+1 的 Hs_branch
        Relief = Hs_root - Hs_branch (正值代表 Branch 有效降低了不确定性)
        """
        relief_data = []
        
        # 遍历所有分支
        for idx, branch_meta in self.df_meta.iterrows():
            if branch_meta['IsRoot']: continue
            
            parent_id = branch_meta['ParentID']
            branch_at = branch_meta['BranchTurn'] # index
            
            # 获取 Parent 轨迹数据
            # 注意：Parent 必须是 Root 或者在该节点有记录
            # 为简化，我们假设 Parent 存在且有后续
            
            # 提取 Turn 数据
            branch_turns = self.df_turns[self.df_turns['TrajID'] == branch_meta['TrajID']]
            parent_turns = self.df_turns[self.df_turns['TrajID'] == parent_id]
            
            # 我们需要 Turn T+1 的 Hs (即 Branch 发生后的下一轮)
            target_round = branch_at + 2 # BranchAt is index (0=R1). Action at R1. Next State at R2.
            
            b_next = branch_turns[branch_turns['Turn'] == target_round]
            p_next = parent_turns[parent_turns['Turn'] == target_round]
            
            if not b_next.empty and not p_next.empty:
                hs_branch = b_next.iloc[0]['Hs']
                hs_parent = p_next.iloc[0]['Hs']
                
                # 获取当前轮的策略
                curr_round = branch_at + 1
                b_curr = branch_turns[branch_turns['Turn'] == curr_round]
                p_curr = parent_turns[parent_turns['Turn'] == curr_round]
                
                strat_branch = b_curr.iloc[0]['Strategy'] if not b_curr.empty else "Unknown"
                strat_parent = p_curr.iloc[0]['Strategy'] if not p_curr.empty else "Unknown"

                relief_data.append({
                    "User": branch_meta['User'],
                    "Turn": curr_round,
                    "EntropyRelief": hs_parent - hs_branch, # Positive is good
                    "StrategyFrom": strat_parent,
                    "StrategyTo": strat_branch,
                    "BranchOutcome": "Success" if branch_meta['Success'] else "Fail"
                })
                
        return pd.DataFrame(relief_data)
