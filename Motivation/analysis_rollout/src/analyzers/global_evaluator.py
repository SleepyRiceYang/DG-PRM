import pandas as pd
import numpy as np

class GlobalEvaluator:
    def __init__(self, df_meta, df_persona):
        self.df_meta = df_meta
        self.df_persona = df_persona
        # 关联人格信息
        self.df_meta = self.df_meta.join(self.df_persona, on='User')

    def compute_trajectory_level_stats(self):
        """
        Trajectory Level:
        - 每条轨迹一条记录，补齐用户人格/决策风格与分支元信息
        - 复用 user-level 命名语义：Root_Success, Success
        """
        if self.df_meta.empty:
            return pd.DataFrame()

        df = self.df_meta.copy()

        # 每个用户的 root 轨迹成功标记（原始轨迹成功）
        root_map = (
            df[df['IsRoot'] == True]
            .drop_duplicates(subset=['User'])
            .set_index('User')['Success']
            .to_dict()
        )

        # 对于当前轨迹：Success 视为“干预/修改后轨迹成功”
        # root 轨迹本身没有干预，语义上仍记为当前轨迹成功（与 user-level 字段一致）
        df['Root_Success'] = df['User'].map(root_map).fillna(False).astype(bool)
        df['Intervened_Success'] = df['Success'].astype(bool)
        df['Outcome'] = np.where(df['Success'], 'Success', 'Fail')

        cols = [
            'User', 'TrajID', 'ParentID', 'BranchTurn', 'IsRoot',
            'Success', 'Intervened_Success', 'Root_Success', 'Outcome',
            'Length', 'BigFive', 'Style'
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan

        return df[cols].sort_values(by=['User', 'IsRoot', 'TrajID'], ascending=[True, False, True]).reset_index(drop=True)

    def compute_detailed_stats(self):
        """
        计算三个层级的详细统计数据：
        1. User Level: 每个用户的成功率、轨迹占比等
        2. Persona Level: 按人格分组的聚合指标
        3. Global Level: 整体实验的宏观指标
        """
        
        # --- 1. User Level Statistics ---
        user_stats = []
        for user, group in self.df_meta.groupby('User'):
            # 基础计数
            total_trajs = len(group)
            success_trajs = group['Success'].sum()
            success_traj_ratio = success_trajs / total_trajs if total_trajs > 0 else 0
            
            # Root 分析
            root_row = group[group['IsRoot'] == True]
            if root_row.empty: continue # Should not happen
            root_success = bool(root_row.iloc[0]['Success'])
            
            # Branch 分析
            branches = group[group['IsRoot'] == False]
            branch_success_count = branches['Success'].sum()
            has_branch_success = branch_success_count > 0
            
            # 状态判定
            is_rescued = (not root_success) and has_branch_success
            
            user_stats.append({
                "User": user,
                "BigFive": group.iloc[0]['BigFive'],
                "Style": group.iloc[0]['Style'],
                "Total_Trajs": total_trajs,
                "Success_Trajs": success_trajs,
                "Success_Traj_Ratio": success_traj_ratio,
                "Root_Success": int(root_success),     # 0 or 1
                "Is_Rescued": int(is_rescued),         # 0 or 1
                "Needs_Rescue": int(not root_success)  # 0 or 1 (Root失败才需要挽救)
            })
            
        df_user_stats = pd.DataFrame(user_stats)
        
        # --- 2. Persona Level Statistics ---
        # 我们需要聚合 User Level 的数据来计算比率
        persona_stats = []
        # 按 BigFive 和 Style 分组
        for (bf, st), group in df_user_stats.groupby(['BigFive', 'Style']):
            # 人数
            n_users = len(group)
            
            # 1. 初始成功率 (User级): 初始成功的用户数 / 总用户数
            init_success_rate = group['Root_Success'].sum() / n_users
            
            # 2. 挽救率 (User级): 挽救成功的用户数 / 初始失败的用户数
            n_failed_roots = group['Needs_Rescue'].sum()
            rescue_rate = group['Is_Rescued'].sum() / n_failed_roots if n_failed_roots > 0 else 0.0
            
            # 3. 成功轨迹占比 (Trajectory级): 该组所有成功轨迹 / 该组所有轨迹
            # 注意：这里是加权平均
            total_grp_trajs = group['Total_Trajs'].sum()
            total_grp_succ_trajs = group['Success_Trajs'].sum()
            traj_success_ratio = total_grp_succ_trajs / total_grp_trajs if total_grp_trajs > 0 else 0
            
            persona_stats.append({
                "BigFive": bf,
                "Style": st,
                "User_Count": n_users,
                "Initial_Success_Rate": init_success_rate,
                "Rescue_Rate": rescue_rate,
                "Avg_Success_Trajs_Per_User": total_grp_succ_trajs / n_users,
                "Trajectory_Success_Ratio": traj_success_ratio
            })
            
        df_persona_stats = pd.DataFrame(persona_stats)
        
        # --- 3. Global Level Statistics ---
        global_stats = {
            "Total_Users": len(df_user_stats),
            "Total_Trajs": df_user_stats['Total_Trajs'].sum(),
            "Total_Success_Trajs": df_user_stats['Success_Trajs'].sum(),
            "Global_Trajectory_Success_Ratio": df_user_stats['Success_Trajs'].sum() / df_user_stats['Total_Trajs'].sum(),
            "Global_Initial_Success_Rate": df_user_stats['Root_Success'].mean(),
            # 全局挽救率
            "Global_Rescue_Rate": df_user_stats['Is_Rescued'].sum() / df_user_stats['Needs_Rescue'].sum()
        }
        
        return df_user_stats, df_persona_stats, global_stats