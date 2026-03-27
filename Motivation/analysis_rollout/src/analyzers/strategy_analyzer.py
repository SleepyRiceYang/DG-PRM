# -*- coding: utf-8 -*-
"""
Strategy analysis: preference by personality/style, per-turn tendency,
same-turn branch re-selection, and overall statistics.
"""
import pandas as pd
import numpy as np


class StrategyAnalyzer:
    def __init__(self, df_turns, df_persona, df_meta=None):
        self.df_turns = df_turns.copy()
        self.df_turns = self.df_turns.join(df_persona, on='User')
        self.df_meta = df_meta  # for branch/parent analysis
        self.df_persona = df_persona

    # -------------------------------------------------------------------------
    # 1. Strategy preference by Big Five and by Style (separately), with tendency strength
    # -------------------------------------------------------------------------

    def _strategy_preference_for_category(self, group_col, use_outcome=None):
        """
        Generic: for each value of group_col (e.g. BigFive or Style), compute strategy counts,
        proportions, top strategies, and tendency strength (max proportion; concentration = 1 - norm_entropy).
        use_outcome: if 'Success' or 'Fail', filter turns by outcome; else use all.
        """
        df = self.df_turns.copy()
        if use_outcome:
            df = df[df['Outcome'] == use_outcome]
        df = df[df['Strategy'].notna() & (df['Strategy'].astype(str).str.strip() != '')]
        if df.empty:
            return []

        out = []
        for cat_val, grp in df.groupby(group_col):
            total = len(grp)
            cnt = grp['Strategy'].value_counts()
            if cnt.empty:
                continue
            prop = (cnt / total).to_dict()
            top = cnt.index.tolist()[:5]
            top_props = [float(cnt[s] / total) for s in top]
            # Tendency strength: max proportion (dominance) and 1 - normalized entropy (concentration)
            probs = cnt / total
            ent = -np.sum(probs * np.log(probs + 1e-10))
            max_ent = np.log(len(probs) + 1e-10)
            norm_ent = ent / max_ent if max_ent > 0 else 0
            concentration = float(1 - norm_ent)  # 1 = all one strategy, 0 = uniform
            dominance = float(probs.max())

            out.append({
                'category': cat_val,
                'total_turns': int(total),
                'strategy_counts': cnt.to_dict(),
                'strategy_proportions': {k: float(v) for k, v in prop.items()},
                'top_strategies': top,
                'top_proportions': top_props,
                'tendency_strength': {
                    'dominance': dominance,
                    'concentration': concentration,
                }
            })
        return out

    def strategy_preference_by_bigfive(self, use_outcome='Success'):
        """Strategy preference per Big Five (separate from Style). Returns list of dicts."""
        return self._strategy_preference_for_category('BigFive', use_outcome=use_outcome)

    def strategy_preference_by_style(self, use_outcome='Success'):
        """Strategy preference per Decision-Making Style. Returns list of dicts."""
        return self._strategy_preference_for_category('Style', use_outcome=use_outcome)

    def strategy_preference_bigfive_dataframe(self, use_outcome='Success'):
        """Return a DataFrame: BigFive, Strategy, Count, Proportion, Dominance, Concentration (per row per BigFive strategy)."""
        prefs = self.strategy_preference_by_bigfive(use_outcome=use_outcome)
        rows = []
        for p in prefs:
            bf = p['category']
            dom = p['tendency_strength']['dominance']
            conc = p['tendency_strength']['concentration']
            for strat, prop in p['strategy_proportions'].items():
                rows.append({
                    'BigFive': bf,
                    'Strategy': strat,
                    'Count': p['strategy_counts'].get(strat, 0),
                    'Proportion': prop,
                    'Dominance': dom,
                    'Concentration': conc,
                })
        return pd.DataFrame(rows)

    def strategy_preference_style_dataframe(self, use_outcome='Success'):
        """Return a DataFrame: Style, Strategy, Count, Proportion, Dominance, Concentration."""
        prefs = self.strategy_preference_by_style(use_outcome=use_outcome)
        rows = []
        for p in prefs:
            st = p['category']
            dom = p['tendency_strength']['dominance']
            conc = p['tendency_strength']['concentration']
            for strat, prop in p['strategy_proportions'].items():
                rows.append({
                    'Style': st,
                    'Strategy': strat,
                    'Count': p['strategy_counts'].get(strat, 0),
                    'Proportion': prop,
                    'Dominance': dom,
                    'Concentration': conc,
                })
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # 2. Per-turn strategy tendency (each turn has a preferred strategy?)
    # -------------------------------------------------------------------------

    def strategy_by_turn(self):
        """
        For each Turn, strategy distribution (count and proportion) and dominant strategy.
        Returns: dict with turn_stats (list), and pivot table for heatmap (counts).
        """
        df = self.df_turns[self.df_turns['Strategy'].notna() & (self.df_turns['Strategy'].astype(str).str.strip() != '')]
        if df.empty:
            return {'turn_stats': [], 'count_matrix': None, 'proportion_matrix': None}

        turn_stats = []
        for turn, grp in df.groupby('Turn'):
            total = len(grp)
            cnt = grp['Strategy'].value_counts()
            prop = cnt / total
            dominant = cnt.index[0]
            dominant_prop = float(prop.iloc[0])
            turn_stats.append({
                'Turn': int(turn),
                'total_turns': int(total),
                'strategy_counts': cnt.to_dict(),
                'strategy_proportions': {k: float(v) for k, v in (cnt / total).items()},
                'dominant_strategy': dominant,
                'dominant_proportion': dominant_prop,
            })

        count_matrix = df.pivot_table(index='Turn', columns='Strategy', values='User', aggfunc='count').fillna(0)
        proportion_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
        return {
            'turn_stats': turn_stats,
            'count_matrix': count_matrix,
            'proportion_matrix': proportion_matrix,
        }

    # -------------------------------------------------------------------------
    # 3. Same-turn branch: parent vs branch strategy; re-selection proportion and correlation
    # -------------------------------------------------------------------------

    def same_turn_branch_strategy_analysis(self):
        """
        For each branching event (parent at Turn T, branch at Turn T): parent strategy vs branch strategy.
        - same_vs_different: count of branches that chose same strategy vs different.
        - re_selection_sufficient_ratio: proportion of branches that chose a *different* strategy (re-selection).
        - parent_branch_crosstab: contingency table (parent strategy x branch strategy).
        """
        if self.df_meta is None:
            return {
                'same_count': 0, 'different_count': 0, 're_selection_sufficient_ratio': 0.0,
                'parent_branch_crosstab': None, 'by_turn': [], 'details': []
            }

        branches = self.df_meta[~self.df_meta['IsRoot']].copy()
        if branches.empty:
            return {
                'same_count': 0, 'different_count': 0, 're_selection_sufficient_ratio': 0.0,
                'parent_branch_crosstab': None, 'by_turn': [], 'details': []
            }

        details = []
        for _, row in branches.iterrows():
            traj_id = row['TrajID']
            parent_id = row['ParentID']
            branch_at = row['BranchTurn']
            turn_at = branch_at + 1  # human round
            parent_turns = self.df_turns[self.df_turns['TrajID'] == parent_id]
            branch_turns = self.df_turns[self.df_turns['TrajID'] == traj_id]
            p_at = parent_turns[parent_turns['Turn'] == turn_at]
            b_at = branch_turns[branch_turns['Turn'] == turn_at]
            if p_at.empty or b_at.empty:
                continue
            parent_strat = p_at.iloc[0]['Strategy']
            branch_strat = b_at.iloc[0]['Strategy']
            same = 1 if parent_strat == branch_strat else 0
            details.append({
                'TrajID': traj_id, 'ParentID': parent_id, 'Turn': int(turn_at),
                'parent_strategy': parent_strat, 'branch_strategy': branch_strat,
                'same': same,
            })

        if not details:
            return {
                'same_count': 0, 'different_count': 0, 're_selection_sufficient_ratio': 0.0,
                'parent_branch_crosstab': None, 'by_turn': [], 'details': details
            }

        df_d = pd.DataFrame(details)
        same_count = int((df_d['same'] == 1).sum())
        different_count = int((df_d['same'] == 0).sum())
        total = len(df_d)
        re_selection_sufficient_ratio = different_count / total if total > 0 else 0.0

        crosstab = pd.crosstab(df_d['parent_strategy'], df_d['branch_strategy'])
        by_turn = df_d.groupby('Turn').agg(
            same_count=('same', 'sum'),
            total=('same', 'count')
        ).reset_index()
        by_turn['different_count'] = by_turn['total'] - by_turn['same_count']
        by_turn['re_selection_ratio'] = by_turn['different_count'] / by_turn['total'].replace(0, np.nan)

        return {
            'same_count': same_count,
            'different_count': different_count,
            'total_branches': total,
            're_selection_sufficient_ratio': float(re_selection_sufficient_ratio),
            'parent_branch_crosstab': crosstab,
            'by_turn': by_turn.to_dict('records'),
            'details': details[:500],
        }

    # -------------------------------------------------------------------------
    # 4. Overall strategy statistics
    # -------------------------------------------------------------------------

    def overall_strategy_stats(self):
        """
        Global strategy counts, by outcome, and summary table.
        """
        df = self.df_turns.copy()
        df = df[df['Strategy'].notna() & (df['Strategy'].astype(str).str.strip() != '')]

        total_count = len(df)
        global_counts = df['Strategy'].value_counts()
        global_proportions = (global_counts / total_count).to_dict()

        by_outcome = df.groupby('Outcome')['Strategy'].value_counts().unstack(fill_value=0)
        by_outcome_prop = by_outcome.div(by_outcome.sum(axis=1), axis=0)

        summary = {
            'total_turns': int(total_count),
            'unique_strategies': int(global_counts.shape[0]),
            'strategy_counts': global_counts.to_dict(),
            'strategy_proportions': {k: float(v) for k, v in global_proportions.items()},
            'top_5_strategies': global_counts.head(5).index.tolist(),
            'top_5_counts': global_counts.head(5).tolist(),
            'by_outcome_counts': by_outcome.to_dict() if not by_outcome.empty else {},
            'by_outcome_proportions': by_outcome_prop.to_dict() if not by_outcome_prop.empty else {},
        }
        return summary

    # -------------------------------------------------------------------------
    # 5. Other: strategy vs outcome (success rate per strategy), transition matrix (optional)
    # -------------------------------------------------------------------------

    def strategy_outcome_success_rate(self):
        """Per-strategy success rate (at turn level: proportion of turns that belong to success trajectories)."""
        df = self.df_turns.copy()
        df = df[df['Strategy'].notna() & (df['Strategy'].astype(str).str.strip() != '')]
        if df.empty:
            return pd.DataFrame()
        grp = df.groupby('Strategy')['Outcome'].agg(['count', lambda x: (x == 'Success').sum()])
        grp.columns = ['total_turns', 'success_turns']
        grp['success_rate'] = grp['success_turns'] / grp['total_turns']
        return grp.reset_index()

    def strategy_transition_matrix(self):
        """Within trajectory: from Turn t to t+1, strategy transition counts (optional extra)."""
        df = self.df_turns.sort_values(['TrajID', 'Turn'])
        df = df[df['Strategy'].notna() & (df['Strategy'].astype(str).str.strip() != '')]
        transitions = []
        for _, g in df.groupby('TrajID'):
            g = g.sort_values('Turn')
            for i in range(len(g) - 1):
                from_s = g.iloc[i]['Strategy']
                to_s = g.iloc[i + 1]['Strategy']
                transitions.append({'from': from_s, 'to': to_s})
        if not transitions:
            return None
        trans_df = pd.DataFrame(transitions)
        return pd.crosstab(trans_df['from'], trans_df['to'])

    # -------------------------------------------------------------------------
    # Radar: strategy set comparison Success vs Failure (for radar chart)
    # -------------------------------------------------------------------------

    def strategy_proportions_by_outcome(self):
        """
        Strategy proportions by Outcome (Success vs Fail) for radar chart.
        Returns: dict with 'strategies' (list), 'Success' (list of proportions), 'Fail' (list of proportions).
        """
        df = self.df_turns.copy()
        df = df[df['Strategy'].notna() & (df['Strategy'].astype(str).str.strip() != '')]
        if df.empty:
            return {'strategies': [], 'Success': [], 'Fail': []}
        all_strategies = df['Strategy'].unique().tolist()
        out = {'strategies': all_strategies, 'Success': [], 'Fail': []}
        for outcome in ['Success', 'Fail']:
            sub = df[df['Outcome'] == outcome]
            total = len(sub)
            if total == 0:
                out[outcome] = [0.0] * len(all_strategies)
                continue
            cnt = sub['Strategy'].value_counts()
            props = [float(cnt.get(s, 0) / total) for s in all_strategies]
            out[outcome] = props
        return out

    # -------------------------------------------------------------------------
    # Legacy (keep for backward compatibility)
    # -------------------------------------------------------------------------

    def analyze_fingerprint(self):
        """Strategy fingerprint by BigFive (success only)."""
        success_turns = self.df_turns[self.df_turns['Outcome'] == 'Success']
        fingerprint = success_turns.groupby(['BigFive', 'Strategy']).size().reset_index(name='Count')
        totals = fingerprint.groupby('BigFive')['Count'].transform('sum')
        fingerprint['Frequency'] = fingerprint['Count'] / totals
        return fingerprint

    def analyze_diversity(self):
        """Strategy diversity per trajectory."""
        grouped = self.df_turns.groupby(['User', 'TrajID'])
        diversity = []
        for name, group in grouped:
            user, traj_id = name
            strategies = group['Strategy'].unique()
            score = len(strategies) / len(group)
            diversity.append({
                "User": user, "TrajID": traj_id,
                "Outcome": group.iloc[0]['Outcome'],
                "DiversityScore": score
            })
        df_div = pd.DataFrame(diversity)
        user_map = self.df_turns[['User', 'BigFive']].drop_duplicates().set_index('User')
        df_div = df_div.join(user_map, on='User')
        return df_div
