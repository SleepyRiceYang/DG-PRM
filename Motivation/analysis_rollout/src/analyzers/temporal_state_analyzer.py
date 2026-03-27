# -*- coding: utf-8 -*-
"""
动态时序与状态演变分析器
- 对比组：成功轨迹群 vs 失败轨迹群
- 四指标：Hs, Ha, Dissonance, Trend
- 三层：总体 / 人格决策类 / 个人
"""
import pandas as pd
import numpy as np


class TemporalStateAnalyzer:
    """为动态时序与状态演变提供三层聚合数据（成功 vs 失败）。"""

    METRIC_COLS = ['Hs', 'Ha', 'Dissonance', 'Trend']

    def __init__(self, df_metrics, df_meta, df_personas, traj_lookup=None):
        """
        df_metrics: 带 Turn, User, TrajID, Outcome, Hs, Ha, Dissonance, Trend
        df_meta: 带 User, TrajID, IsRoot, Success 等
        df_personas: 带 User -> BigFive, Style
        traj_lookup: 原始轨迹查找表（可选，用于提取 turn 文本/token entropy 明细）
        """
        self.df_metrics = df_metrics.copy()
        self.df_meta = df_meta.copy()
        self.df_personas = df_personas.copy()
        self.traj_lookup = traj_lookup or {}
        self._attach_persona()

    def _attach_persona(self):
        """在 df_metrics 上附加 BigFive, Style（按 TrajID 从 meta 取）。"""
        meta = self.df_meta.join(self.df_personas, on='User')
        traj_persona = meta[['TrajID', 'BigFive', 'Style']].drop_duplicates()
        self.df_metrics = self.df_metrics.merge(traj_persona, on='TrajID', how='left')
        self._meta_with_persona = meta

    # ---------- 1. 总体层次 ----------
    def get_overall_trend_by_outcome(self):
        """
        总体层次：按 Outcome 聚合，得到每个 Turn 上四指标的均值±标准差。
        Returns: dict Outcome -> DataFrame columns [Turn, Hs, Ha, Dissonance, Trend, *_std]
        """
        out = {}
        for outcome in ['Success', 'Fail']:
            sub = self.df_metrics[self.df_metrics['Outcome'] == outcome]
            if sub.empty:
                out[outcome] = pd.DataFrame()
                continue
            agg = sub.groupby('Turn')[self.METRIC_COLS].agg(['mean', 'std']).reset_index()
            agg.columns = ['Turn'] + [f'{c}_mean' for c in self.METRIC_COLS] + [f'{c}_std' for c in self.METRIC_COLS]
            out[outcome] = agg
        return out

    # ---------- 2. Persona level: Big Five and Decision Style (separate) ----------
    def get_bigfive_categories(self):
        """Unique Big Five personality values, excluding Unknown."""
        cats = self.df_metrics['BigFive'].dropna().unique().tolist()
        return [c for c in cats if str(c).strip() != 'Unknown']

    def get_style_categories(self):
        """Unique decision-making style values, excluding Unknown."""
        cats = self.df_metrics['Style'].dropna().unique().tolist()
        return [c for c in cats if str(c).strip() != 'Unknown']

    def get_trend_by_outcome_for_bigfive(self, big_five):
        """
        Trend by outcome for one Big Five category (all styles combined).
        Returns: dict Outcome -> DataFrame (Turn, four metrics mean/std)
        """
        sub = self.df_metrics[self.df_metrics['BigFive'] == big_five]
        out = {}
        for outcome in ['Success', 'Fail']:
            s = sub[sub['Outcome'] == outcome]
            if s.empty:
                out[outcome] = pd.DataFrame()
                continue
            agg = s.groupby('Turn')[self.METRIC_COLS].agg(['mean', 'std']).reset_index()
            agg.columns = ['Turn'] + [f'{c}_mean' for c in self.METRIC_COLS] + [f'{c}_std' for c in self.METRIC_COLS]
            out[outcome] = agg
        return out

    def get_trend_by_outcome_for_style(self, style):
        """
        Trend by outcome for one decision-making style (all Big Five combined).
        Returns: dict Outcome -> DataFrame (Turn, four metrics mean/std)
        """
        sub = self.df_metrics[self.df_metrics['Style'] == style]
        out = {}
        for outcome in ['Success', 'Fail']:
            s = sub[sub['Outcome'] == outcome]
            if s.empty:
                out[outcome] = pd.DataFrame()
                continue
            agg = s.groupby('Turn')[self.METRIC_COLS].agg(['mean', 'std']).reset_index()
            agg.columns = ['Turn'] + [f'{c}_mean' for c in self.METRIC_COLS] + [f'{c}_std' for c in self.METRIC_COLS]
            out[outcome] = agg
        return out

    # ---------- 3. 个人层次 ----------
    def get_user_list(self):
        """所有用户列表（与 df_meta 一致）。"""
        return self.df_meta['User'].unique().tolist()

    # ---------- 4. 轨迹层次 ----------
    def get_trajectory_list(self):
        """所有轨迹元信息（用于 trajectory-level PDF 分页）。"""
        meta = self._meta_with_persona.copy()
        cols = ['User', 'TrajID', 'ParentID', 'BranchTurn', 'IsRoot', 'Success', 'Length', 'BigFive', 'Style']
        for c in cols:
            if c not in meta.columns:
                meta[c] = np.nan
        return meta[cols].drop_duplicates(subset=['TrajID']).sort_values(['User', 'IsRoot', 'TrajID'], ascending=[True, False, True]).reset_index(drop=True)

    def get_user_trend_by_outcome(self, user):
        """某一用户下，按 Outcome 聚合的时序数据（该用户可能既有成功也有失败轨迹）。"""
        sub = self.df_metrics[self.df_metrics['User'] == user]
        out = {}
        for outcome in ['Success', 'Fail']:
            s = sub[sub['Outcome'] == outcome]
            if s.empty:
                out[outcome] = pd.DataFrame()
                continue
            agg = s.groupby('Turn')[self.METRIC_COLS].agg(['mean', 'std']).reset_index()
            agg.columns = ['Turn'] + [f'{c}_mean' for c in self.METRIC_COLS] + [f'{c}_std' for c in self.METRIC_COLS]
            out[outcome] = agg
        return out

    def get_user_summary_for_display(self, df_user_stats):
        """
        Build per-user summary for PDF (English keys to avoid encoding issues).
        """
        rows = []
        for _, r in df_user_stats.iterrows():
            rows.append({
                'User': r['User'],
                'BigFive': r['BigFive'],
                'Style': r['Style'],
                'Total_Trajs': int(r['Total_Trajs']),
                'Success_Trajs': int(r['Success_Trajs']),
                'Trajectory_Success_Rate': r['Success_Traj_Ratio'],
                'Initial_Success': 'Yes' if r['Root_Success'] == 1 else 'No',
                'Rescued': 'Yes' if r['Is_Rescued'] == 1 else 'No',
                'Needs_Rescue': 'Yes' if r['Needs_Rescue'] == 1 else 'No',
            })
        return rows

    def get_trajectory_trend(self, traj_id):
        """
        某条轨迹的逐轮时序（最细粒度，不按 outcome 聚合）。
        返回列包含：Turn, Hs, Ha, Delta_Hs, Delta_Ha（若存在）
        """
        sub = self.df_metrics[self.df_metrics['TrajID'] == traj_id].copy()
        if sub.empty:
            return pd.DataFrame()
        cols = ['Turn', 'Hs', 'Ha', 'Delta_Hs', 'Delta_Ha']
        for c in cols:
            if c not in sub.columns:
                sub[c] = np.nan
        return sub[cols].sort_values('Turn').reset_index(drop=True)

    def get_trajectory_feature_table(self, df_traj_stats):
        """
        为 trajectory-level 代表性筛选提供轨迹级聚合特征。
        输出字段：Hs_mean, Ha_mean, Delta_Hs_abs_mean, Delta_Ha_abs_mean, Dissonance_mean
        """
        if df_traj_stats is None or df_traj_stats.empty:
            return pd.DataFrame()

        req_cols = ['TrajID', 'Hs', 'Ha', 'Delta_Hs', 'Delta_Ha', 'Dissonance']
        m = self.df_metrics.copy()
        for c in req_cols:
            if c not in m.columns:
                m[c] = np.nan

        feat = m.groupby('TrajID').agg(
            Hs_mean=('Hs', 'mean'),
            Ha_mean=('Ha', 'mean'),
            Delta_Hs_abs_mean=('Delta_Hs', lambda s: np.nanmean(np.abs(s.values)) if len(s) > 0 else np.nan),
            Delta_Ha_abs_mean=('Delta_Ha', lambda s: np.nanmean(np.abs(s.values)) if len(s) > 0 else np.nan),
            Dissonance_mean=('Dissonance', 'mean'),
        ).reset_index()

        merged = df_traj_stats.merge(feat, on='TrajID', how='left')
        for c in ['Hs_mean', 'Ha_mean', 'Delta_Hs_abs_mean', 'Delta_Ha_abs_mean', 'Dissonance_mean']:
            if c not in merged.columns:
                merged[c] = np.nan
        return merged

    def get_trajectory_summary_for_display(self, df_traj_stats):
        """
        Build per-trajectory summary for PDF card display.
        """
        rows = []
        for _, r in df_traj_stats.iterrows():
            rows.append({
                'TrajID': r.get('TrajID'),
                'User': r.get('User'),
                'BigFive': r.get('BigFive', 'Unknown'),
                'Style': r.get('Style', 'Unknown'),
                'IsRoot': bool(r.get('IsRoot', False)),
                'ParentID': r.get('ParentID', None),
                'BranchTurn': r.get('BranchTurn', -1),
                'Length': int(r.get('Length', 0)) if pd.notna(r.get('Length', np.nan)) else 0,
                'Root_Success': 'Yes' if bool(r.get('Root_Success', False)) else 'No',
                'Intervened_Success': 'Yes' if bool(r.get('Intervened_Success', False)) else 'No',
                'Outcome': r.get('Outcome', 'Unknown')
            })
        return rows

    def _normalize_token_metrics(self, metrics):
        """把不同格式的 token metrics 统一为 [{'token','entropy','top_5','segment'}]。"""
        if not metrics:
            return []

        out = []
        for m in metrics:
            if not isinstance(m, dict):
                continue
            token = m.get('token', m.get('text', ''))
            ent = m.get('entropy', m.get('ent', m.get('token_entropy', None)))
            top5 = m.get('top_5', m.get('top5', m.get('candidates', [])))
            segment = m.get('segment', m.get('segment_name', m.get('token_type', None)))
            if token is None:
                token = ''
            try:
                ent = None if ent is None else float(ent)
            except Exception:
                ent = None
            out.append({'token': str(token), 'entropy': ent, 'top_5': top5 if isinstance(top5, list) else [], 'segment': segment})
        return out

    def _extract_turn_metrics_from_lookup(self, raw_traj, turn):
        """优先从 *_metrics.json 挂载的 turn_metrics 提取；失败再回退到 raw turn 中的 metrics。"""
        if not isinstance(raw_traj, dict):
            return [], 'traj_missing'

        turn_metrics_map = raw_traj.get('turn_metrics', {}) or {}

        # direct by turn / round
        direct = turn_metrics_map.get(turn, None)
        if direct is None:
            direct = turn_metrics_map.get(str(turn), None)
        if direct is not None:
            tokens = self._normalize_token_metrics(direct)
            if tokens:
                return tokens, 'metrics_file'

        # unknown-turn bucket
        unknown_tokens = turn_metrics_map.get(None, [])
        unknown_tokens = self._normalize_token_metrics(unknown_tokens)
        if unknown_tokens:
            return unknown_tokens, 'metrics_file_unknown_turn'

        # fallback from raw turn itself
        raw_turns = raw_traj.get('turns', [])
        for rt in raw_turns:
            if rt.get('role') != 'Persuader':
                continue
            if rt.get('round') == turn:
                tokens = self._normalize_token_metrics(rt.get('metrics', []))
                if tokens:
                    return tokens, 'rollout_turn_metrics'
                return [], 'raw_turn_metrics_missing'

        return [], 'turn_not_matched'

    def _segment_tokens_from_metrics(self, metrics):
        """segment 优先用结构化字段；其次标签切分；至少保证 all 可用。"""
        if not metrics:
            return {'state_analysis': [], 'strategy': [], 'response': [], 'all': []}, 'no_metrics'

        all_tokens = self._normalize_token_metrics(metrics)
        if not all_tokens:
            return {'state_analysis': [], 'strategy': [], 'response': [], 'all': []}, 'no_valid_token'

        # 优先：结构化 segment 字段
        state = [t for t in all_tokens if str(t.get('segment', '')).lower() in {'state_analysis', 'state', 'reason', 'analysis'}]
        strategy = [t for t in all_tokens if str(t.get('segment', '')).lower() in {'strategy'}]
        response = [t for t in all_tokens if str(t.get('segment', '')).lower() in {'response', 'action', 'reply'}]
        if state or strategy or response:
            return {'state_analysis': state, 'strategy': strategy, 'response': response, 'all': all_tokens}, 'structured_segment'

        # 次优：标签切分
        text = ''.join([t['token'] for t in all_tokens])
        def _find_span(start_tag, end_tag):
            s = text.find(start_tag)
            e = text.find(end_tag)
            if s == -1 or e == -1 or e <= s:
                return None
            return (s + len(start_tag), e)

        spans = {
            'state_analysis': _find_span('<state_analysis>', '</state_analysis>'),
            'strategy': _find_span('<strategy>', '</strategy>'),
            'response': _find_span('<response>', '</response>'),
        }

        boundaries = []
        cur = 0
        for i, t in enumerate(all_tokens):
            nxt = cur + len(t['token'])
            boundaries.append((i, cur, nxt))
            cur = nxt

        out = {'state_analysis': [], 'strategy': [], 'response': [], 'all': all_tokens}
        for key, span in spans.items():
            if span is None:
                continue
            start_char, end_char = span
            seg = []
            for i, l, r in boundaries:
                if r <= start_char:
                    continue
                if l >= end_char:
                    break
                seg.append(all_tokens[i])
            out[key] = seg

        if out['state_analysis'] or out['response'] or out['strategy']:
            return out, 'tag_segment'
        return out, 'all_only'

    def _compute_segment_quality(self, tokens, high_entropy_threshold=1.2):
        ents = [x.get('entropy') for x in tokens if x.get('entropy') is not None]
        if not ents:
            return {'avg_entropy': None, 'max_entropy': None, 'high_entropy_ratio': None, 'quality_tag': 'unknown'}
        arr = np.array(ents, dtype=float)
        avg_e = float(np.mean(arr))
        max_e = float(np.max(arr))
        high_ratio = float(np.mean(arr > high_entropy_threshold))
        if avg_e < 0.8 and high_ratio < 0.2:
            tag = 'stable'
        elif avg_e < 1.2 and high_ratio < 0.35:
            tag = 'converging'
        elif avg_e < 1.8:
            tag = 'exploratory'
        else:
            tag = 'risky'
        return {'avg_entropy': avg_e, 'max_entropy': max_e, 'high_entropy_ratio': high_ratio, 'quality_tag': tag}

    def _get_parent_chain(self, traj_id):
        """返回从 root 到当前 traj 的链（root -> ... -> traj）。"""
        chain = []
        seen = set()
        cur = traj_id
        while cur and cur not in seen:
            seen.add(cur)
            traj = self.traj_lookup.get(cur)
            if not isinstance(traj, dict):
                break
            chain.append(cur)
            cur = traj.get('parent_id')
        return list(reversed(chain))

    def _reconstruct_full_turns(self, traj_id):
        """按 parent chain 重建完整 turns，并标注每个 turn 的来源 traj。"""
        chain = self._get_parent_chain(traj_id)
        if not chain:
            return [], []

        full_turns = []
        source_map = []

        for idx, tid in enumerate(chain):
            traj = self.traj_lookup.get(tid, {})
            local_turns = traj.get('turns', []) if isinstance(traj, dict) else []
            if idx == 0:
                full_turns = list(local_turns)
                source_map = [tid] * len(full_turns)
                continue

            branch_turn = traj.get('branch_at_turn')
            if branch_turn is None:
                full_turns = list(local_turns)
                source_map = [tid] * len(full_turns)
                continue

            # parent prefix up to split system turn (excluded)
            split_idx = None
            for i, t in enumerate(full_turns):
                if t.get('role') == 'Persuader' and t.get('round') == branch_turn:
                    split_idx = i
                    break
            if split_idx is None:
                split_idx = len(full_turns)

            parent_prefix = full_turns[:split_idx]
            parent_sources = source_map[:split_idx]

            child_has_history = any(
                t.get('role') == 'Persuader' and t.get('round', 0) < branch_turn
                for t in local_turns
            )
            if child_has_history:
                full_turns = list(local_turns)
                source_map = [tid] * len(full_turns)
            else:
                full_turns = parent_prefix + list(local_turns)
                source_map = parent_sources + [tid] * len(local_turns)

        return full_turns, source_map

    def _build_turn_metrics_from_reconstructed(self, reconstructed_turns):
        """从重建后的 Persuader turns 计算 turn-level metrics，用于完整轨迹视角。"""
        pers = [t for t in reconstructed_turns if t.get('role') == 'Persuader']
        rows = []
        for t in pers:
            turn = t.get('round')
            hs = t.get('hs')
            ha = t.get('ha')
            if hs is None:
                hs = np.nan
            if ha is None:
                ha = np.nan
            rows.append({'Turn': turn, 'Hs': hs, 'Ha': ha})

        rows = sorted(rows, key=lambda x: x['Turn'])
        prev_hs, prev_ha = None, None
        for r in rows:
            r['Dissonance'] = (r['Hs'] - r['Ha']) if pd.notna(r['Hs']) and pd.notna(r['Ha']) else np.nan
            if prev_hs is None or pd.isna(prev_hs) or pd.isna(r['Hs']):
                r['Delta_Hs'] = np.nan
            else:
                r['Delta_Hs'] = r['Hs'] - prev_hs
            if prev_ha is None or pd.isna(prev_ha) or pd.isna(r['Ha']):
                r['Delta_Ha'] = np.nan
            else:
                r['Delta_Ha'] = r['Ha'] - prev_ha
            prev_hs = r['Hs']
            prev_ha = r['Ha']
        return rows

    def build_representative_trajectory_heatmap_payload(self, df_selected_traj):
        """构建代表性轨迹 heatmap payload，并返回调试统计。"""
        if df_selected_traj is None or df_selected_traj.empty:
            return {'trajectories': []}, {'summary': {}, 'samples': []}

        payload = {'trajectories': []}
        debug_samples = []
        recon_debug = []
        counters = {
            'total_turns': 0,
            'matched_turns': 0,
            'turns_with_tokens': 0,
            'turns_with_entropy': 0,
            'turns_with_state_segment': 0,
            'turns_with_strategy_segment': 0,
            'turns_with_response_segment': 0,
            'turns_with_strategy': 0,
            'turns_missing_strategy': 0,
            'turns_with_inherited_metrics': 0,
            'turns_with_local_metrics': 0,
            'turns_all_only': 0,
            'reasons': {},
        }

        for _, row in df_selected_traj.iterrows():
            traj_id = row.get('TrajID')
            tmeta = {
                'TrajID': traj_id,
                'User': row.get('User'),
                'BigFive': row.get('BigFive', 'Unknown'),
                'Style': row.get('Style', 'Unknown'),
                'Quadrant': row.get('Quadrant', 'N/A'),
                'Root_Success': bool(row.get('Root_Success', False)),
                'Current_Success': bool(row.get('Intervened_Success', False)),
                'Representative_Tag': row.get('Representative_Tag', 'BucketSelected'),
                'Hs_mean': float(row.get('Hs_mean', np.nan)) if pd.notna(row.get('Hs_mean', np.nan)) else None,
                'Ha_mean': float(row.get('Ha_mean', np.nan)) if pd.notna(row.get('Ha_mean', np.nan)) else None,
                'Dissonance_mean': float(row.get('Dissonance_mean', np.nan)) if pd.notna(row.get('Dissonance_mean', np.nan)) else None,
                'Length': int(row.get('Length', 0)) if pd.notna(row.get('Length', np.nan)) else 0,
                'Outcome': row.get('Outcome', 'Unknown'),
            }

            turns_payload = []
            raw_traj = self.traj_lookup.get(traj_id, {})

            reconstructed_turns, source_trajs = self._reconstruct_full_turns(traj_id)
            recon_turn_rows = self._build_turn_metrics_from_reconstructed(reconstructed_turns)
            recon_by_round = {r['Turn']: r for r in recon_turn_rows}

            pers_turns = [rt for rt in reconstructed_turns if rt.get('role') == 'Persuader']
            raw_by_round = {rt.get('round'): rt for rt in pers_turns}

            # reconstruction debug
            if len(recon_debug) < 50:
                chain = self._get_parent_chain(traj_id)
                rd = {
                    'TrajID': traj_id,
                    'ParentChain': chain,
                    'RootID': chain[0] if chain else None,
                    'BranchTurn': raw_traj.get('branch_at_turn') if isinstance(raw_traj, dict) else None,
                    'local_turn_count': len(raw_traj.get('turns', [])) if isinstance(raw_traj, dict) else 0,
                    'reconstructed_turn_count': len(reconstructed_turns),
                    'inherited_turn_count': int(sum(1 for s in source_trajs if s != traj_id)),
                    'own_turn_count': int(sum(1 for s in source_trajs if s == traj_id)),
                    'reconstructed_turns': []
                }
                for idx_t, rt in enumerate(reconstructed_turns[:60]):
                    if rt.get('role') != 'Persuader':
                        continue
                    source_tid = source_trajs[idx_t] if idx_t < len(source_trajs) else None
                    tok, _rs = self._extract_turn_metrics_from_lookup(self.traj_lookup.get(source_tid, {}), rt.get('round'))
                    seg, _ = self._segment_tokens_from_metrics(tok)
                    rd['reconstructed_turns'].append({
                        'Turn': rt.get('round'),
                        'source_traj': source_tid,
                        'source_type': 'local' if source_tid == traj_id else 'inherited',
                        'has_text': bool(str(rt.get('content', '')).strip()),
                        'has_metrics': len(tok) > 0,
                        'has_tokens': len(seg.get('all', [])) > 0,
                        'available_segments': [k for k in ['state_analysis', 'strategy', 'response', 'all'] if len(seg.get(k, [])) > 0],
                    })
                recon_debug.append(rd)

            for tr in recon_turn_rows:
                counters['total_turns'] += 1
                turn = int(tr['Turn']) if pd.notna(tr.get('Turn', np.nan)) else None
                raw = raw_by_round.get(turn, {})

                source_tid = None
                for idx_t, t in enumerate(reconstructed_turns):
                    if t.get('role') == 'Persuader' and t.get('round') == turn:
                        source_tid = source_trajs[idx_t] if idx_t < len(source_trajs) else traj_id
                        break
                source_traj = self.traj_lookup.get(source_tid, raw_traj)

                tokens, source_reason = self._extract_turn_metrics_from_lookup(source_traj, turn)
                seg_tokens, seg_reason = self._segment_tokens_from_metrics(tokens)
                counters['reasons'][source_reason] = counters['reasons'].get(source_reason, 0) + 1

                if source_reason not in {'traj_missing', 'turn_not_matched', 'raw_turn_metrics_missing'}:
                    counters['matched_turns'] += 1
                if source_tid == traj_id:
                    counters['turns_with_local_metrics'] += 1
                else:
                    counters['turns_with_inherited_metrics'] += 1
                if len(seg_tokens.get('all', [])) > 0:
                    counters['turns_with_tokens'] += 1
                if any(t.get('entropy') is not None for t in seg_tokens.get('all', [])):
                    counters['turns_with_entropy'] += 1
                if len(seg_tokens.get('state_analysis', [])) > 0:
                    counters['turns_with_state_segment'] += 1
                if len(seg_tokens.get('strategy', [])) > 0:
                    counters['turns_with_strategy_segment'] += 1
                if len(seg_tokens.get('response', [])) > 0:
                    counters['turns_with_response_segment'] += 1
                if seg_reason == 'all_only':
                    counters['turns_all_only'] += 1

                text_candidates = [
                    raw.get('response_text', ''),
                    raw.get('content', ''),
                    raw.get('text', ''),
                    raw.get('utterance', ''),
                ]
                raw_text = next((t for t in text_candidates if isinstance(t, str) and t.strip()), '')

                state_stats = self._compute_segment_quality(seg_tokens.get('state_analysis', []))
                strategy_stats = self._compute_segment_quality(seg_tokens.get('strategy', []))
                response_stats = self._compute_segment_quality(seg_tokens.get('response', []))
                all_stats = self._compute_segment_quality(seg_tokens.get('all', []))

                dis = float(tr.get('Dissonance', 0.0)) if pd.notna(tr.get('Dissonance', np.nan)) else 0.0
                dhs = float(tr.get('Delta_Hs', 0.0)) if pd.notna(tr.get('Delta_Hs', np.nan)) else 0.0
                dha = float(tr.get('Delta_Ha', 0.0)) if pd.notna(tr.get('Delta_Ha', np.nan)) else 0.0
                if abs(dis) >= 1.5 or all_stats['quality_tag'] == 'risky':
                    risk_tag = 'delusional'
                elif dhs < 0 and dha < 0:
                    risk_tag = 'converging'
                elif dhs > 0 or dha > 0:
                    risk_tag = 'exploratory'
                else:
                    risk_tag = 'stable'

                selected_strategy = raw.get('strategy_name', tr.get('Strategy', 'Unknown')) if isinstance(raw, dict) else tr.get('Strategy', 'Unknown')
                if selected_strategy and str(selected_strategy).strip() and str(selected_strategy).strip().lower() != 'unknown':
                    counters['turns_with_strategy'] += 1
                    strategy_source = 'turn_field'
                else:
                    counters['turns_missing_strategy'] += 1
                    strategy_source = 'missing_in_turn'

                turns_payload.append({
                    'Turn': turn,
                    'Hs': float(tr.get('Hs', 0.0)) if pd.notna(tr.get('Hs', np.nan)) else None,
                    'Ha': float(tr.get('Ha', 0.0)) if pd.notna(tr.get('Ha', np.nan)) else None,
                    'Delta_Hs': dhs,
                    'Delta_Ha': dha,
                    'Dissonance': dis,
                    'TurnRiskTag': risk_tag,
                    'RawText': raw_text,
                    'SelectedStrategy': selected_strategy,
                    'StrategySource': strategy_source,
                    'Ha_Source': 'strategy',
                    'Segments': {
                        'state_analysis': seg_tokens.get('state_analysis', []),
                        'strategy': seg_tokens.get('strategy', []),
                        'response': seg_tokens.get('response', []),
                        'all': seg_tokens.get('all', []),
                    },
                    'SegmentQuality': {
                        'state_analysis': state_stats,
                        'strategy': strategy_stats,
                        'response': response_stats,
                        'all': all_stats,
                    },
                    'TokenSource': source_reason,
                    'SegmentParseMode': seg_reason,
                    'ReconstructedSourceTraj': source_tid,
                })

                if len(debug_samples) < 80:
                    debug_samples.append({
                        'TrajID': traj_id,
                        'Turn': turn,
                        'RawText': raw_text,
                        'token_count': len(seg_tokens.get('all', [])),
                        'has_entropy': any(t.get('entropy') is not None for t in seg_tokens.get('all', [])),
                        'has_top5': any(len(t.get('top_5', [])) > 0 for t in seg_tokens.get('all', [])),
                        'available_segments': [k for k in ['state_analysis', 'strategy', 'response', 'all'] if len(seg_tokens.get(k, [])) > 0],
                        'state_analysis_token_count': len(seg_tokens.get('state_analysis', [])),
                        'strategy_token_count': len(seg_tokens.get('strategy', [])),
                        'response_token_count': len(seg_tokens.get('response', [])),
                        'all_token_count': len(seg_tokens.get('all', [])),
                        'avg_entropy_all': all_stats.get('avg_entropy'),
                        'max_entropy_all': all_stats.get('max_entropy'),
                        'token_source': source_reason,
                        'segment_mode': seg_reason,
                    })

            tmeta['Turns'] = turns_payload
            payload['trajectories'].append(tmeta)

        debug = {'summary': counters, 'samples': debug_samples, 'reconstruction': recon_debug}
        return payload, debug
