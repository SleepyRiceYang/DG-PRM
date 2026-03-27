import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from math import pi
from matplotlib.backends.backend_pdf import PdfPages
import json

# Dynamic temporal: outcome colors and English labels (avoid PDF encoding issues)
OUTCOME_COLORS = {'Success': '#27ae60', 'Fail': '#e74c3c'}
OUTCOME_LABELS = {'Success': 'Success Trajectories', 'Fail': 'Failure Trajectories'}
METRIC_TITLES = {
    'Hs': 'State Entropy ($H_s$)',
    'Ha': 'Action Entropy ($H_a$)',
    'Dissonance': 'Cognitive Dissonance ($D = H_s - H_a$)',
    'Trend': 'Trend Momentum ($S_{Trend}$)',
}


class Visualizer:
    def __init__(self, output_dir):
        self.base_dir = output_dir
        base_strategy = os.path.join(output_dir, 'figures/strategy')
        self.fig_dirs = {
            'global': os.path.join(output_dir, 'figures/global_stats'),
            'dynamics': os.path.join(output_dir, 'figures/dynamics'),
            'strategy': base_strategy,
            'strategy_preference': os.path.join(base_strategy, 'preference'),
            'strategy_by_turn': os.path.join(base_strategy, 'by_turn'),
            'strategy_branch': os.path.join(base_strategy, 'branch'),
            'strategy_overall': os.path.join(base_strategy, 'overall'),
            'strategy_radar': os.path.join(base_strategy, 'radar'),
            'strategy_outcome': os.path.join(base_strategy, 'outcome'),
            'dynamics_reports': os.path.join(output_dir, 'figures/dynamics/reports'),
            'persona_pdfs': os.path.join(output_dir, 'figures/dynamics/reports/persona'),
            'individual_pdf': os.path.join(output_dir, 'figures/dynamics/reports/individual'),
            'trajectory_pdf': os.path.join(output_dir, 'figures/dynamics/reports/trajectory'),
            'dynamics_unique': os.path.join(output_dir, 'figures/dynamics/unique_segments'),
            # 'branch_audit': os.path.join(output_dir, 'figures/dynamics/branch_audit'),
        }
        for d in self.fig_dirs.values():
            os.makedirs(d, exist_ok=True)

        sns.set_theme(style="whitegrid", font_scale=1.1)
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    # =========================================================
    # 1. 整体分析可视化
    # =========================================================

    def plot_global_summary_bar(self, global_stats):
        metrics = {
            'Init. Success Rate': global_stats['Global_Initial_Success_Rate'],
            'Rescue Rate': global_stats['Global_Rescue_Rate'],
            'Traj. Success Ratio': global_stats['Global_Trajectory_Success_Ratio']
        }

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ['#3498db', '#e67e22', '#2ecc71']

        bars = ax.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8, width=0.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Ratio (0-1)", fontsize=12, fontweight='bold')
        ax.set_title("Global Performance Overview", fontsize=14, fontweight='bold', pad=20)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f'{height:.1%}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

        info_text = (
            f"Total Users: {int(global_stats['Total_Users'])}\n"
            f"Total Trajectories: {int(global_stats['Total_Trajs'])}\n"
            f"Success Trajectories: {int(global_stats['Total_Success_Trajs'])}"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.95, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)

        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['global'], 'global_summary_bar.png'), dpi=300)
        plt.close()

    def plot_persona_metrics_dashboard(self, df_persona):
        if df_persona.empty:
            return

        plot_configs = [
            {
                'metric': 'Initial_Success_Rate',
                'title': 'Initial Success Rate (User Level)',
                'cmap': 'Blues',
                'fmt': '.1%'
            },
            {
                'metric': 'Rescue_Rate',
                'title': 'Rescue Rate (User Level | Failed Roots)',
                'cmap': 'Oranges',
                'fmt': '.1%'
            },
            {
                'metric': 'Trajectory_Success_Ratio',
                'title': 'Trajectory Success Ratio (Global Efficiency)',
                'cmap': 'Greens',
                'fmt': '.1%'
            },
            {
                'metric': 'Avg_Success_Trajs_Per_User',
                'title': 'Avg. Success Trajectories per User',
                'cmap': 'Purples',
                'fmt': '.1f'
            }
        ]

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        axes = axes.flatten()

        for i, config in enumerate(plot_configs):
            ax = axes[i]
            metric = config['metric']
            pivot_data = df_persona.pivot(index='BigFive', columns='Style', values=metric)
            sns.heatmap(pivot_data, annot=True, cmap=config['cmap'], ax=ax,
                        fmt=config['fmt'], annot_kws={"size": 12, "weight": "bold"},
                        cbar_kws={'label': metric.replace('_', ' ')})
            ax.set_title(config['title'], fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel("Decision Style", fontsize=11)
            ax.set_ylabel("Big-Five Personality", fontsize=11)

        plt.suptitle("Comprehensive Performance Matrix by Persona & Style",
                     fontsize=20, y=0.98, fontweight='bold')
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        plt.savefig(os.path.join(self.fig_dirs['global'], 'persona_matrix_dashboard.png'), dpi=300)
        plt.close()

    def plot_user_success_distribution(self, df_user):
        df_sorted = df_user.sort_values(by='Success_Traj_Ratio', ascending=False)
        fig, ax1 = plt.subplots(figsize=(14, 7))

        x = np.arange(len(df_sorted))
        users = df_sorted['User']

        ax1.bar(x, df_sorted['Total_Trajs'], color='gray', alpha=0.3, label='Total Trajectories')
        ax1.bar(x, df_sorted['Success_Trajs'], color='#2ecc71', alpha=0.9, label='Success Trajectories')

        ax1.set_ylabel("Trajectory Count", fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(users, rotation=45, ha='right', fontsize=9)
        ax1.set_title("User-Level Trajectory Efficiency", fontsize=16, fontweight='bold')
        # 与柱状图共享同一 x 轴边界，避免折线点位与刻度视觉错位
        ax1.set_xlim(-0.5, len(df_sorted) - 0.5)

        ax2 = ax1.twinx()
        ax2.plot(x, df_sorted['Success_Traj_Ratio'].to_numpy(), color='#e74c3c', marker='o',
                 linewidth=2, label='Success Ratio')
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylabel("Success Ratio", color='#e74c3c', fontweight='bold')
        ax2.set_ylim(0, 1.1)
        ax2.tick_params(axis='y', labelcolor='#e74c3c')

        for i, ratio in enumerate(df_sorted['Success_Traj_Ratio']):
            ax2.text(i, ratio + 0.05, f"{ratio:.1f}", ha='center',
                     color='#e74c3c', fontsize=9)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['global'], 'user_trajectory_stats.png'), dpi=300)
        plt.close()

    # =========================================================
    # Unique-segment & Deep-mode specific visualizations
    # =========================================================

    def plot_unique_metrics_dashboard(self, trends, fig_dir=None,
                                      title_prefix=None, mode_name=None):
        """
        3x2 面板：六个 unique 段指标的 Success vs Fail 均值±std 时序图。
        trends: {status: {metric: {rounds, mean, std}}}
        """
        if fig_dir is None:
            fig_dir = self.fig_dirs['dynamics_unique']
        os.makedirs(fig_dir, exist_ok=True)

        metrics = ['hs', 'ha', 'delta_hs', 'delta_ha', 'z_hs', 'z_ha']
        pretty_names = {
            'hs': 'State Entropy $H_s$',
            'ha': 'Action Entropy $H_a$',
            'delta_hs': '$\\Delta H_s$ (Change in State Entropy)',
            'delta_ha': '$\\Delta H_a$ (Change in Action Entropy)',
            'z_hs': 'Z-Score of $H_s$',
            'z_ha': 'Z-Score of $H_a$',
        }

        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        colors = {'success': '#27ae60', 'fail': '#e74c3c'}

        for i, m in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            for status in ['success', 'fail']:
                d = trends.get(status, {}).get(m, {})
                rounds = d.get('rounds', [])
                if not rounds:
                    continue
                mu = np.array(d.get('mean', []))
                std = np.array(d.get('std', []))
                label = 'Success' if status == 'success' else 'Fail'
                ax.plot(rounds, mu, label=label,
                        color=colors[status], marker='o', lw=2)
                if std.size and not np.all(np.isnan(std)):
                    ax.fill_between(rounds, mu - std, mu + std,
                                    color=colors[status], alpha=0.15)
            ax.set_title(pretty_names.get(m, m), fontsize=13, fontweight='bold')
            ax.set_xlabel('Round (Unique Segment)', fontsize=11)
            ax.set_ylabel('Metric Value', fontsize=11)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', frameon=True, framealpha=0.8, fontsize=9)

        if title_prefix is None:
            title_prefix = 'Rollout Unique-Segment Metrics Analysis'
        if mode_name is not None:
            main_title = f"{title_prefix} - Mode: {mode_name}"
        else:
            main_title = title_prefix

        subtitle = "(root from Round 1; branches from branch_at_turn+; unique segments only)"
        full_title = f"{main_title}\n{subtitle}"

        plt.suptitle(full_title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = os.path.join(fig_dir, 'unique_metrics_dashboard.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_timing_success_fail(self, timing_stats, fig_dir):
        """
        分支时机分析：左图成功率 + 总/成功分支数；右图失败率 + 总/失败分支数。
        """
        if not timing_stats:
            return
        os.makedirs(fig_dir, exist_ok=True)

        rounds = sorted(timing_stats.keys())
        if not rounds:
            return

        total_counts = [timing_stats[r]['total'] for r in rounds]
        success_counts = [timing_stats[r]['success'] for r in rounds]
        fail_counts = [timing_stats[r]['fail'] for r in rounds]

        success_rates = [s / t if t > 0 else 0
                         for s, t in zip(success_counts, total_counts)]
        fail_rates = [f / t if t > 0 else 0
                      for f, t in zip(fail_counts, total_counts)]

        fig, (ax_s, ax_f) = plt.subplots(1, 2, figsize=(18, 7))
        plt.subplots_adjust(wspace=0.3)

        ax_s_count = ax_s.twinx()
        ax_s_count.bar(rounds, total_counts, color='lightgray',
                       alpha=0.4, label='Total Branches')
        ax_s_count.bar(rounds, success_counts, color='#2ecc71',
                       alpha=0.6, label='Success Branches')
        ax_s.plot(rounds, success_rates, color='#27ae60', marker='o',
                  linewidth=3, label='Success Rate')

        ax_s.set_ylabel('Success Rate', color='#27ae60', fontsize=12)
        ax_s_count.set_ylabel('Counts', color='gray', fontsize=12)
        ax_s.set_title('When Does Intervention Work? (Success Timing)',
                       fontsize=14, fontweight='bold')
        ax_s.set_xlabel('Round (Intervention Point)', fontsize=12)
        ax_s.set_ylim(0, 1.1)
        ax_s.grid(True, alpha=0.3)
        lines_s, labels_s = ax_s.get_legend_handles_labels()
        lines_sc, labels_sc = ax_s_count.get_legend_handles_labels()
        ax_s.legend(lines_s + lines_sc, labels_s + labels_sc,
                    loc='upper left', frameon=True)

        ax_f_count = ax_f.twinx()
        ax_f_count.bar(rounds, total_counts, color='lightgray',
                       alpha=0.4, label='Total Branches')
        ax_f_count.bar(rounds, fail_counts, color='#e74c3c',
                       alpha=0.6, label='Fail Branches')
        ax_f.plot(rounds, fail_rates, color='#c0392b', marker='x',
                  linewidth=3, label='Fail Rate')

        ax_f.set_ylabel('Fail Rate', color='#c0392b', fontsize=12)
        ax_f_count.set_ylabel('Counts', color='gray', fontsize=12)
        ax_f.set_title('When Does Intervention Fail? (Failure Timing)',
                       fontsize=14, fontweight='bold')
        ax_f.set_xlabel('Round (Intervention Point)', fontsize=12)
        ax_f.set_ylim(0, 1.1)
        ax_f.grid(True, alpha=0.3)
        lines_f, labels_f = ax_f.get_legend_handles_labels()
        lines_fc, labels_fc = ax_f_count.get_legend_handles_labels()
        ax_f.legend(lines_f + lines_fc, labels_f + labels_fc,
                    loc='upper left', frameon=True)

        out_path = os.path.join(fig_dir, 'timing_success_fail_analysis.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_mode_comprehensive_performance(self, mode_user_stats,
                                            mode_traj_stats, fig_dir):
        """
        多 mode 综合表现看板：
        子图1：各 mode 的 Root / Oracle / Rescue rate；
        子图2：Root vs Branch 轨迹成功率；
        子图3：按用户的分支成功率在不同 mode 下的对比。
        """
        if not mode_traj_stats:
            return
        os.makedirs(fig_dir, exist_ok=True)

        modes = list(mode_traj_stats.keys())
        if not modes:
            return

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))
        plt.subplots_adjust(wspace=0.25)

        x = np.arange(len(modes))
        width = 0.25

        oracle_srs, rescue_rates, root_srs = [], [], []
        for m in modes:
            stats_u = mode_user_stats.get(m, {})
            total_users = len(stats_u)
            if total_users == 0:
                root_srs.append(0.0)
                oracle_srs.append(0.0)
                rescue_rates.append(0.0)
                continue
            succ_users = sum(1 for u in stats_u.values()
                             if u['root_succ'] or u['branch_succ_count'] > 0)
            root_succ_users = sum(1 for u in stats_u.values()
                                  if u['root_succ'])
            root_fail_users = total_users - root_succ_users
            rescue_users = sum(1 for u in stats_u.values()
                               if (not u['root_succ']) and
                               u['branch_succ_count'] > 0)

            root_srs.append(root_succ_users / total_users)
            oracle_srs.append(succ_users / total_users)
            rescue_rates.append(rescue_users / (root_fail_users + 1e-6))

        ax1.bar(x - width, root_srs, width, label='Root SR',
                color='#bdc3c7')
        ax1.bar(x, oracle_srs, width, label='Oracle SR',
                color='#2ecc71')
        ax1.bar(x + width, rescue_rates, width, label='Rescue Rate',
                color='#3498db')

        for i, m in enumerate(modes):
            stats_u = mode_user_stats.get(m, {})
            total = len(stats_u)
            if total == 0:
                continue
            o_succ = sum(1 for u in stats_u.values()
                         if u['root_succ'] or u['branch_succ_count'] > 0)
            ax1.text(i, oracle_srs[i] + 0.02,
                     f"{oracle_srs[i]:.1%} ({o_succ}/{total})",
                     ha='center', va='bottom', fontsize=9,
                     fontweight='bold')

        ax1.set_title('Global Performance: How Much Did We Improve?',
                      fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(modes)
        ax1.set_ylabel('Rate')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3, linestyle='--')

        branch_srs = []
        for m in modes:
            total = mode_traj_stats[m]['branch_total']
            succ = mode_traj_stats[m]['branch_succ']
            branch_srs.append(succ / (total + 1e-6))

        ax2.bar(x - 0.2, root_srs, 0.4, label='Root Trajectory SR',
                color='#95a5a6', alpha=0.7)
        ax2.bar(x + 0.2, branch_srs, 0.4, label='Branch Trajectory SR',
                color='#e67e22', alpha=0.8)

        for i, m in enumerate(modes):
            r_succ = mode_traj_stats[m]['root_succ']
            r_total = mode_traj_stats[m]['root_total']
            b_succ = mode_traj_stats[m]['branch_succ']
            b_total = mode_traj_stats[m]['branch_total']
            if r_total > 0:
                ax2.text(i - 0.2, root_srs[i] + 0.02,
                         f"{r_succ}/{r_total}", ha='center', fontsize=8)
            if b_total > 0:
                ax2.text(i + 0.2, branch_srs[i] + 0.02,
                         f"{b_succ}/{b_total}", ha='center', fontsize=8)

        ax2.set_title('Trajectory-level Win Rate: Root vs Branch',
                      fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(modes)
        ax2.set_ylabel('Success Rate')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3, linestyle='--')

        all_users = sorted({u for m in modes
                            for u in mode_user_stats.get(m, {}).keys()})
        n_users = len(all_users)
        if n_users > 0:
            user_width = min(0.2, 0.8 / n_users)
        else:
            user_width = 0.15

        for i, user_id in enumerate(all_users):
            offset = (i - (n_users - 1) / 2) * user_width
            user_offsets = x + offset
            u_rates, u_texts = [], []
            for m in modes:
                u_stat = mode_user_stats.get(m, {}).get(
                    user_id, {'branch_succ_count': 0, 'num_branches': 0})
                succ = u_stat['branch_succ_count']
                total = u_stat['num_branches']
                rate = succ / total if total > 0 else 0.0
                u_rates.append(rate)
                u_texts.append(f"{succ}/{total}")
            label_name = f"User_{user_id[-3:]}" if len(str(user_id)) > 3 \
                else f"User_{user_id}"
            bars = ax3.bar(user_offsets, u_rates, user_width,
                           label=label_name)
            for j, bar in enumerate(bars):
                if mode_user_stats.get(modes[j], {}).get(
                        user_id, {}).get('num_branches', 0) > 0:
                    ax3.text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.01, u_texts[j],
                             ha='center', va='bottom', fontsize=6,
                             rotation=45)

        ax3.set_title('Per-User Branch Success Ratio Across Modes',
                      fontsize=14, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(modes)
        ax3.set_ylabel('Success Ratio (Branches)')
        ax3.set_ylim(0, 1.05)
        ax3.grid(axis='y', linestyle='--', alpha=0.5)
        ax3.legend(title='Users', fontsize=8, loc='upper left',
                   bbox_to_anchor=(1, 1))

        out_path = os.path.join(fig_dir, 'mode_comprehensive_performance.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

    # =========================================================
    # Branch audit visualizations
    # =========================================================

    def plot_branch_audit_length_distribution(self, df_details):
        """分支长度分布：成功 vs 失败。"""
        if df_details is None or df_details.empty:
            return
        df = df_details[df_details['Type'] == 'Branch'].copy()
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        for success, color, label in [
            (True, '#27ae60', 'Success branches'),
            (False, '#e74c3c', 'Fail branches'),
        ]:
            sub = df[df['Success'] == success]
            if sub.empty:
                continue
            sns.histplot(sub['Turns'], bins=10, kde=True, stat='count', ax=ax,
                         color=color, alpha=0.25,
                         label=f"{label} (n={len(sub)})")
        ax.set_xlabel('Branch trajectory length (Turns; last round index)',
                      fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Branch Trajectory Length Distribution: Success vs Failure',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['branch_audit'],
                                 'branch_length_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_branch_audit_length_by_turn(self, df_details):
        """分支轮次 vs 成功分支长度 + 成功率。"""
        if df_details is None or df_details.empty:
            return
        df = df_details[df_details['Type'] == 'Branch'].copy()
        if df.empty:
            return
        grouped = df.groupby('Branch Round')
        stats = grouped['Success'].agg(['sum', 'count']).reset_index()
        stats.rename(columns={'sum': 'SuccessCount',
                              'count': 'TotalCount'}, inplace=True)
        stats['SuccessRate'] = stats['SuccessCount'] / stats['TotalCount']

        df_success = df[df['Success'] == True].copy()
        if df_success.empty:
            return
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(x='Branch Round', y='Turns', data=df_success,
                    ax=ax, color='#95a5a6', fliersize=3)
        ax.set_xlabel('Branch Round (Intervention point)', fontsize=12)
        ax.set_ylabel('Length of successful branch (Turns)', fontsize=12)
        ax.set_title('Successful Branch Length by Branch Round',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        ax2 = ax.twinx()
        ax2.plot(stats['Branch Round'], stats['SuccessRate'],
                 color='#2980b9', marker='o', linewidth=2,
                 label='Success rate (all branches)')
        ax2.set_ylabel('Success rate', color='#2980b9', fontsize=12)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis='y', labelcolor='#2980b9')

        for _, row in stats.iterrows():
            ax2.text(row['Branch Round'], row['SuccessRate'] + 0.03,
                     f"n={int(row['TotalCount'])}", ha='center',
                     va='bottom', fontsize=8, color='#2980b9')
        lines2, labels2 = ax2.get_legend_handles_labels()
        if lines2:
            ax2.legend(lines2, labels2, loc='upper right', frameon=True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['branch_audit'],
                                 'branch_length_by_turn.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def plot_branch_audit_strategy_change_success(self, df_details):
        """策略是否改变 vs 分支成功率。"""
        if df_details is None or df_details.empty:
            return
        df = df_details[df_details['Type'] == 'Branch'].copy()
        if df.empty:
            return

        def _changed(row):
            old_s = str(row['Old Strategy'])
            new_s = str(row['New Strategy'])
            if old_s in ['-', 'N/A'] or new_s in ['-', 'N/A']:
                return 'Unknown'
            return 'Changed' if old_s != new_s else 'Same'

        df['ChangeType'] = df.apply(_changed, axis=1)
        df = df[df['ChangeType'] != 'Unknown']
        if df.empty:
            return
        grp = df.groupby('ChangeType')['Success'].agg(['sum', 'count']).reset_index()
        grp['SuccessRate'] = grp['sum'] / grp['count']

        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(grp))
        bars = ax.bar(x, grp['SuccessRate'],
                      color=['#27ae60' if ct == 'Changed' else '#95a5a6'
                             for ct in grp['ChangeType']],
                      alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(grp['ChangeType'])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Success rate of branches', fontsize=12)
        ax.set_title('Does Changing Strategy at Branch Help?',
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        for i, bar in enumerate(bars):
            rate = grp.loc[i, 'SuccessRate']
            cnt = grp.loc[i, 'count']
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02,
                    f"{rate:.1%}\n(n={int(cnt)})",
                    ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['branch_audit'],
                                 'branch_strategy_change_success.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()



    # =========================================================
    # 2. 动态 & 3. 策略 (保持原有的优秀实现，略微调整风格)
    # =========================================================

    # def plot_4_metric_trend(self, df_turns):
    #     """2.1 四维指标时序图"""
    #     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    #     metrics = [
    #         ('Hs', 'State Entropy ($H_s$)', '#3498db'),
    #         ('Ha', 'Action Entropy ($H_a$)', '#e67e22'),
    #         ('Dissonance', 'Cognitive Dissonance ($D$)', '#9b59b6'),
    #         ('Trend', 'Trend ($S_{Trend}$)', '#2ecc71')
    #     ]
        
    #     for i, (col, title, color) in enumerate(metrics):
    #         ax = axes[i//2, i%2]
    #         sns.lineplot(data=df_turns, x='Turn', y=col, hue='Outcome', 
    #                      palette={'Success': '#27ae60', 'Fail': '#c0392b'}, 
    #                      ax=ax, linewidth=2.5, errorbar='sd', marker='o') 
    #         ax.set_title(title, fontweight='bold', fontsize=14)
    #         ax.grid(True, linestyle='--', alpha=0.4)
    #         if col == 'Trend': ax.axhline(0, color='gray', ls='--', alpha=0.5)

    #     plt.suptitle("Dynamic Temporal Evolution (Success vs Fail)", y=0.98, fontsize=18, fontweight='bold')
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(self.fig_dirs['dynamics'], '4_metric_trends.png'), dpi=300)
    #     plt.close()

    def plot_critical_turns(self, df_summary, df_critical=None):
        """
        Critical rescue turns: per-round total branches vs success branches, and success ratio.
        - x-axis: Round 1..10 only.
        - Bars: total (gray), success (green); twin axis: success rate line with markers.
        - Annotations: N= total and percentage per round (style ref: analysis_token).
        df_summary: DataFrame with Round, TotalBranches, SuccessBranches, SuccessRatio (from causal_analyzer.get_critical_turns_summary).
        """
        if df_summary is None or df_summary.empty:
            return
        df = df_summary.copy()
        # Restrict to rounds 1..10
        df = df[(df['Round'] >= 1) & (df['Round'] <= 10)].sort_values('Round').reset_index(drop=True)
        if df.empty:
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        x = df['Round'].astype(int)
        width = 0.36
        off = width / 2
        # Total branches (background)
        bars_total = ax.bar(x - off, df['TotalBranches'], width=width, color='#bdc3c7', label='Total Branches', alpha=0.9)
        # Success branches (foreground)
        bars_success = ax.bar(x + off, df['SuccessBranches'], width=width, color='#2ecc71', label='Success Branches', alpha=0.9)

        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Branch Count', fontsize=12, fontweight='bold')
        ax.set_title('Critical Rescue Turns: Success Branches vs Total Branches by Round', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(range(1, 11))
        ax.set_xticklabels(range(1, 11))
        ax.set_xlim(0.5, 10.5)
        ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        ax.legend(loc='upper right', frameon=True)

        # Twin axis: success rate line (ref: analysis_token plot_success_rate_dashboard)
        ax_twin = ax.twinx()
        ax_twin.plot(x, df['SuccessRatio'], color='#e74c3c', marker='D', linewidth=2.5, markersize=7, label='Success Rate')
        ax_twin.set_ylim(0, 1.1)
        ax_twin.set_ylabel('Success Rate', fontsize=12, fontweight='bold', color='#c0392b')
        ax_twin.tick_params(axis='y', labelcolor='#c0392b')
        ax_twin.grid(False)

        # Annotations: N= on total bar, percentage above success-rate line (ref: analysis_token)
        for _, row in df.iterrows():
            r = int(row['Round'])
            if row['TotalBranches'] > 0:
                ax.text(r - off, row['TotalBranches'] + 0.15, f"N={int(row['TotalBranches'])}",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax_twin.text(r, row['SuccessRatio'] + 0.04, f"{row['SuccessRatio']:.0%}",
                        ha='center', va='bottom', color='#c0392b', fontsize=10, fontweight='bold')

        # Merge legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['dynamics'], 'critical_turn_hist.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_entropy_relief(self, df_relief):
        if df_relief.empty: return
        plt.figure(figsize=(10, 7))
        sns.boxplot(x='BranchOutcome', y='EntropyRelief', hue='BranchOutcome', 
                    data=df_relief, palette={'Success': '#2ecc71', 'Fail': '#e74c3c'}, legend=False)
        plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
        plt.title('Entropy Relief Effect ($\Delta H_s$: Parent - Branch)', fontsize=16, fontweight='bold')
        plt.ylabel('Entropy Relief (Positive = Improved Certainty)', fontsize=12)
        plt.savefig(os.path.join(self.fig_dirs['dynamics'], 'entropy_relief_box.png'), dpi=300)
        plt.close()

    def plot_strategy_radar(self, df_fingerprint):
        if df_fingerprint.empty: return
        
        pivot = df_fingerprint.pivot(index='BigFive', columns='Strategy', values='Frequency').fillna(0)
        strategies = list(pivot.columns)
        num_vars = len(strategies)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        palette = sns.color_palette("bright", len(pivot))
        
        for i, (idx, row) in enumerate(pivot.iterrows()):
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=idx, color=palette[i])
            ax.fill(angles, values, color=palette[i], alpha=0.1)
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(strategies, fontsize=10, fontweight='bold')
        # 优化图例位置
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Personality", title_fontsize='12')
        plt.title("Strategy Fingerprint by Personality (Success Cases)", y=1.08, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['strategy_radar'], 'strategy_radar_fingerprint.png'), dpi=300)
        plt.close()

    def plot_strategy_radar_success_vs_fail(self, radar_data):
        """Radar chart: compare strategy set (proportions) between Success and Failure trajectories."""
        if not radar_data or not radar_data.get('strategies'):
            return
        strategies = radar_data['strategies']
        success_vals = radar_data.get('Success', [])
        fail_vals = radar_data.get('Fail', [])
        if len(success_vals) != len(strategies) or len(fail_vals) != len(strategies):
            return
        num_vars = len(strategies)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]
        success_vals = success_vals + success_vals[:1]
        fail_vals = fail_vals + fail_vals[:1]
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
        ax.plot(angles, success_vals, linewidth=2.5, label='Success Trajectories', color='#27ae60')
        ax.fill(angles, success_vals, color='#27ae60', alpha=0.2)
        ax.plot(angles, fail_vals, linewidth=2.5, label='Failure Trajectories', color='#e74c3c')
        ax.fill(angles, fail_vals, color='#e74c3c', alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(strategies, fontsize=9, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), frameon=True)
        ax.set_title('Strategy Set: Success vs Failure Trajectories', y=1.08, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['strategy_radar'], 'strategy_radar_success_vs_fail.png'), dpi=300)
        plt.close()

    def plot_diversity(self, df_div):
        plt.figure(figsize=(12, 7))
        sns.violinplot(x='BigFive', y='DiversityScore', hue='Outcome', 
                       data=df_div, split=True, palette={'Success': '#2ecc71', 'Fail': '#e74c3c'}, inner='quart')
        plt.title('Strategy Diversity Score (Consistency vs Diversity)', fontsize=16, fontweight='bold')
        plt.ylabel('Diversity Ratio (Unique/Total)', fontsize=12)
        plt.savefig(os.path.join(self.fig_dirs['strategy_overall'], 'strategy_diversity.png'), dpi=300)
        plt.close()

    # =========================================================
    # Strategy analysis: preference, per-turn, branch re-selection, overall
    # =========================================================

    def plot_strategy_preference_bigfive(self, df_pref):
        """Heatmap: Big Five x Strategy (proportion); tendency strength (dominance) as secondary bar."""
        if df_pref is None or df_pref.empty:
            return
        pivot = df_pref.pivot(index='BigFive', columns='Strategy', values='Proportion').fillna(0)
        # Tendency per BigFive (take first row per BigFive for Dominance)
        tendency = df_pref.groupby('BigFive').agg({'Dominance': 'first', 'Concentration': 'first'}).reindex(pivot.index)
        fig, axes = plt.subplots(2, 1, figsize=(max(14, pivot.shape[1] * 0.8), 10), gridspec_kw={'height_ratios': [2, 0.6]})
        ax1, ax2 = axes
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax1, cbar_kws={'label': 'Proportion'}, linewidths=0.3)
        ax1.set_title('Strategy Preference by Big Five Personality (Success Turns)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Big Five')
        # Tendency bar (grouped)
        x = np.arange(len(tendency))
        w = 0.35
        ax2.bar(x - w/2, tendency['Dominance'], width=w, color='#3498db', alpha=0.8, label='Dominance (top strategy share)')
        ax2.bar(x + w/2, tendency['Concentration'], width=w, color='#9b59b6', alpha=0.8, label='Concentration (1 - norm entropy)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(tendency.index, rotation=45, ha='right')
        ax2.set_ylabel('Tendency strength')
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['strategy_preference'], 'strategy_preference_bigfive.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_strategy_preference_style(self, df_pref):
        """Heatmap: Decision Style x Strategy (proportion); tendency bar below."""
        if df_pref is None or df_pref.empty:
            return
        pivot = df_pref.pivot(index='Style', columns='Strategy', values='Proportion').fillna(0)
        tendency = df_pref.groupby('Style').agg({'Dominance': 'first', 'Concentration': 'first'}).reindex(pivot.index)

        # 根据 style 数量动态增高，避免纵轴标签重叠
        fig_h = max(8, 0.9 * pivot.shape[0] + 4)
        fig, axes = plt.subplots(
            2, 1,
            figsize=(max(14, pivot.shape[1] * 0.8), fig_h),
            gridspec_kw={'height_ratios': [2.2, 0.8]}
        )
        ax1, ax2 = axes
        sns.heatmap(
            pivot, annot=True, fmt='.2f', cmap='Oranges', ax=ax1,
            cbar_kws={'label': 'Proportion'}, linewidths=0.3
        )
        ax1.set_title('Strategy Preference by Decision-Making Style (Success Turns)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Style')
        # 关键：强制 y 轴标签水平显示并留出左侧空间
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, va='center')
        ax1.tick_params(axis='y', labelsize=10)

        x = np.arange(len(tendency))
        w = 0.35
        ax2.bar(x - w/2, tendency['Dominance'], width=w, color='#e67e22', alpha=0.8, label='Dominance')
        ax2.bar(x + w/2, tendency['Concentration'], width=w, color='#9b59b6', alpha=0.8, label='Concentration')
        ax2.set_xticks(x)
        ax2.set_xticklabels(tendency.index, rotation=20, ha='right')
        ax2.set_ylabel('Tendency strength')
        ax2.legend(fontsize=8)
        ax2.set_ylim(0, 1.05)

        # 增加左边距，防止长 style 名被截断或重叠
        plt.subplots_adjust(left=0.26, hspace=0.35)
        plt.savefig(os.path.join(self.fig_dirs['strategy_preference'], 'strategy_preference_style.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_strategy_by_turn(self, turn_data):
        """Heatmap: Turn x Strategy (proportion); shows per-turn tendency."""
        prop = turn_data.get('proportion_matrix')
        if prop is None or prop.empty:
            return
        fig, ax = plt.subplots(figsize=(max(12, prop.shape[1] * 0.6), max(6, prop.shape[0] * 0.35)))
        sns.heatmap(prop, annot=True, fmt='.2f', cmap='Blues', ax=ax, cbar_kws={'label': 'Proportion'}, linewidths=0.3)
        ax.set_title('Strategy Distribution by Turn (All Trajectories)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Turn')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['strategy_by_turn'], 'strategy_by_turn_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_strategy_by_turn_dominant(self, turn_data):
        """Bar: per turn, dominant strategy proportion (tendency strength per turn)."""
        stats = turn_data.get('turn_stats', [])
        if not stats:
            return
        df = pd.DataFrame(stats)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(df['Turn'], df['dominant_proportion'], color='#3498db', alpha=0.8, edgecolor='#2c3e50')
        ax.set_xlabel('Turn', fontsize=12)
        ax.set_ylabel('Dominant strategy proportion', fontsize=12)
        ax.set_title('Per-Turn Strategy Tendency (Dominant Strategy Share)', fontsize=14, fontweight='bold')
        ax.set_xticks(df['Turn'])
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['strategy_by_turn'], 'strategy_by_turn_dominant.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_branch_reselection(self, branch_data):
        """Same vs different strategy at branch; re-selection ratio; parent x branch crosstab heatmap."""
        if not branch_data or branch_data.get('total_branches', 0) == 0:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # Left: same vs different counts + ratio
        ax1 = axes[0]
        same = branch_data['same_count']
        diff = branch_data['different_count']
        total = branch_data['total_branches']
        ratio = branch_data['re_selection_sufficient_ratio']
        ax1.bar([0], [same], width=0.5, color='#95a5a6', label=f'Same as parent (n={same})')
        ax1.bar([1], [diff], width=0.5, color='#2ecc71', label=f'Re-selected / different (n={diff})')
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(['Same', 'Re-selected'])
        ax1.set_ylabel('Branch count')
        ax1.set_title('Same-Turn Branch: Strategy Same vs Re-Selected', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.text(0.5, 0.95, f'Re-selection ratio: {ratio:.1%}', transform=ax1.transAxes, ha='center', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.4)
        # Right: crosstab heatmap (parent x branch)
        ax2 = axes[1]
        ctab = branch_data.get('parent_branch_crosstab')
        if ctab is not None and not ctab.empty:
            sns.heatmap(ctab, annot=True, fmt='d', cmap='YlOrRd', ax=ax2, linewidths=0.3)
            ax2.set_title('Parent Strategy x Branch Strategy (Counts)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Branch strategy')
            ax2.set_ylabel('Parent strategy')
        else:
            ax2.text(0.5, 0.5, 'No crosstab data', ha='center', va='center', transform=ax2.transAxes)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['strategy_branch'], 'strategy_branch_reselection.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # Optional: re-selection ratio by turn
        by_turn = branch_data.get('by_turn', [])
        if by_turn:
            bt = pd.DataFrame(by_turn)
            fig2, ax = plt.subplots(figsize=(8, 4))
            ax.bar(bt['Turn'], bt['re_selection_ratio'], color='#2ecc71', alpha=0.8)
            ax.set_xlabel('Turn')
            ax.set_ylabel('Re-selection ratio')
            ax.set_title('Re-Selection Ratio by Turn (at Branch)', fontsize=12, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(self.fig_dirs['strategy_branch'], 'strategy_branch_reselection_by_turn.png'), dpi=300, bbox_inches='tight')
            plt.close()

    def plot_strategy_overall(self, summary):
        """Overall strategy counts bar chart and summary text."""
        if not summary or not summary.get('strategy_counts'):
            return
        counts = summary['strategy_counts']
        strat = list(counts.keys())
        cnt = list(counts.values())
        fig, ax = plt.subplots(figsize=(max(10, len(strat) * 0.5), 6))
        bars = ax.bar(range(len(strat)), cnt, color='#3498db', alpha=0.8, edgecolor='#2c3e50')
        ax.set_xticks(range(len(strat)))
        ax.set_xticklabels(strat, rotation=45, ha='right')
        ax.set_ylabel('Turn count')
        ax.set_xlabel('Strategy')
        ax.set_title('Overall Strategy Usage (All Turns)', fontsize=14, fontweight='bold')
        for b, c in zip(bars, cnt):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, str(int(c)), ha='center', va='bottom', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['strategy_overall'], 'strategy_overall_counts.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_strategy_outcome_success_rate(self, df_sr):
        """Bar: success rate per strategy (turns in success trajectories)."""
        if df_sr is None or df_sr.empty:
            return
        fig, ax = plt.subplots(figsize=(max(10, len(df_sr) * 0.5), 6))
        x = range(len(df_sr))
        ax.bar(x, df_sr['success_rate'], color='#2ecc71', alpha=0.8, edgecolor='#27ae60')
        ax.set_xticks(x)
        ax.set_xticklabels(df_sr['Strategy'], rotation=45, ha='right')
        ax.set_ylabel('Success rate (turn-level)')
        ax.set_xlabel('Strategy')
        ax.set_title('Strategy vs Outcome: Success Rate per Strategy', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['strategy_outcome'], 'strategy_success_rate.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_strategy_transition_matrix(self, trans_matrix):
        """Heatmap: strategy transition (from -> to) within trajectories."""
        if trans_matrix is None or trans_matrix.empty:
            return
        fig, ax = plt.subplots(figsize=(max(10, trans_matrix.shape[1] * 0.6), max(8, trans_matrix.shape[0] * 0.4)))
        sns.heatmap(trans_matrix, annot=True, fmt='d', cmap='Purples', ax=ax, linewidths=0.3)
        ax.set_title('Strategy Transition Matrix (Turn t -> t+1 within Trajectory)', fontsize=14, fontweight='bold')
        ax.set_xlabel('To strategy')
        ax.set_ylabel('From strategy')
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_dirs['strategy_outcome'], 'strategy_transition_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # =========================================================
    # 动态时序与状态演变 · 三层分析（成功轨迹群 vs 失败轨迹群）
    # =========================================================

    def _draw_four_metric_panel(self, axes, data_by_outcome, title_prefix=''):
        """
        在 2x2 axes 上绘制四指标时序图，每子图含成功/失败两条线及标准差阴影。
        data_by_outcome: dict {'Success': df, 'Fail': df}，df 含 Turn, Hs_mean, Hs_std, ...
        """
        metrics = ['Hs', 'Ha', 'Dissonance', 'Trend']
        for i, col in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            for outcome in ['Success', 'Fail']:
                df = data_by_outcome.get(outcome)
                if df is None or df.empty:
                    continue
                t = df['Turn']
                mu = df[f'{col}_mean']
                std = df[f'{col}_std']
                std = std.fillna(0)
                color = OUTCOME_COLORS[outcome]
                label = OUTCOME_LABELS[outcome]
                ax.plot(t, mu, color=color, linewidth=2.5, marker='o', markersize=6, label=label)
                if std.any():
                    ax.fill_between(t, mu - std, mu + std, color=color, alpha=0.2)
            ax.set_title(METRIC_TITLES.get(col, col), fontsize=13, fontweight='bold', color='#2c3e50', pad=10)
            ax.set_xlabel('Turn', fontsize=11)
            ax.set_ylabel('Metric Value', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='best', fontsize='x-small', frameon=True, framealpha=0.7)
            if col == 'Trend':
                ax.axhline(0, color='gray', ls='--', alpha=0.5)

    def plot_temporal_overall_level(self, data_by_outcome, save_pdf=True):
        """
        (1) Overall level: four metrics Success vs Failure over turns.
        Output: PNG + one-page PDF (English only to avoid encoding issues).
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        self._draw_four_metric_panel(axes, data_by_outcome)
        plt.suptitle('Dynamic Temporal & State Evolution - Overall Level\nSuccess vs Failure Trajectories', fontsize=18, fontweight='bold', color='#1a252f', y=0.98)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        png_path = os.path.join(self.fig_dirs['dynamics'], 'temporal_overall_success_vs_fail.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        if save_pdf:
            pdf_path = os.path.join(self.fig_dirs['dynamics_reports'], 'temporal_overall_level.pdf')
            with PdfPages(pdf_path) as pdf:
                fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
                self._draw_four_metric_panel(axes2, data_by_outcome)
                plt.suptitle('Dynamic Temporal & State Evolution - Overall Level\nSuccess vs Failure Trajectories', fontsize=18, fontweight='bold', color='#1a252f', y=0.98)
                plt.tight_layout(rect=[0, 0.02, 1, 0.96])
                pdf.savefig(fig2, bbox_inches='tight')
                plt.close()

    def plot_temporal_persona_level(self, analyzer, save_pdf=True):
        """
        (2) Persona level: two separate PDFs (flip pages to browse).
        - temporal_persona_BigFive.pdf: one page per Big Five personality.
        - temporal_persona_Style.pdf: one page per decision-making style.
        """
        # PDF 1: Big Five — one page per personality
        bigfive_list = analyzer.get_bigfive_categories()
        if bigfive_list:
            pdf_bf_path = os.path.join(self.fig_dirs['persona_pdfs'], 'temporal_persona_BigFive.pdf')
            with PdfPages(pdf_bf_path) as pdf:
                for big_five in bigfive_list:
                    data_by_outcome = analyzer.get_trend_by_outcome_for_bigfive(big_five)
                    if data_by_outcome.get('Success', pd.DataFrame()).empty and data_by_outcome.get('Fail', pd.DataFrame()).empty:
                        continue
                    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                    self._draw_four_metric_panel(axes, data_by_outcome)
                    plt.suptitle(f'Big Five Personality: {big_five}\nSuccess vs Failure Trajectories', fontsize=18, fontweight='bold', color='#1a252f', y=0.98)
                    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
        # PDF 2: Decision-making style — one page per style
        style_list = analyzer.get_style_categories()
        if style_list:
            pdf_style_path = os.path.join(self.fig_dirs['persona_pdfs'], 'temporal_persona_Style.pdf')
            with PdfPages(pdf_style_path) as pdf:
                for style in style_list:
                    data_by_outcome = analyzer.get_trend_by_outcome_for_style(style)
                    if data_by_outcome.get('Success', pd.DataFrame()).empty and data_by_outcome.get('Fail', pd.DataFrame()).empty:
                        continue
                    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                    self._draw_four_metric_panel(axes, data_by_outcome)
                    plt.suptitle(f'Decision-Making Style: {style}\nSuccess vs Failure Trajectories', fontsize=18, fontweight='bold', color='#1a252f', y=0.98)
                    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

    def plot_temporal_individual_level(self, analyzer, df_user_stats, save_pdf=True, users_per_page=2):
        """
        (3) Individual level: one PDF, each page shows user summary (initial success, rescue, trajectory counts/rate) and four-metric trends (English only).
        """
        user_summaries = analyzer.get_user_summary_for_display(df_user_stats)
        if not user_summaries:
            return
        pdf_path = os.path.join(self.fig_dirs['individual_pdf'], 'temporal_individual_level.pdf')
        with PdfPages(pdf_path) as pdf:
            for i in range(0, len(user_summaries), users_per_page):
                chunk = user_summaries[i : i + users_per_page]
                n = len(chunk)
                fig, axes = plt.subplots(n, 5, figsize=(18, 4 * n), squeeze=False)
                for u_idx, uinfo in enumerate(chunk):
                    ax0 = axes[u_idx, 0]
                    ax0.set_axis_off()
                    card = (
                        f"User: {uinfo['User']}\n"
                        f"Personality: {uinfo['BigFive']}\n"
                        f"Decision Style: {uinfo['Style']}\n"
                        f"Total Trajectories: {uinfo['Total_Trajs']}\n"
                        f"Success Trajectories: {uinfo['Success_Trajs']}\n"
                        f"Trajectory Success Rate: {uinfo['Trajectory_Success_Rate']:.1%}\n"
                        f"Initial Success: {uinfo['Initial_Success']}\n"
                        f"Needs Rescue: {uinfo['Needs_Rescue']}\n"
                        f"Rescued: {uinfo['Rescued']}"
                    )
                    ax0.text(0.05, 0.95, card, transform=ax0.transAxes, fontsize=10,
                             verticalalignment='top', fontfamily='sans-serif',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.35))

                    data_by_outcome = analyzer.get_user_trend_by_outcome(uinfo['User'])
                    for m_idx, col in enumerate(['Hs', 'Ha', 'Dissonance', 'Trend']):
                        ax = axes[u_idx, m_idx + 1]
                        has_any = False
                        for outcome in ['Success', 'Fail']:
                            df = data_by_outcome.get(outcome)
                            if df is None or df.empty:
                                continue
                            has_any = True
                            ax.plot(df['Turn'], df[f'{col}_mean'], color=OUTCOME_COLORS[outcome],
                                    marker='o', markersize=4, linewidth=1.5, label=OUTCOME_LABELS[outcome])
                        if not has_any:
                            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=9)
                        ax.set_title(METRIC_TITLES.get(col, col), fontsize=10, fontweight='bold')
                        ax.set_xlabel('Turn')
                        ax.legend(fontsize=7)
                        ax.grid(True, linestyle='--', alpha=0.4)
                        if col == 'Trend':
                            ax.axhline(0, color='gray', ls='--', alpha=0.5)
                plt.suptitle(f'Individual Level - User Details (Page {i//users_per_page + 1})', fontsize=16, fontweight='bold', y=0.98)
                plt.tight_layout(rect=[0, 0.02, 1, 0.96])
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

    def _normalize_series(self, s):
        s = s.astype(float)
        if s.isna().all():
            return pd.Series(np.zeros(len(s)), index=s.index)
        mn, mx = np.nanmin(s.values), np.nanmax(s.values)
        if np.isclose(mx, mn):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mn) / (mx - mn)

    def _assign_quadrant(self, row, hs_med, ha_med):
        hs_high = row['Hs_mean'] >= hs_med
        ha_high = row['Ha_mean'] >= ha_med
        if hs_high and ha_high:
            return 'Q1_HighHs_HighHa'
        if (not hs_high) and ha_high:
            return 'Q2_LowHs_HighHa'
        if (not hs_high) and (not ha_high):
            return 'Q3_LowHs_LowHa'
        return 'Q4_HighHs_LowHa'

    def select_representative_trajectories(
        self,
        df_traj_features,
        max_total=24,
        max_per_user=2,
        max_per_persona=3,
        topk_per_bucket=3,
        min_target=16,
        max_extreme=8,
    ):
        """
        四象限平衡筛选（仅用于 trajectory-level PDF 展示，不影响全量统计输出）。
        Step1: quadrant x outcome 分桶选 topk；
        Step2: 极值补充；
        Step3: 多样性约束 + 总量约束。
        """
        if df_traj_features is None or df_traj_features.empty:
            return pd.DataFrame()

        df = df_traj_features.copy()
        required = [
            'TrajID', 'User', 'BigFive', 'Style', 'Outcome', 'Length',
            'Root_Success', 'Intervened_Success',
            'Hs_mean', 'Ha_mean', 'Delta_Hs_abs_mean', 'Delta_Ha_abs_mean', 'Dissonance_mean'
        ]
        for c in required:
            if c not in df.columns:
                df[c] = np.nan

        # Clean outcome label to Success/Fail
        fallback_outcome = np.where(df['Intervened_Success'].astype(bool), 'Success', 'Fail')
        df['Outcome'] = df['Outcome'].where(df['Outcome'].notna(), pd.Series(fallback_outcome, index=df.index))
        df['Outcome'] = df['Outcome'].replace({'Failure': 'Fail'})

        # Quadrant by global medians
        hs_med = df['Hs_mean'].median(skipna=True)
        ha_med = df['Ha_mean'].median(skipna=True)
        df['Quadrant'] = df.apply(lambda r: self._assign_quadrant(r, hs_med, ha_med), axis=1)

        # Scoring for analytical value
        df['norm_abs_dissonance'] = self._normalize_series(df['Dissonance_mean'].abs())
        df['norm_delta_hs_vol'] = self._normalize_series(df['Delta_Hs_abs_mean'])
        df['norm_delta_ha_vol'] = self._normalize_series(df['Delta_Ha_abs_mean'])
        df['norm_length'] = self._normalize_series(df['Length'].fillna(0))
        df['selection_score'] = (
            0.35 * df['norm_abs_dissonance']
            + 0.25 * df['norm_delta_hs_vol']
            + 0.25 * df['norm_delta_ha_vol']
            + 0.15 * df['norm_length']
        )

        # Representative tags
        df['Representative_Tag'] = ''
        rescue_mask = (df['Root_Success'].astype(bool) == False) & (df['Intervened_Success'].astype(bool) == True)
        fragile_mask = (df['Root_Success'].astype(bool) == True) & (df['Intervened_Success'].astype(bool) == False)
        df.loc[rescue_mask, 'Representative_Tag'] = 'Rescue'
        df.loc[fragile_mask, 'Representative_Tag'] = df.loc[fragile_mask, 'Representative_Tag'].apply(
            lambda x: 'Fragile' if x == '' else f"{x}|Fragile"
        )

        candidate_ids = []

        def _append_with_tag(traj_id, tag):
            if traj_id not in candidate_ids:
                candidate_ids.append(traj_id)
                idx = df.index[df['TrajID'] == traj_id]
                if len(idx) > 0:
                    cur = str(df.loc[idx[0], 'Representative_Tag'])
                    if cur and cur != '':
                        if tag not in cur.split('|'):
                            df.loc[idx[0], 'Representative_Tag'] = f"{cur}|{tag}"
                    else:
                        df.loc[idx[0], 'Representative_Tag'] = tag

        # Step 1: bucket selection (quadrant x outcome)
        quadrants = [
            'Q1_HighHs_HighHa',
            'Q2_LowHs_HighHa',
            'Q3_LowHs_LowHa',
            'Q4_HighHs_LowHa',
        ]
        outcomes = ['Success', 'Fail']
        for q in quadrants:
            for o in outcomes:
                sub = df[(df['Quadrant'] == q) & (df['Outcome'] == o)]
                if sub.empty:
                    continue
                sub = sub.sort_values('selection_score', ascending=False)
                for traj_id in sub['TrajID'].head(topk_per_bucket).tolist():
                    _append_with_tag(traj_id, 'BucketSelected')

        # Step 2: extreme supplements
        extremes = []
        if not df['Hs_mean'].dropna().empty:
            extremes += [df.loc[df['Hs_mean'].idxmax(), 'TrajID'], df.loc[df['Hs_mean'].idxmin(), 'TrajID']]
        if not df['Ha_mean'].dropna().empty:
            extremes += [df.loc[df['Ha_mean'].idxmax(), 'TrajID'], df.loc[df['Ha_mean'].idxmin(), 'TrajID']]
        if not df['Delta_Hs_abs_mean'].dropna().empty:
            extremes += [df.loc[df['Delta_Hs_abs_mean'].idxmax(), 'TrajID']]
        if not df['Delta_Ha_abs_mean'].dropna().empty:
            extremes += [df.loc[df['Delta_Ha_abs_mean'].idxmax(), 'TrajID']]
        if not df['Dissonance_mean'].dropna().empty:
            extremes += [df.loc[df['Dissonance_mean'].idxmax(), 'TrajID']]

        extreme_added = 0
        for tid in extremes:
            if extreme_added >= max_extreme:
                break
            before = len(candidate_ids)
            _append_with_tag(tid, 'ExtremeCase')
            if len(candidate_ids) > before:
                extreme_added += 1

        # Fallback pool by score (for reaching min_target under constraints)
        fallback_ranked = df.sort_values('selection_score', ascending=False)['TrajID'].tolist()
        for tid in fallback_ranked:
            if len(candidate_ids) >= max(min_target, len(candidate_ids)):
                break
            _append_with_tag(tid, 'BucketSelected')
            if len(candidate_ids) >= min_target:
                break

        # Step 3: diversity constraints + final cap
        selected_rows = []
        user_count = {}
        persona_count = {}

        candidate_df = df.set_index('TrajID').loc[[tid for tid in candidate_ids if tid in set(df['TrajID'])]].reset_index()
        candidate_df = candidate_df.sort_values('selection_score', ascending=False)

        for _, r in candidate_df.iterrows():
            if len(selected_rows) >= max_total:
                break
            user = r['User']
            persona = f"{r['BigFive']}|{r['Style']}"
            if user_count.get(user, 0) >= max_per_user:
                continue
            if persona_count.get(persona, 0) >= max_per_persona:
                continue
            selected_rows.append(r)
            user_count[user] = user_count.get(user, 0) + 1
            persona_count[persona] = persona_count.get(persona, 0) + 1

        # If too strict and still below min_target, relax by filling from all trajectories under remaining constraints
        if len(selected_rows) < min_target:
            rest_df = df.sort_values('selection_score', ascending=False)
            selected_ids = {r['TrajID'] for r in selected_rows}
            for _, r in rest_df.iterrows():
                if len(selected_rows) >= min(max_total, min_target):
                    break
                if r['TrajID'] in selected_ids:
                    continue
                user = r['User']
                persona = f"{r['BigFive']}|{r['Style']}"
                if user_count.get(user, 0) >= max_per_user:
                    continue
                if persona_count.get(persona, 0) >= max_per_persona:
                    continue
                tag = str(r['Representative_Tag'])
                if tag == '' or pd.isna(tag):
                    r['Representative_Tag'] = 'BucketSelected'
                selected_rows.append(r)
                selected_ids.add(r['TrajID'])
                user_count[user] = user_count.get(user, 0) + 1
                persona_count[persona] = persona_count.get(persona, 0) + 1

        if not selected_rows:
            return pd.DataFrame()

        selected = pd.DataFrame(selected_rows)
        return selected.sort_values(['Quadrant', 'Outcome', 'selection_score'], ascending=[True, True, False]).reset_index(drop=True)

    def _build_trajectory_heatmap_html(self, payload, html_path):
        """生成纯静态可打开的交互式 trajectory heatmap HTML（内嵌数据，无后端依赖）。"""
        data_json = json.dumps(payload, ensure_ascii=False)
        html_template = """
<!DOCTYPE html>
<html lang=\"en\"><head><meta charset=\"UTF-8\"/><meta name=\"viewport\" content=\"width=device-width,initial-scale=1.0\"/>
<title>Trajectory Representative Heatmap Report</title>
<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
<style>
body{margin:0;font-family:Arial,sans-serif;background:#f4f6f8} .wrap{display:grid;grid-template-columns:340px 1fr;min-height:100vh}
.sidebar{background:#111827;color:#e5e7eb;padding:12px;overflow:auto}.main{padding:12px;overflow:auto}
.traj-item{border:1px solid #374151;border-radius:8px;padding:8px;margin-bottom:8px;cursor:pointer;background:#1f2937}
.traj-item.active{border-color:#60a5fa;background:#1e3a8a}.card{background:#fff;border:1px solid #d1d5db;border-radius:10px;padding:10px;margin-bottom:10px}
.summary{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.kv{background:#fafafa;border:1px solid #e5e7eb;border-radius:8px;padding:6px}
.turn{background:#fff;border:1px solid #d1d5db;border-radius:10px;margin-bottom:10px}.th{padding:10px;background:#f9fafb;cursor:pointer;display:flex;justify-content:space-between}
.tb{display:none;padding:10px;border-top:1px solid #e5e7eb}.tb.show{display:block}.token{display:inline-block;white-space:pre-wrap;padding:1px 0;border-radius:2px;cursor:pointer}
#tip{position:fixed;display:none;background:#111827;color:#fff;border-radius:6px;padding:8px;font-size:12px;z-index:9999;max-width:340px}
</style></head><body><div id=\"tip\"></div><div class=\"wrap\"><aside class=\"sidebar\">
<div style=\"font-weight:700;font-size:18px;margin-bottom:8px\">Representative Trajectories</div>
<select id=\"fq\" style=\"width:100%;margin-bottom:6px\"><option value=\"ALL\">All Quadrants</option></select>
<select id=\"fo\" style=\"width:100%;margin-bottom:6px\"><option value=\"ALL\">All Outcomes</option><option value=\"Success\">Success</option><option value=\"Fail\">Fail</option></select>
<select id=\"ft\" style=\"width:100%;margin-bottom:10px\"><option value=\"ALL\">All Tags</option></select>
<div id=\"list\"></div></aside><main class=\"main\"><div id=\"summary\" class=\"card\"></div>
<div class=\"card\"><div id=\"c1\" style=\"height:300px\"></div><div id=\"c2\" style=\"height:300px\"></div></div><div id=\"turns\"></div></main></div>
<script>
const DATA=__DATA_JSON_PLACEHOLDER__ ;
const uniq=a=>[...new Set(a.filter(x=>x!==null&&x!==undefined&&x!==''))];
const fmt=(x,n=4)=>x===null||x===undefined||Number.isNaN(x)?'N/A':Number(x).toFixed(n);
function col(v,min,max){if(v===null||v===undefined||Number.isNaN(v))return '#f3f4f6';const t=(max-min)<1e-9?0:(v-min)/(max-min);if(t<0.5){const k=t*2;return `rgb(${255-Math.round(180*k)},${255-Math.round(120*k)},255)`;}const k=(t-0.5)*2;return `rgb(${75+Math.round(180*k)},${135-Math.round(95*k)},${255-Math.round(210*k)})`;}
function tcol(v,min,max){if(v===null||v===undefined||Number.isNaN(v))return '#111';const t=(max-min)<1e-9?0:(v-min)/(max-min);return t>0.55?'#fff':'#111';}
function tokenSpan(tok,min,max){const s=(tok.token||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');if((tok.token||'').includes('\n')&&!(tok.token||'').trim())return '<span style="display:block;height:6px"></span>';const info=encodeURIComponent(JSON.stringify(tok));return `<span class='token' style='background:${col(tok.entropy,min,max)};color:${tcol(tok.entropy,min,max)}' data-i='${info}'>${s}</span>`;}
function qline(q){if(!q)return 'quality: N/A';return `avg=${fmt(q.avg_entropy,3)} | max=${fmt(q.max_entropy,3)} | high-ratio=${fmt(q.high_entropy_ratio,3)} | tag=${q.quality_tag||'unknown'}`;}
function fdata(){const q=document.getElementById('fq').value,o=document.getElementById('fo').value,t=document.getElementById('ft').value;return (DATA.trajectories||[]).filter(x=>{if(q!=='ALL'&&x.Quadrant!==q)return false;if(o!=='ALL'&&x.Outcome!==o)return false;if(t!=='ALL'&&!String(x.Representative_Tag||'').split('|').includes(t))return false;return true;});}
function renderList(){const arr=fdata(),box=document.getElementById('list');box.innerHTML='';if(!arr.length){box.innerHTML='No trajectory';return;}arr.forEach(x=>{const d=document.createElement('div');d.className='traj-item'+(DATA.__id===x.TrajID?' active':'');d.innerHTML=`<div style='display:flex;justify-content:space-between;font-size:12px'><span>${x.Quadrant||'N/A'}</span><span>${x.Outcome||'N/A'}</span></div><div style='font-weight:700;font-size:13px'>${x.TrajID}</div><div style='font-size:12px'>User: ${x.User} | Len: ${x.Length}</div><div style='font-size:11px;color:#a7f3d0'>${x.Representative_Tag||''}</div>`;d.onclick=()=>{DATA.__id=x.TrajID;renderList();renderMain(x)};box.appendChild(d);});const cur=arr.find(x=>x.TrajID===DATA.__id)||arr[0];DATA.__id=cur.TrajID;renderMain(cur);}
function renderMain(t){document.getElementById('summary').innerHTML=`<div style='font-size:18px;font-weight:700;margin-bottom:8px'>Trajectory Summary</div><div class='summary'>${[['User',t.User],['TrajID',t.TrajID],['BigFive',t.BigFive||'Unknown'],['Style',t.Style||'Unknown'],['Quadrant',t.Quadrant||'N/A'],['Root_Success',t.Root_Success?'Yes':'No'],['Current_Success',t.Current_Success?'Yes':'No'],['Representative_Tag',t.Representative_Tag||''],['Hs_mean',fmt(t.Hs_mean)],['Ha_mean',fmt(t.Ha_mean)],['Dissonance_mean',fmt(t.Dissonance_mean)],['Length',t.Length]].map(kv=>`<div class='kv'><div style='font-size:11px;color:#6b7280'>${kv[0]}</div><div style='font-size:13px;font-weight:700'>${kv[1]}</div></div>`).join('')}</div>`;
const turns=(t.Turns||[]).map(x=>x.Turn),hs=(t.Turns||[]).map(x=>x.Hs),ha=(t.Turns||[]).map(x=>x.Ha),dhs=(t.Turns||[]).map(x=>x.Delta_Hs),dha=(t.Turns||[]).map(x=>x.Delta_Ha);
Plotly.newPlot('c1',[{x:turns,y:hs,mode:'lines+markers',name:'Hs'},{x:turns,y:ha,mode:'lines+markers',name:'Ha'}],{title:'Hs / Ha by Turn',xaxis:{title:'Turn'},yaxis:{title:'Value'},margin:{t:40,l:45,r:20,b:35}},{responsive:true});
Plotly.newPlot('c2',[{x:turns,y:dhs,mode:'lines+markers',name:'ΔHs'},{x:turns,y:dha,mode:'lines+markers',name:'ΔHa'}],{title:'ΔHs / ΔHa by Turn',xaxis:{title:'Turn'},yaxis:{title:'Delta'},margin:{t:40,l:45,r:20,b:35}},{responsive:true});
const go=ev=>{if(!ev.points||!ev.points.length)return;const trn=ev.points[0].x;const el=document.getElementById('turn-'+trn);if(el){el.scrollIntoView({behavior:'smooth',block:'start'});const b=el.querySelector('.tb');if(b)b.classList.add('show');}};document.getElementById('c1').on('plotly_click',go);document.getElementById('c2').on('plotly_click',go);
const all=[];(t.Turns||[]).forEach(tr=>['state_analysis','strategy','response','all'].forEach(k=>(tr.Segments&&tr.Segments[k]||[]).forEach(z=>{if(z.entropy!==null&&z.entropy!==undefined)all.push(Number(z.entropy));})));const mn=all.length?Math.min(...all):0,mx=all.length?Math.max(...all):1;
const tbox=document.getElementById('turns');tbox.innerHTML='';(t.Turns||[]).forEach((tr,i)=>{const risk=tr.TurnRiskTag||'unknown';const seg=(name,label)=>{const toks=(tr.Segments&&tr.Segments[name])||[];return `<div class='card' style='margin:6px 0'><div style='font-weight:700;font-size:12px'>${label}</div><div style='font-size:11px;color:#374151'>${qline((tr.SegmentQuality||{})[name])}</div><div>${toks.length?toks.map(z=>tokenSpan(z,mn,mx)).join(''):'<span style="color:#6b7280">No token-level segment data.</span>'}</div></div>`;};
const raw=(tr.RawText||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const d=document.createElement('div');d.className='turn';d.id='turn-'+tr.Turn;d.innerHTML=`<div class='th'><div><b>Turn ${tr.Turn}</b> <span style='font-size:11px;color:#6b7280'>(${risk})</span></div><div>click to fold</div></div><div class='tb ${i<2?'show':''}'><div style='display:grid;grid-template-columns:repeat(9,minmax(0,1fr));gap:6px'>${[['Hs',tr.Hs],['Ha',tr.Ha],['Delta_Hs',tr.Delta_Hs],['Delta_Ha',tr.Delta_Ha],['Dissonance',tr.Dissonance],['TurnRiskTag',risk],['SelectedStrategy',tr.SelectedStrategy||'Unknown'],['StrategySource',tr.StrategySource||'missing'],['Ha_Source',tr.Ha_Source||'strategy']].map(kv=>`<div class='kv'><div style='font-size:10px;color:#6b7280'>${kv[0]}</div><div style='font-size:12px;font-weight:700'>${typeof kv[1]==='number'?fmt(kv[1]):kv[1]}</div></div>`).join('')}</div>${seg('state_analysis','State Analysis Token Heatmap')}${seg('strategy','Strategy Token Heatmap (Ha Source)')}${seg('response','Response Token Heatmap')}<div class='card' style='margin:6px 0'><div style='font-weight:700;font-size:12px'>Fallback / Full Text Token Heatmap</div><div style='font-size:11px;color:#374151'>${qline((tr.SegmentQuality||{}).all)}</div><div>${((tr.Segments&&tr.Segments.all)||[]).length?(tr.Segments.all||[]).map(z=>tokenSpan(z,mn,mx)).join(''):'<span style="color:#6b7280">No token-level entropy data; showing raw text only.</span>'}</div><div style='margin-top:6px;font-size:12px;color:#374151;white-space:pre-wrap'><b>Raw Text:</b> ${raw}</div></div></div>`;tbox.appendChild(d);});

document.querySelectorAll('.th').forEach(h=>h.onclick=()=>h.nextElementSibling.classList.toggle('show'));
const tip=document.getElementById('tip');document.querySelectorAll('.token').forEach(el=>{el.onmousemove=e=>{const d=JSON.parse(decodeURIComponent(el.getAttribute('data-i')));const top=(d.top_5||[]).map((x,i)=>`${i+1}. ${(x.token||'').replace(/</g,'&lt;')}: ${Number(x.prob||0).toFixed(4)}`).join('<br>');tip.innerHTML=`<b>${(d.token||'').replace(/</g,'&lt;')}</b><br>entropy=${fmt(d.entropy,4)}${top?'<hr style=\"margin:4px 0\">'+top:''}`;tip.style.display='block';tip.style.left=(e.pageX+12)+'px';tip.style.top=(e.pageY+12)+'px';};el.onmouseleave=()=>tip.style.display='none';});
}
(function boot(){const ts=DATA.trajectories||[];uniq(ts.map(x=>x.Quadrant)).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;document.getElementById('fq').appendChild(o);});uniq(ts.flatMap(x=>String(x.Representative_Tag||'').split('|'))).forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;document.getElementById('ft').appendChild(o);});['fq','fo','ft'].forEach(id=>document.getElementById(id).onchange=()=>{DATA.__id=null;renderList();});renderList();})();
</script></body></html>
"""
        html = html_template.replace('__DATA_JSON_PLACEHOLDER__', data_json)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)

    def plot_temporal_trajectory_level(
        self,
        analyzer,
        df_traj_stats,
        report_path=None,
        save_pdf=True,
        trajs_per_page=4,
        max_trajectories_for_pdf=24,
        selection_strategy='quadrant_balanced',
        max_per_user=2,
        max_per_persona=3,
        topk_per_bucket=3,
        generate_html_report=True,
    ):
        """
        (4) Trajectory level: representative trajectory PDF + interactive HTML report.
        - Full CSV remains unchanged; selection only affects representative outputs.
        """
        df_traj_features = analyzer.get_trajectory_feature_table(df_traj_stats)
        if df_traj_features is None or df_traj_features.empty:
            return

        if selection_strategy == 'quadrant_balanced':
            selected_df = self.select_representative_trajectories(
                df_traj_features,
                max_total=min(max_trajectories_for_pdf, 32),
                max_per_user=max_per_user,
                max_per_persona=max_per_persona,
                topk_per_bucket=topk_per_bucket,
                min_target=16,
                max_extreme=8,
            )
        else:
            selected_df = df_traj_features.head(min(max_trajectories_for_pdf, 32)).copy()

        if selected_df is None or selected_df.empty:
            return

        if report_path is not None:
            os.makedirs(report_path, exist_ok=True)
            selected_df.to_csv(os.path.join(report_path, '0_trajectory_representative_selection.csv'), index=False)

        traj_summaries = analyzer.get_trajectory_summary_for_display(selected_df)
        if not traj_summaries:
            return

        extra_map = selected_df.set_index('TrajID')[[
            'Quadrant', 'Hs_mean', 'Ha_mean', 'Dissonance_mean', 'Representative_Tag', 'Intervened_Success', 'Root_Success'
        ]].to_dict('index')

        if save_pdf:
            pdf_path = os.path.join(self.fig_dirs['trajectory_pdf'], 'temporal_trajectory_level.pdf')
            with PdfPages(pdf_path) as pdf:
                for i in range(0, len(traj_summaries), trajs_per_page):
                    chunk = traj_summaries[i : i + trajs_per_page]
                    n = len(chunk)
                    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n), squeeze=False)

                    for t_idx, tinfo in enumerate(chunk):
                        ax0 = axes[t_idx, 0]
                        ax0.set_axis_off()
                        branch_round = tinfo['BranchTurn']
                        branch_round_text = 'N/A' if pd.isna(branch_round) or int(branch_round) < 0 else str(int(branch_round) + 1)
                        ext = extra_map.get(tinfo['TrajID'], {})
                        card = (
                            f"Trajectory: {tinfo['TrajID']}\n"
                            f"User: {tinfo['User']}\n"
                            f"Personality: {tinfo['BigFive']}\n"
                            f"Decision Style: {tinfo['Style']}\n"
                            f"Is Root: {'Yes' if tinfo['IsRoot'] else 'No'}\n"
                            f"Parent ID: {tinfo['ParentID']}\n"
                            f"Branch Round: {branch_round_text}\n"
                            f"Turns: {tinfo['Length']}\n"
                            f"Quadrant: {ext.get('Quadrant', 'N/A')}\n"
                            f"Hs_mean: {ext.get('Hs_mean', np.nan):.4f}\n"
                            f"Ha_mean: {ext.get('Ha_mean', np.nan):.4f}\n"
                            f"Dissonance_mean: {ext.get('Dissonance_mean', np.nan):.4f}\n"
                            f"Root_Success: {'Yes' if bool(ext.get('Root_Success', False)) else 'No'}\n"
                            f"Current_Success: {'Yes' if bool(ext.get('Intervened_Success', False)) else 'No'}\n"
                            f"Representative_Tag: {ext.get('Representative_Tag', 'BucketSelected')}"
                        )
                        ax0.text(0.03, 0.97, card, transform=ax0.transAxes, fontsize=9.1,
                                 verticalalignment='top', fontfamily='sans-serif',
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.35))

                        traj_df = analyzer.get_trajectory_trend(tinfo['TrajID'])

                        ax1 = axes[t_idx, 1]
                        if traj_df.empty:
                            ax1.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax1.transAxes, fontsize=9)
                        else:
                            ax1.plot(traj_df['Turn'], traj_df['Hs'], color='#2980b9', marker='o', linewidth=1.8, label='$H_s$')
                            ax1.plot(traj_df['Turn'], traj_df['Ha'], color='#e67e22', marker='s', linewidth=1.8, label='$H_a$')
                        ax1.set_title('Trajectory Metrics: $H_s$ / $H_a$', fontsize=10, fontweight='bold')
                        ax1.set_xlabel('Turn')
                        ax1.set_ylabel('Value')
                        ax1.grid(True, linestyle='--', alpha=0.4)
                        ax1.legend(fontsize=8, loc='best')

                        ax2 = axes[t_idx, 2]
                        if traj_df.empty:
                            ax2.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax2.transAxes, fontsize=9)
                        else:
                            ax2.plot(traj_df['Turn'], traj_df['Delta_Hs'], color='#16a085', marker='o', linewidth=1.8, label='$\Delta H_s$')
                            ax2.plot(traj_df['Turn'], traj_df['Delta_Ha'], color='#c0392b', marker='s', linewidth=1.8, label='$\Delta H_a$')
                        ax2.axhline(0, color='gray', ls='--', alpha=0.5)
                        ax2.set_title('Trajectory Delta Metrics: $\Delta H_s$ / $\Delta H_a$', fontsize=10, fontweight='bold')
                        ax2.set_xlabel('Turn')
                        ax2.set_ylabel('Delta Value')
                        ax2.grid(True, linestyle='--', alpha=0.4)
                        ax2.legend(fontsize=8, loc='best')

                    plt.suptitle(
                        f'Trajectory Level - Representative Cases ({len(traj_summaries)} total) (Page {i//trajs_per_page + 1})',
                        fontsize=16,
                        fontweight='bold',
                        y=0.98,
                    )
                    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()

        if generate_html_report:
            payload, debug_payload = analyzer.build_representative_trajectory_heatmap_payload(selected_df)
            with open(os.path.join(self.fig_dirs['trajectory_pdf'], 'trajectory_heatmap_data.json'), 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            if report_path is not None:
                with open(os.path.join(report_path, '0_trajectory_heatmap_debug.json'), 'w', encoding='utf-8') as f:
                    json.dump(debug_payload, f, ensure_ascii=False, indent=2)
                with open(os.path.join(report_path, '0_trajectory_reconstruction_debug.json'), 'w', encoding='utf-8') as f:
                    json.dump({'reconstruction': debug_payload.get('reconstruction', [])}, f, ensure_ascii=False, indent=2)
                s = debug_payload.get('summary', {})
                print("    [Heatmap Debug] total_turns=", s.get('total_turns', 0),
                      "matched_turns=", s.get('matched_turns', 0),
                      "with_tokens=", s.get('turns_with_tokens', 0),
                      "with_entropy=", s.get('turns_with_entropy', 0),
                      "state_seg=", s.get('turns_with_state_segment', 0),
                      "strategy_seg=", s.get('turns_with_strategy_segment', 0),
                      "response_seg=", s.get('turns_with_response_segment', 0),
                      "all_only=", s.get('turns_all_only', 0))
                print("    [Heatmap Debug] strategy_present=", s.get('turns_with_strategy', 0),
                      "strategy_missing=", s.get('turns_missing_strategy', 0),
                      "inherited_metrics=", s.get('turns_with_inherited_metrics', 0),
                      "local_metrics=", s.get('turns_with_local_metrics', 0))
                print("    [Heatmap Debug] source_reasons=", s.get('reasons', {}))
            self._build_trajectory_heatmap_html(
                payload,
                os.path.join(self.fig_dirs['trajectory_pdf'], 'trajectory_heatmap_report.html')
            )
                    
            plt.close()