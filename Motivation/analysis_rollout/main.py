import os
import argparse

import numpy as np
import pandas as pd
import json

from src.data_loader import DataLoader
from src.metric_calculator import MetricCalculator
from src.visualizer import Visualizer

# 导入分析器模块
from src.analyzers.global_evaluator import GlobalEvaluator
from src.analyzers.trend_analyzer import TrendAnalyzer
from src.analyzers.causal_analyzer import CausalAnalyzer
from src.analyzers.strategy_analyzer import StrategyAnalyzer
from src.analyzers.temporal_state_analyzer import TemporalStateAnalyzer

def save_json(data, path):
    """辅助函数：保存字典为JSON"""
    # 将 numpy 类型转换为 Python 原生类型
    def convert(o):
        if isinstance(o, pd.Series): return o.to_dict()
        if isinstance(o, np.int64): return int(o)
        if isinstance(o, np.float64): return float(o)
        return str(o)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, default=convert)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_file", required=True, help="Path to rollout results json")
    parser.add_argument("--persona_file", required=True, help="Path to user persona json")
    parser.add_argument("--metrics_file", default=None,
                        help="Optional explicit token-level metrics json (defaults to auto infer: rollout_file -> *_metrics.json)")
    parser.add_argument("--output_name", default="analysis_results", help="Name of the output folder")
    parser.add_argument("--max_trajectories_for_pdf", type=int, default=24,
                        help="Max representative trajectories shown in trajectory-level PDF (default: 24, hard cap 32)")
    parser.add_argument("--selection_strategy", default="quadrant_balanced",
                        help="Trajectory representative selection strategy for PDF (default: quadrant_balanced)")
    parser.add_argument("--max_per_user", type=int, default=2,
                        help="Max selected trajectories per user in trajectory PDF (default: 2)")
    parser.add_argument("--max_per_persona", type=int, default=3,
                        help="Max selected trajectories per BigFive+Style persona (default: 3)")
    parser.add_argument("--topk_per_bucket", type=int, default=3,
                        help="Top-k per (quadrant x outcome) bucket before constraints (default: 3)")
    args = parser.parse_args()

    # 路径设置
    input_abs_path = os.path.abspath(args.rollout_file)
    input_dir = os.path.dirname(input_abs_path)
    output_dir = os.path.join(input_dir, args.output_name)
    
    print(f"==================================================")
    print(f"Input:  {input_abs_path}")
    print(f"Output: {output_dir}")
    print(f"==================================================")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "reports")
    os.makedirs(report_path, exist_ok=True)

    # 1. Loading
    print(">>> 1. Loading Data...")
    loader = DataLoader(args.rollout_file, args.persona_file, metrics_file=args.metrics_file)
    df_personas = loader.load_personas()
    df_turns, df_meta, traj_lookup = loader.load_trajectories()
    print(f"    Loaded {len(df_meta)} unique trajectories.")

    # 2. Base Metrics
    print(">>> 2. Calculating Base Metrics...")
    calc = MetricCalculator(df_turns)
    df_metrics = calc.compute_basic_metrics()
    
    # Analyzers
    global_eval = GlobalEvaluator(df_meta, df_personas)
    trend_anlz = TrendAnalyzer(df_metrics)
    causal_anlz = CausalAnalyzer(df_turns, df_meta, traj_lookup)
    strat_anlz = StrategyAnalyzer(df_turns, df_personas, df_meta)
    
    viz = Visualizer(output_dir)

    # =========================================================
    # 3.1 Global Effectiveness Analysis (Detailed Refinement)
    # =========================================================
    print(">>> 3.1 Global Detailed Analysis...")
    
    # 计算详细的三层级统计数据 + trajectory 层级明细
    df_trajectory_stats = global_eval.compute_trajectory_level_stats()
    df_user_stats, df_persona_stats, global_stats = global_eval.compute_detailed_stats()
    
    # 保存数据
    if not df_trajectory_stats.empty:
        df_trajectory_stats.to_csv(os.path.join(report_path, "0_trajectory_level_stats.csv"), index=False)
    df_user_stats.to_csv(os.path.join(report_path, "1_user_level_stats.csv"), index=False)
    df_persona_stats.to_csv(os.path.join(report_path, "2_persona_level_stats.csv"), index=False)
    save_json(global_stats, os.path.join(report_path, "3_global_stats.json"))
    
    # 绘制高级图表
    viz.plot_global_summary_bar(global_stats)
    viz.plot_persona_metrics_dashboard(df_persona_stats)
    viz.plot_user_success_distribution(df_user_stats)

    # =========================================================
    # 3.2 Dynamic Temporal & State Evolution (四指标 · 成功 vs 失败 · 三层)
    # =========================================================
    print(">>> 3.2 Dynamic Temporal & State Evolution (Detailed)...")
    # viz.plot_4_metric_trend(df_metrics)
    temporal_analyzer = TemporalStateAnalyzer(df_metrics, df_meta, df_personas, traj_lookup=traj_lookup)
    # （1）总体层次
    overall_data = temporal_analyzer.get_overall_trend_by_outcome()
    viz.plot_temporal_overall_level(overall_data, save_pdf=True)
    # （2）人格决策类层次：每个类别一张 PDF
    viz.plot_temporal_persona_level(temporal_analyzer, save_pdf=True)
    # （3）个人层次：初始成功、挽救率、轨迹成功数/率 + 四指标时序，输出 PDF
    viz.plot_temporal_individual_level(temporal_analyzer, df_user_stats, save_pdf=True)
    # （4）轨迹层次：单条轨迹信息 + Hs/Ha 与 Delta_Hs/Delta_Ha 时序，输出 PDF
    viz.plot_temporal_trajectory_level(
        temporal_analyzer,
        df_trajectory_stats,
        report_path=report_path,
        save_pdf=True,
        max_trajectories_for_pdf=args.max_trajectories_for_pdf,
        selection_strategy=args.selection_strategy,
        max_per_user=args.max_per_user,
        max_per_persona=args.max_per_persona,
        topk_per_bucket=args.topk_per_bucket,
        generate_html_report=True,
    )

    # =========================================================
    # 3.3 Causal Branching Analysis
    # =========================================================
    print(">>> 3.3 Causal Branching Analysis...")
    df_critical = causal_anlz.identify_critical_turns()
    if not df_critical.empty:
        df_critical.to_csv(os.path.join(report_path, "critical_turns.csv"), index=False)
    df_critical_summary = causal_anlz.get_critical_turns_summary()
    if not df_critical_summary.empty:
        df_critical_summary.to_csv(os.path.join(report_path, "critical_turns_summary.csv"), index=False)
    viz.plot_critical_turns(df_critical_summary, df_critical)
    
    df_relief = causal_anlz.analyze_entropy_relief()
    if not df_relief.empty:
        df_relief.to_csv(os.path.join(report_path, "entropy_relief.csv"), index=False)
        viz.plot_entropy_relief(df_relief)
    else:
        print("    [Warn] No valid parent-child pairs found for entropy relief.")

    # =========================================================
    # 3.4 Strategy Analysis (comprehensive; outputs in typed subfolders)
    # =========================================================
    print(">>> 3.4 Strategy Analysis...")
    report_strategy = os.path.join(report_path, "strategy")
    r_pref = os.path.join(report_strategy, "preference")
    r_by_turn = os.path.join(report_strategy, "by_turn")
    r_branch = os.path.join(report_strategy, "branch")
    r_overall = os.path.join(report_strategy, "overall")
    r_radar = os.path.join(report_strategy, "radar")
    r_outcome = os.path.join(report_strategy, "outcome")
    for d in [report_strategy, r_pref, r_by_turn, r_branch, r_overall, r_radar, r_outcome]:
        os.makedirs(d, exist_ok=True)

    # Radar: fingerprint (by personality) + Success vs Failure comparison
    df_fingerprint = strat_anlz.analyze_fingerprint()
    if not df_fingerprint.empty:
        df_fingerprint.to_csv(os.path.join(r_radar, "strategy_fingerprint.csv"), index=False)
        viz.plot_strategy_radar(df_fingerprint)
    radar_svf = strat_anlz.strategy_proportions_by_outcome()
    if radar_svf.get('strategies'):
        save_json(radar_svf, os.path.join(r_radar, "strategy_radar_success_vs_fail.json"))
        viz.plot_strategy_radar_success_vs_fail(radar_svf)
    df_div = strat_anlz.analyze_diversity()
    if not df_div.empty:
        df_div.to_csv(os.path.join(r_overall, "strategy_diversity.csv"), index=False)
    viz.plot_diversity(df_div)

    # 1. Preference (Big Five / Style)
    prefs_bf = strat_anlz.strategy_preference_by_bigfive(use_outcome='Success')
    save_json(prefs_bf, os.path.join(r_pref, "strategy_preference_bigfive.json"))
    df_pref_bf = strat_anlz.strategy_preference_bigfive_dataframe(use_outcome='Success')
    if not df_pref_bf.empty:
        df_pref_bf.to_csv(os.path.join(r_pref, "strategy_preference_bigfive.csv"), index=False)
        viz.plot_strategy_preference_bigfive(df_pref_bf)
    prefs_style = strat_anlz.strategy_preference_by_style(use_outcome='Success')
    save_json(prefs_style, os.path.join(r_pref, "strategy_preference_style.json"))
    df_pref_style = strat_anlz.strategy_preference_style_dataframe(use_outcome='Success')
    if not df_pref_style.empty:
        df_pref_style.to_csv(os.path.join(r_pref, "strategy_preference_style.csv"), index=False)
        viz.plot_strategy_preference_style(df_pref_style)

    # 2. By turn
    turn_data = strat_anlz.strategy_by_turn()
    if turn_data.get('turn_stats'):
        save_json(turn_data['turn_stats'], os.path.join(r_by_turn, "strategy_by_turn.json"))
        turn_df = pd.DataFrame(turn_data['turn_stats'])
        if not turn_df.empty:
            expand = turn_df.drop(columns=['strategy_counts', 'strategy_proportions'], errors='ignore')
            expand.to_csv(os.path.join(r_by_turn, "strategy_by_turn_summary.csv"), index=False)
        if turn_data.get('count_matrix') is not None and not turn_data['count_matrix'].empty:
            turn_data['count_matrix'].to_csv(os.path.join(r_by_turn, "strategy_by_turn_counts.csv"))
        viz.plot_strategy_by_turn(turn_data)
        viz.plot_strategy_by_turn_dominant(turn_data)

    # 3. Branch re-selection
    branch_data = strat_anlz.same_turn_branch_strategy_analysis()
    save_json({k: v for k, v in branch_data.items() if k != 'parent_branch_crosstab' and k != 'details'}, os.path.join(r_branch, "strategy_branch_reselection.json"))
    if branch_data.get('parent_branch_crosstab') is not None and not branch_data['parent_branch_crosstab'].empty:
        branch_data['parent_branch_crosstab'].to_csv(os.path.join(r_branch, "strategy_branch_crosstab.csv"))
    if branch_data.get('details'):
        pd.DataFrame(branch_data['details']).to_csv(os.path.join(r_branch, "strategy_branch_details.csv"), index=False)
    if branch_data.get('total_branches', 0) > 0:
        viz.plot_branch_reselection(branch_data)

    # 4. Overall
    overall = strat_anlz.overall_strategy_stats()
    save_json(overall, os.path.join(r_overall, "strategy_overall.json"))
    overall_df = pd.DataFrame([{'Strategy': k, 'Count': v} for k, v in overall['strategy_counts'].items()])
    if not overall_df.empty:
        overall_df.to_csv(os.path.join(r_overall, "strategy_overall_counts.csv"), index=False)
    viz.plot_strategy_overall(overall)

    # 5. Outcome (success rate, transition matrix)
    df_sr = strat_anlz.strategy_outcome_success_rate()
    if not df_sr.empty:
        df_sr.to_csv(os.path.join(r_outcome, "strategy_success_rate.csv"), index=False)
        viz.plot_strategy_outcome_success_rate(df_sr)
    trans = strat_anlz.strategy_transition_matrix()
    if trans is not None and not trans.empty:
        trans.to_csv(os.path.join(r_outcome, "strategy_transition_matrix.csv"))
        viz.plot_strategy_transition_matrix(trans)

    print(f"\n>>> Analysis Complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()