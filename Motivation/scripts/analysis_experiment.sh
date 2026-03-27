clear

cd /root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation

BASE_DIR="/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments"

# FILE1="${BASE_DIR}/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_False_user_gpt-4o/rollout_v1_t_1.0_thinking_False/results_exp_t_1.0.json"
FILE1="${BASE_DIR}/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_False_user_gpt-4o/rollout_v1_t_1.0_thinking_False/results_exp_t_1.0.json"
FILE2="${BASE_DIR}/exp_2026-01-13T11-13-31_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_True/rollout_v1_t_1.0_thinking_True/results_exp_t_1.0.json"

MODEL1="Qwen3-8B"
MODEL2="Qwen3-8B-Thinking"

# python analysis_experiment.py \
#   --files "$FILE1" "$FILE2" \
#   --names "$MODEL1" "$MODEL2" \
#   --output_dir "/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/experiment_analysis_v1"

## 特征与优势的相关性计算
# python analysis_rl_exp.py \
#   --files "$FILE1" "$FILE2" \
#   --names "$MODEL1" "$MODEL2" \
#   --output_dir "/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/rl_experiment_analysis_v2"

## 评估干预方式是否真正改变了策略轨迹
# python analysis_intervation_effect.py \
#   --files "$FILE1" "$FILE2" \
#   --names "$MODEL1" "$MODEL2" \
#   --output_dir "/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/rl_experiment_analysis_v1"

## 评估策略轨迹
# python analysis_strategy_exp_v2.py \
#   --files "$FILE1" \
#   --names "$MODEL1" \
#   --output_dir "/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/analysis_results/strategy_analysis_v2"

## 验证解耦和趋势的必要性
python validation_decoupling_trend.py \
  --files "$FILE1" \
  --names "$MODEL1" \
  --output_dir "/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/analysis_results/validation_decoupling_trend"