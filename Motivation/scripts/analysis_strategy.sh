cd /root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/analysis_add_strategy_reason/

BASE_DIR="/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments"

# FILE1="${BASE_DIR}/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_False_user_gpt-4o/results_unified_2025-12-22T15-55-18_False_False_t_0.0.json"
# FILE2="${BASE_DIR}/exp_2026-01-14T14-56-23_Order_False_User_False_t_0.0_sys_qwen2.5-7B-Instruct_think_False/results_unified_2026-01-14T14-56-23_False_False_t_0.0.json"
# FILE3="${BASE_DIR}/exp_2026-01-13T11-13-31_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_True/results_unified_2026-01-13T11-13-31_False_False_t_0.0.json"
# MODEL1="Qwen3-8B"
# MODEL2="Qwen2.5-7B"
# MODEL3="Qwen3-8B-Thinking"

# FILE1="${BASE_DIR}/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_False_user_gpt-4o/rollout_v1_t_1.0_thinking_False/results_exp_t_1.0.json"
FILE1="/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_False_user_gpt-4o/rollout_v1_t_1.0_thinking_False/results_exp_t_1.0.json"
FILE2="${BASE_DIR}/exp_2026-01-13T11-13-31_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_True/rollout_v1_t_1.0_thinking_True/results_exp_t_1.0.json"

MODEL1="Rollout-Qwen3-8B"
MODEL2="Rollout-Qwen3-8B-Thinking"
OUTPUT_DIR="/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/analysis_add_strategy_reason/strategy_analysis_plots"

TEST1="${BASE_DIR}/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_False_user_gpt-4o/rollout_v1_t_1.0_thinking_False/results_exp_t_1.0_test.json"
TEST_MODEL1="Rollout-Qwen3-8B-Test"

python analysis_strategy.py \
  --files "$FILE1" "$FILE2" \
  --names "$MODEL1" "$MODEL2" \
  --output_dir "$OUTPUT_DIR" \
  --save_labeled

# python analysis_strategy.py \
#   --files "$TEST1" \
#   --names "$MODEL1" \
#   --output_dir "$OUTPUT_DIR" \
#   --save_labeled