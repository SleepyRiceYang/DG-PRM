cd /root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation

FILE1="/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_False_user_gpt-4o/rollout_v1_t_1.0_thinking_False/results_exp_t_1.0.json"
FILE2="/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_False_user_gpt-4o/rollout_v3_t_0.0/results_exp_t_0.0.json"
FILE3="/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2026-01-13T11-13-31_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_True/rollout_v1_t_1.0_thinking_True/results_exp_t_1.0.json"
echo "开始处理文件 $FILE1 和 $FILE2"

MODEL1="Qwen3-8B Default"
MODEL1_1="Qwen3-8B Default T0.0"
MODEL2="Qwen3-8B Thinking" 
OUTPUT_DIR="/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/analysis_rollout_branch/v2"

# python eval_offline_branch_v2.py \
#   --files "$FILE1" "$FILE2" "$FILE3" \
#   --names "$MODEL1" "$MODEL1_1" "$MODEL2" \
#   --output_dir "$OUTPUT_DIR" \
#   --budget 3

python eval_offline_branch_v2.py \
  --files "$FILE1" "$FILE3" \
  --names "$MODEL1" "$MODEL2" \
  --output_dir "$OUTPUT_DIR" \
  --budget 5