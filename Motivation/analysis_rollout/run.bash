# Activate conda env before running: conda activate conv_ai
conda activate conv_ai

# ==========================
# Config
# ==========================
# All analysis will be unified into this single subfolder
OUT_NAME="analysis_results"

# (1) Single rollout file (for main.py and unique_main.py)
ROLLOUT_FILE="/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2026-01-13T11-13-31_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_True/rollout_v1_t_1.0_thinking_True/results_exp_t_1.0.json"
PERSONA_FILE="/root/EvolvingAgent-master/EvolvingAgentTest_wym/user_personas.json"

# (2) Multi-mode deep comparison (for deep_main.py)
BASE_DIR="/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2026-01-13T11-13-31_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_True/rollout_v1_t_1.0_thinking_True"
FILES=(
  "results_exp_t_1.0.json"
)

# Mode tag for unique_main.py
MODE_NAME="exp"
REMARK="unique_v1"

# ==========================
# 1) Full pipeline analysis (main.py)
# ==========================
python main.py \
  --rollout_file  "${ROLLOUT_FILE}" \
  --persona_file  "${PERSONA_FILE}" \
  --output_name   "${OUT_NAME}"

# ==========================
# 2) Unique segment deep analysis (unique_main.py)
# ==========================
python unique_main.py \
  --rollout_file  "${ROLLOUT_FILE}" \
  --mode          "${MODE_NAME}" \
  --remark        "${REMARK}" \
  --output_name   "${OUT_NAME}"

# ==========================
# 3) Multi-mode deep comparison (deep_main.py)
# ==========================
python deep_main.py \
  --base   "${BASE_DIR}" \
  --files  "${FILES[@]}" \
  --output "${OUT_NAME}"
