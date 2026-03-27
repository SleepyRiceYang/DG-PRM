#!/bin/bash

# ==============================================================================
#                                用户配置区域
# ==============================================================================

# 1. 输入目标：可以是一个具体的 .json 文件，也可以是一个包含多个 json 的文件夹
INPUT_TARGET="/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2026-01-13T11-13-31_Order_False_User_False_t_0.0_sys_Qwen3-8B_think_True/rollout_v1_t_1.0_thinking_True/results_exp_t_1.0.json"

# 2. 运行的分析脚本类型：
# 可选项: token, heatmap_split, heatmap_all
SCRIPTS_TO_RUN="token"

# 3. 实验备注名称：用于生成文件夹后缀和图表标识
REMARK="v2_10users"

# 4. 人格配置文件路径 (如果需要指定特定的 persona 文件)
PERSONA_FILE="/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/analysis_add_strategy_reason/user_personas.json"

# 5. Python 解析器路径 (默认通常是 python3)
PYTHON_BIN="python3"

# ==============================================================================
#                                执行逻辑区域 (无需修改)
# ==============================================================================
cd /root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/analysis_add_strategy_reason/
# 检查 run_analysis.py 是否在当前目录
if [ ! -f "run_analysis.py" ]; then
    echo "❌ 错误: 当前目录下未找到 run_analysis.py，请检查路径。"
    exit 1
fi

# 处理单个文件的函数
process_file() {
    local file=$1
    echo "----------------------------------------------------------------"
    echo "🚀 正在处理文件: $(basename "$file")"
    
    # 构建执行命令
    # 注意：这里会把上面配置的所有参数传递给 Python 脚本
    $PYTHON_BIN run_analysis.py \
        --input "$file" \
        --scripts "$SCRIPTS_TO_RUN" \
        --remark "$REMARK" \
        --persona "$PERSONA_FILE"
}

# --- 检查输入路径类型并开始执行 ---

if [ -f "$INPUT_TARGET" ]; then
    # 场景 A: 输入是一个单独的文件
    echo "🎯 目标是单个文件。"
    process_file "$INPUT_TARGET"

elif [ -d "$INPUT_TARGET" ]; then
    # 场景 B: 输入是一个文件夹
    echo "📂 目标是文件夹。正在搜索符合条件的文件 (results_unified_*.json)..."
    
    # 启用 nullglob 防止匹配不到文件时出现星号原样输出
    shopt -s nullglob
    FILES=("$INPUT_TARGET"/results_unified_*.json)
    
    if [ ${#FILES[@]} -eq 0 ]; then
        echo "⚠️  警告: 在目录 $INPUT_TARGET 中未找到 results_unified_*.json 文件。"
        exit 1
    fi

    # 循环处理文件夹下的每个文件
    for f in "${FILES[@]}"; do
        process_file "$f"
    done
else
    # 场景 C: 路径无效
    echo "❌ 错误: 配置的路径 '$INPUT_TARGET' 无效 (不是文件也不是目录)。"
    exit 1
fi

echo "================================================================"
echo "✨ 所有任务已执行完毕！"