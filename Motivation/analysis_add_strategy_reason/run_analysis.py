import subprocess
import os
import sys
import argparse
import re

# ==============================================================================
# 1. 脚本与子目录的对应配置
# ==============================================================================
ANALYSIS_CONFIG = {
    "token": ("analysis_token.py", "analysis/entropy+varentropy"),
    "heatmap_split": ("analysis_heatmap_reason+response.py", "analysis/heatmap"),
    "heatmap_all": ("analysis_heatmap_all.py", "analysis/heatmap"),
}

# 获取当前脚本所在目录的绝对路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 默认 persona 文件就在脚本同级目录下
DEFAULT_PERSONA = os.path.join(SCRIPT_DIR, "user_personas.json")

def parse_output_path(input_path):
    """
    逻辑：解析输入文件名并生成带有完整后缀的 exp_... 文件夹
    示例输入: results_unified_2026-01-12T22-12-57_False_False_t_0.0_sys_gpt-4o-mini.json
    生成结果: exp_2026-01-12T22-12-57_Order_False_User_False_t_0.0_sys_gpt-4o-mini
    """
    base_name = os.path.basename(input_path)
    name_no_ext = os.path.splitext(base_name)[0]
    parts = name_no_ext.split('_')
    
    # 基础结构: results(0), unified(1), TIMESTAMP(2), ORDER(3), USER(4)
    if len(parts) >= 5:
        timestamp = parts[2]
        bool_order = parts[3]
        bool_user = parts[4]
        
        # 提取从第 5 位开始的所有后缀 (如 t, 0.0, sys, gpt-4o-mini)
        suffix_parts = parts[5:]
        suffix_str = "_" + "_".join(suffix_parts) if suffix_parts else ""
        
        folder_name = f"exp_{timestamp}_Order_{bool_order}_User_{bool_user}{suffix_str}"
    else:
        # 兜底逻辑：如果格式不标准，直接替换前缀
        folder_name = name_no_ext.replace("results_unified_", "exp_")
    
    # 返回与输入文件同级的路径
    return os.path.join(os.path.dirname(os.path.abspath(input_path)), folder_name)

def run_sub_script(script_name, input_file, output_dir, remark, persona_path):
    print(f"\n>>> 🚀 Executing: {script_name}")
    script_abs_path = os.path.join(SCRIPT_DIR, script_name)
    
    if not os.path.exists(script_abs_path):
        print(f"    [Skip] {script_name} not found at {script_abs_path}")
        return False

    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        sys.executable, script_abs_path,
        "--input", input_file,
        "--output_dir", output_dir,
        "--remark", remark
    ]
    
    # 为分析脚本注入 persona 参数 (如果脚本需要)
    # 根据之前的需求，token 分析脚本需要人格信息
    if "analysis_token.py" in script_name:
        if not os.path.exists(persona_path):
            print(f"    [Error] Persona file not found: {persona_path}")
            return False
        cmd.extend(["--persona", persona_path])
    
    try:
        # 运行子脚本
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"    [Error] {script_name} returned non-zero exit code.")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--persona", type=str, default=DEFAULT_PERSONA)
    parser.add_argument("--scripts", type=str, default="all")
    parser.add_argument("--remark", type=str, default="analysis")
    args = parser.parse_args()

    # 1. 动态生成输出路径 (包含模型后缀)
    base_output_path = parse_output_path(args.input)
    print(f"🎯 Input: {args.input}")
    print(f"📂 Auto Output Dir: {base_output_path}")

    # 2. 确定运行任务
    if args.scripts == "all":
        selected_tasks = ANALYSIS_CONFIG.keys()
    else:
        selected_tasks = [s.strip() for s in args.scripts.split(",")]

    # 3. 循环执行任务
    results = {}
    for task_key in selected_tasks:
        if task_key in ANALYSIS_CONFIG:
            script_file, sub_rel_dir = ANALYSIS_CONFIG[task_key]
            # 拼接最终子目录
            target_dir = os.path.join(base_output_path, sub_rel_dir)
            
            success = run_sub_script(script_file, args.input, target_dir, args.remark, args.persona)
            results[task_key] = "✅" if success else "❌"

    # 4. 打印汇总
    print("\n" + "="*40)
    print(f"Summary for: {os.path.basename(args.input)}")
    for k, v in results.items():
        print(f"  {v} {k}")
    print("="*40)

if __name__ == "__main__":
    main()