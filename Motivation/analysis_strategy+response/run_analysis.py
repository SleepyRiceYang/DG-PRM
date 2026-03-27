import subprocess
import os
import sys

# === 统一配置区域 ===
INPUT_FILE = "results_unified_2025-12-04T16-45-29.json"  # 你的输入文件
REMARK = "12_4_v1"                                            # 你的备注（用于区分不同次运行的文件后缀）
BASE_OUTPUT_DIR = "analysis_results/result_12_4_v1"                     # 基础输出目录
# ===================

def run_script(script_name, input_file, output_dir, remark):
    print(f"\n{'='*20} Running {script_name} {'='*20}")
    
    # 确保脚本文件存在
    if not os.path.exists(script_name):
        print(f"Error: Script {script_name} not found.")
        return

    # 构建命令
    cmd = [
        sys.executable, script_name,
        "--input", input_file,
        "--output_dir", output_dir,
        "--remark", remark
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {script_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with error: {e}")

def main():
    # 确保输入文件存在
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file '{INPUT_FILE}' not found.")
        return

    # 这里的子目录你可以根据需要调整，或者都输出到同一个目录
    # 这里示例为每个分析脚本创建子文件夹
    dirs = {
        "token": os.path.join(BASE_OUTPUT_DIR, "token_analysis"),
        "sentence": os.path.join(BASE_OUTPUT_DIR, "sentence_analysis"),
        "heatmap_split": os.path.join(BASE_OUTPUT_DIR, "heatmap_split"),
        "heatmap_all": os.path.join(BASE_OUTPUT_DIR, "heatmap_all"),
    }

    # 执行四个脚本
    # 1. analysis_token.py
    run_script("analysis_token.py", INPUT_FILE, dirs["token"], REMARK)

    # 2. analysis_sentence.py
    run_script("analysis_sentence.py", INPUT_FILE, dirs["sentence"], REMARK)

    # 3. analysis_heatmap_reason+response.py
    run_script("analysis_heatmap_reason+response.py", INPUT_FILE, dirs["heatmap_split"], REMARK)

    # 4. analysis_heatmap_all.py
    run_script("analysis_heatmap_all.py", INPUT_FILE, dirs["heatmap_all"], REMARK)

    print("\n🎉 All analyses finished!")

if __name__ == "__main__":
    main()