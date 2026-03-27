import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# ================= 配置区域 =================
MODES = [
    "dissonance",
    "state_entropy",
    "action_entropy",
    "random",
    "exp"
]

# 绘图布局：3行2列
PLOTS_PER_PAGE_ROWS = 3
PLOTS_PER_PAGE_COLS = 2
PLOTS_PER_PAGE = PLOTS_PER_PAGE_ROWS * PLOTS_PER_PAGE_COLS

# 颜色定义
COLORS = {
    'hs': '#1f77b4', # 蓝
    'ha': '#2ca02c', # 绿
    'hr': '#ff7f0e', # 橙
    'z_hs': '#9467bd', # 紫
    'z_ha': '#d62728'  # 红
}

# 设置绘图风格
sns.set_theme(style="whitegrid")

# ================= 核心逻辑函数 =================

def is_traj_success(trajectory):
    """判断单条轨迹是否成功"""
    if trajectory.get('success'):
        return True
    
    if not trajectory.get('turns'): 
        return False
        
    last_turn = trajectory['turns'][-1]
    return last_turn.get('reward', 0.0) >= 1.0

def calculate_running_z_score(values):
    """基于历史数据计算当前的 Z-Score"""
    z_scores = []
    history = []
    
    for v in values:
        if len(history) < 2:
            z_scores.append(0.0)
        else:
            mean = np.mean(history)
            std = np.std(history) + 1e-6
            z = (v - mean) / std
            z_scores.append(z)
        history.append(v)
    return z_scores

def extract_trajectory_data(user_id, traj_data):
    """提取单条轨迹的绘图数据"""
    rounds = []
    hs_list = []
    ha_list = []
    hr_list = [] 

    for turn in traj_data['turns']:
        if turn['role'] == 'Persuader':
            rounds.append(turn['round'])
            
            hs = turn.get('hs')
            ha = turn.get('ha')
            
            hs_list.append(hs if hs is not None else 0.0)
            ha_list.append(ha if ha is not None else 0.0)
            hr_list.append(0.0) 

    z_hs_list = calculate_running_z_score(hs_list)
    z_ha_list = calculate_running_z_score(ha_list)

    success = is_traj_success(traj_data)
    
    # [新增] 提取分叉轮次
    branch_at_turn = traj_data.get('branch_at_turn')

    return {
        "user_id": user_id,
        "traj_id": traj_data.get('id', 'unknown'),
        "is_root": traj_data.get('id') == 'root',
        "branch_at_turn": branch_at_turn, # <--- 新增字段
        "success": success,
        "rounds": rounds,
        "hs": hs_list,
        "ha": ha_list,
        "z_hs": z_hs_list,
        "z_ha": z_ha_list
    }

def get_axis_limits(all_plot_data):
    """计算全局统一的坐标轴范围"""
    max_h = 0.5 
    min_z = -1.5
    max_z = 1.5
    
    for data in all_plot_data:
        local_max_h = max(max(data['hs']), max(data['ha'])) if data['hs'] else 0
        if local_max_h > max_h: max_h = local_max_h
        
        all_z = data['z_hs'] + data['z_ha']
        if all_z:
            local_min_z = min(all_z)
            local_max_z = max(all_z)
            if local_min_z < min_z: min_z = local_min_z
            if local_max_z > max_z: max_z = local_max_z
            
    return {
        "h_ylim": (-0.05, max_h * 1.1),
        "z_ylim": (min_z * 1.1, max_z * 1.1)
    }

def draw_subplot(ax1, data, limits):
    """绘制单个子图"""
    rounds = data['rounds']
    
    # 标题设置
    status_str = "SUCCESS" if data['success'] else "FAIL"
    title_color = "green" if data['success'] else "red"
    traj_type = "ROOT" if data['is_root'] else "BRANCH"
    
    # 在标题中也显示分叉信息
    branch_info = ""
    if not data['is_root'] and data['branch_at_turn']:
        branch_info = f" | Fork @ T{data['branch_at_turn']}"

    title = f"{data['user_id']} | {traj_type}{branch_info}\n[{status_str}] Turns: {len(rounds)}"
    ax1.set_title(title, fontsize=10, color=title_color, weight='bold')
    
    # === [新增] 绘制分叉点垂直线 ===
    if not data['is_root'] and data.get('branch_at_turn'):
        b_turn = data['branch_at_turn']
        # 绘制黑色虚线
        ax1.axvline(x=b_turn, color='black', linestyle='-.', linewidth=1.5, alpha=0.6, label='Branch Point')
        # 在顶部添加文字标注 (稍微偏右一点)
        y_pos = limits['h_ylim'][1] * 0.9
        ax1.text(b_turn + 0.1, y_pos, f'Branch', 
                 color='black', fontsize=8, fontweight='bold', rotation=90, verticalalignment='top')

    # === 左轴：绝对熵 ===
    l1, = ax1.plot(rounds, data['hs'], color=COLORS['hs'], marker='.', linestyle='-', label='$H_s$ (State)')
    l2, = ax1.plot(rounds, data['ha'], color=COLORS['ha'], marker='.', linestyle='-', label='$H_a$ (Action)')
    
    ax1.set_ylabel('Entropy', fontsize=8)
    ax1.set_ylim(limits['h_ylim']) 
    ax1.tick_params(axis='both', labelsize=8)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # === 右轴：Z-Score ===
    ax2 = ax1.twinx()
    l3, = ax2.plot(rounds, data['z_hs'], color=COLORS['z_hs'], linestyle='--', linewidth=1, label='$Z(H_s)$')
    l4, = ax2.plot(rounds, data['z_ha'], color=COLORS['z_ha'], linestyle='--', linewidth=1, label='$Z(H_a)$')
    
    ax2.axhline(1.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    ax2.set_ylabel('Z-Score', fontsize=8, color='purple')
    ax2.set_ylim(limits['z_ylim']) 
    ax2.tick_params(axis='y', labelcolor='purple', labelsize=8)
    
    return [l1, l2, l3, l4]

def process_mode(mode, exp_dir):
    file_name = f"results_{mode}_t_1.0.json"
    file_path = os.path.join(exp_dir, file_name)
    
    if not os.path.exists(file_path):
        print(f"Skipping {mode}: File not found ({file_path})")
        return

    print(f"Processing {mode} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        episodes = json.load(f)

    all_plot_data = []
    
    for ep in episodes:
        user_id = ep.get('user_id', 'Unknown')
        trajectories = ep.get('trajectories', [])
        
        # 排序：Root 第一，其他按 ID
        trajectories.sort(key=lambda x: (0 if x['id']=='root' else 1, x['id']))
        
        for traj in trajectories:
            p_data = extract_trajectory_data(user_id, traj)
            if p_data['rounds']: 
                all_plot_data.append(p_data)
            
    if not all_plot_data:
        print(f"No valid data found for {mode}")
        return

    limits = get_axis_limits(all_plot_data)

    output_pdf = os.path.join(exp_dir, f"analysis_all_trajs_{mode}.pdf")
    with PdfPages(output_pdf) as pdf:
        num_plots = len(all_plot_data)
        num_pages = int(np.ceil(num_plots / PLOTS_PER_PAGE))
        
        for page in range(num_pages):
            fig, axes = plt.subplots(PLOTS_PER_PAGE_ROWS, PLOTS_PER_PAGE_COLS, figsize=(15, 12))
            axes_flat = axes.flatten()
            
            start_idx = page * PLOTS_PER_PAGE
            end_idx = min(start_idx + PLOTS_PER_PAGE, num_plots)
            
            current_batch = all_plot_data[start_idx:end_idx]
            lines = [] 
            
            for i, ax in enumerate(axes_flat):
                if i < len(current_batch):
                    lines = draw_subplot(ax, current_batch[i], limits)
                else:
                    ax.axis('off') 

            if lines:
                labels = [l.get_label() for l in lines]
                # 增加 Branch Line 的图例
                # 注意：如果这页没有画 Branch Line，这里可能不显示，但通用图例通常手动加更好
                # 这里为了简单，只显示返回的 Lines (Entropy & Z)
                fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=12)
            
            plt.suptitle(f"Mode: {mode} (All Trajectories) | Page {page+1}/{num_pages}", fontsize=14)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.08, top=0.92, hspace=0.4)
            
            pdf.savefig(fig)
            plt.close()
            
    print(f"Saved: {output_pdf}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, required=True, help="Directory containing result json files")
    args = parser.parse_args()

    for mode in MODES:
        process_mode(mode, args.exp_dir)

if __name__ == "__main__":
    main()