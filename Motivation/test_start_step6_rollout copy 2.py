import json
import os
import sys
import argparse
import random
import copy
import numpy as np
import pytz
import datetime

from tqdm import tqdm
import traceback

import logging

class ExperimentLogger:
    def __init__(self, log_path=None):
        self.logger = logging.getLogger("Experiment")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = [] # 清除旧的 handlers 防止重复

        # 1. Console Handler (带颜色)
        ch = logging.StreamHandler()
        ch.setFormatter(self.ColoredFormatter())
        self.logger.addHandler(ch)

        # 2. File Handler (如果指定路径)
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(fh)

    def log(self, msg):
        self.logger.info(msg)

    def log_branching(self, user_id, mode, turn_num, score, reason, old_strat, new_strats):
        """专门用于打印分叉信息的格式化日志"""
        msg = (
            f"\n{'='*20} BRANCHING TRIGGERED {'='*20}\n"
            f"User ID      : {user_id}\n"
            f"Mode         : {mode}\n"
            f"Turn         : {turn_num}\n"
            f"Trigger Score: {score:.4f}\n"
            f"Reason       : {reason}\n"
            f"----------------------------------------\n"
            f"Old Strategy : {old_strat}\n"
            f"New Strategies ({len(new_strats)}): \n" + 
            "\n".join([f"  [Branch {i+1}] {s}" for i, s in enumerate(new_strats)]) +
            f"\n{'='*60}"
        )
        self.log(msg)

    class ColoredFormatter(logging.Formatter):
        HEADER = '\033[95m'; BLUE = '\033[94m'; GREEN = '\033[92m'
        YELLOW = '\033[93m'; RED = '\033[91m'; ENDC = '\033[0m'
        
        def format(self, record):
            msg = super().format(record)
            if "BRANCHING TRIGGERED" in msg: return f"{self.RED}{msg}{self.ENDC}"
            if "Success!" in msg: return f"{self.GREEN}{msg}{self.ENDC}"
            if "Processing" in msg: return f"{self.YELLOW}{msg}{self.ENDC}"
            return msg

# ==================== 路径与导入设置 ====================
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
grandparent_dir = os.path.dirname(os.path.dirname(parent_dir))
sys.path.insert(0, grandparent_dir)

from utils.Prompt import human_persuader_strategy_instruction_map, Persuader_First_Sentence

# 复用 Step 4 的核心交互函数
from model.Motivation.test_start_step4 import (
    get_user_response, 
    get_reward_response, 
    generate_system_response,
    PrintLogger,
    calculate_entropy_and_topk
)
from model.Motivation.analysis_add_strategy_reason.analysis_token import segment_metrics_new_format

def str2bool(value):
    if isinstance(value, bool): return value
    if value.lower() in ("true", "t", "1"): return True
    elif value.lower() in ("false", "f", "0"): return False
    else: raise argparse.ArgumentTypeError("Boolean value expected.")

# ==================== 辅助计算函数 ====================
def calculate_turn_metrics(metrics):
    """计算单轮的熵值摘要"""
    if not metrics:
        return {'hs': 0.0, 'ha': 0.0, 'hr': 0.0}
    
    state_tokens, strategy_tokens, response_tokens = segment_metrics_new_format(metrics)
    
    def calc_mean(tokens):
        # assert 'entropy' in tokens[0], "Token missing 'entropy' field"
        return np.mean([t['entropy'] for t in tokens]) if tokens else 0.0
    
    return {
        'hs': calc_mean(state_tokens),
        'ha': calc_mean(strategy_tokens),
        'hr': calc_mean(response_tokens)
    }

def calculate_z_score(history_hs, current_hs):
    if len(history_hs) < 2:
        return 0.0
    mean_hist = np.mean(history_hs)
    std_hist = np.std(history_hs) + 1e-6
    return (current_hs - mean_hist) / std_hist

# ==================== 触发检测器 ====================
class TriggerDetector:
    def __init__(self, mode, global_stats):
        self.mode = mode
        self.stats = global_stats
        
        self.history_hs = []
        self.history_ha = []
        
        # 解析全局阈值
        self.TAU_HA_HIGH = global_stats.get('ha_top_20', 0.6)
        self.TAU_HS_HIGH = global_stats.get('hs_top_20', 0.6)
        # 注意: 盲目自信通常指 Ha 极低，所以用 bottom_20
        self.TAU_HA_LOW = global_stats.get('ha_bottom_20', 0.2) 

        PrintLogger.print_info(f"[Trigger Detector] Mode: {self.mode}")
        PrintLogger.print_info(f"[Trigger Detector] Thresholds:{self.TAU_HA_HIGH}, {self.TAU_HS_HIGH}, {self.TAU_HA_LOW}")
    def update_history(self, hs, ha): # <--- [修改] 接收 hs 和 ha
        self.history_hs.append(hs)
        self.history_ha.append(ha)

    def _get_z_score(self, history, current_val): # <--- [新增] 辅助函数
        if len(history) < 2:
            return 0.0
        mean_hist = np.mean(history)
        std_hist = np.std(history) + 1e-6
        return (current_val - mean_hist) / std_hist

    def check(self, turn_info, history_hs_values, history_ha_values):
        """
        判断单轮是否满足触发条件
        turn_info: {'hs': float, 'ha': float, ...}
        history_hs_values: 这一轮之前的 HS 列表
        """
        hs = turn_info['hs']
        ha = turn_info['ha']
        
        z_hs = self._get_z_score(history_hs_values, hs)
        z_ha = self._get_z_score(history_ha_values, ha)
        
        trigger = False
        reason = ""

        # 1. 随机 (Random) - 注意：随机是在运行时动态决定的，这里只是占位
        # 在实际逻辑中，random 会在遍历时独立处理
        if self.mode == "random":
            pass 

        # 2. 动作熵 (Action Entropy)
        # 高位 (Top 20%) OR 趋势剧烈 (Z > 1.5 - 这里假设 Ha 也可以算 Z，但题目只给了 HS 的 Z)
        elif self.mode == "action_entropy":
            reasons = []
            # 这里简单实现：绝对值高位
            if ha > self.TAU_HA_HIGH:
                reasons.append(f"High Ha ({ha:.2f} > {self.TAU_HA_HIGH:.2f})")
            if z_ha > 1.5: # <--- [新增] 趋势判断
                reasons.append(f"Ha Spike (Z={z_ha:.2f})")
            if reasons:
                trigger = True
                reason = " & ".join(reasons)

        # 3. 状态熵 (State Entropy)
        # 高位 (Top 20%) OR 趋势突变 (Z > 1.5)
        elif self.mode == "state_entropy":
            reasons = []
            if hs > self.TAU_HS_HIGH:
                reasons.append(f"High Hs ({hs:.2f} > {self.TAU_HS_HIGH:.2f})")
            elif z_hs > 1.5:
                reasons.append(f"Hs Spike (Z={z_hs:.2f})")
            if reasons:
                trigger = True
                reason = " & ".join(reasons)

        # 4. 认知失调 (Dissonance) - Ours
        # (Hs 高 OR Hs 突变) AND (Ha 低 OR Ha 下降)
        elif self.mode == "dissonance":
            is_state_crisis = (hs > self.TAU_HS_HIGH) or (z_hs > 1.5)
            # 简化逻辑：Ha 处于低位区域
            is_blind = (ha < self.TAU_HA_LOW) or (z_ha < -1.0) or (ha < history_ha_values[-1] if history_ha_values else False)
            
            if is_state_crisis and is_blind:
                trigger = True
                reason = f"Dissonance (Hs={hs:.2f}, Z_s={z_hs:.2f} | Ha={ha:.2f}, Z_a={z_ha:.2f})"

        elif self.mode == "exp":
            # 总是触发 (All-Trigger)，由外部 Loop 控制轮次范围
            trigger = True
            reason = "Exhaustive Probe"

        return trigger, reason

    def get_score(self, turn_info, history_hs_values, history_ha_values, turn_idx):
        """
        计算分叉优先级得分
        """
        hs = turn_info['hs']
        ha = turn_info['ha']

        if hs is None or ha is None:
            return -float('inf'), "Skip (None Entropy)"

        z_hs = self._get_z_score(history_hs_values, hs)
        
        score = -float('inf')
        reason = "Pass"

        # 1. Random: 固定轮次 2, 5, 7
        if self.mode == "random":
            if turn_idx in [2, 5, 7]: 
                score = 1.0 # 选中
                reason = f"Fixed Turn {turn_idx}"
            else:
                score = -1.0 

        # 2. Action Entropy: 熵越高越优先
        elif self.mode == "action_entropy":
            score = ha 
            reason = f"Ha={ha:.4f}"

        # 3. State Entropy: 熵越高越优先
        elif self.mode == "state_entropy":
            score = hs
            reason = f"Hs={hs:.4f}"

        # 4. Dissonance: 相对失调程度
        elif self.mode == "dissonance":
            # 基础分：Hs - Ha (状态难但动作简单)
            base_score = hs - ha
            # 加分项：状态突变 (Z > 1.5)
            spike_bonus = max(0, z_hs - 1.0) * 0.5
            
            score = base_score + spike_bonus
            reason = f"Diss={score:.4f} (Hs={hs:.2f}-Ha={ha:.2f}+Z={z_hs:.2f})"

        elif self.mode == "exp":
            if turn_idx > 1: 
                score = 100.0 # 极高分
                reason = f"Probe Turn {turn_idx + 1}"
            else:
                score = -99999 # 跳过第一轮

        return score, reason

# ==================== 轨迹管理器 ====================
class RolloutManager:
    def __init__(self, env, user_profile, sys_role, user_role, strategies_map, 
                 trigger_detector, max_n_branches, k_branches_per_node, enable_thinking=False,
                 use_local_api=True, order_change=False, branching_temperature=1.0, logger=None):
        self.env = env
        self.user_profile = user_profile
        self.sys_role = sys_role
        self.user_role = user_role
        self.strategies_map = strategies_map
        self.trigger = trigger_detector
        
        self.logger = logger

        self.MAX_N = max_n_branches # 总分叉节点预算
        self.K = k_branches_per_node # 每个节点分叉数 (含原路径)
        
        self.use_local_api = use_local_api
        self.order_change = order_change
        self.enable_thinking = enable_thinking
        self.branching_temperature = branching_temperature

        self.trajectories = [] # 存储所有轨迹对象
        self.branch_count = 0  # 已使用的分叉预算

    def process(self):
        """主处理循环：先补齐 K 值，再扫描 Root 轨迹选择 Top-N 分叉点"""
        if not self.trajectories: return []
        
        # 1. 获取 Root 轨迹
        root_traj = next((t for t in self.trajectories if t['id'] == 'root'), self.trajectories[0])
        
        # ================= [Phase 1] 检查并补齐 K 值 =================
        # 统计每个分叉点已有的分支数量及对应的 ID 索引
        # 结构: {turn_num: {'count': int, 'batch_idx': int}}
        existing_branches_info = {}
        
        for t in self.trajectories:
            if t['id'] != 'root':
                b_turn = t.get('branch_at_turn')
                t_id = t.get('id', '')
                if b_turn:
                    # 解析 ID 中的 batch index (例如 root_b0_1 -> 0)
                    try:
                        # 假设 ID 格式为 root_b{INDEX}_{SUFFIX}
                        parts = t_id.split('_b')
                        if len(parts) > 1:
                            batch_idx = int(parts[1].split('_')[0])
                            
                            if b_turn not in existing_branches_info:
                                existing_branches_info[b_turn] = {'count': 0, 'batch_idx': batch_idx}
                            
                            existing_branches_info[b_turn]['count'] += 1
                    except:
                        pass # ID 格式不匹配，跳过

        # 更新当前的 branch_count (防止 ID 冲突)
        # 找到最大的 batch_idx，新的分叉从 max + 1 开始
        max_batch_idx = -1
        for info in existing_branches_info.values():
            if info['batch_idx'] > max_batch_idx:
                max_batch_idx = info['batch_idx']
        self.branch_count = max_batch_idx + 1

        # 执行补齐
        for turn_num, info in existing_branches_info.items():
            count = info['count']
            batch_idx = info['batch_idx']
            needed = (self.K - 1) - count
            
            if needed > 0:
                if self.logger:
                    self.logger.log(f"补充 K 值: Round {turn_num} (ID: b{batch_idx}) 需要补充 {needed} 条路径")
                
                # 找到对应的 split_index
                split_index = -1
                existing_strat = ""
                for idx, turn in enumerate(root_traj['turns']):
                    if turn.get('round') == turn_num and turn['role'] == self.sys_role:
                        split_index = idx
                        existing_strat = turn.get('strategy_name')
                        break
                
                if split_index != -1:
                    # 调用分叉逻辑，传入 override_branch_idx
                    self._execute_branching(
                        root_traj, 
                        split_index, 
                        existing_strat, 
                        trigger_info={'score': 0, 'reason': f'Supplement K to {self.K}'},
                        num_to_generate=needed,
                        override_branch_idx=batch_idx # <--- 关键：沿用旧 ID
                    )

        # ================= [Phase 2] 扫描寻找新分叉点 =================
        
        # 如果已经达到最大分叉数（N），则停止
        # 注意：这里计算的是 unique turn 的数量
        current_branched_turns_count = len(existing_branches_info)
        if current_branched_turns_count >= self.MAX_N and self.trigger.mode != "exp":
            return self.trajectories

        sys_turns_indices = [i for i, t in enumerate(root_traj['turns']) if t['role'] == self.sys_role]
        candidates = [] 
        
        history_hs = []
        history_ha = []
        
        for turn_idx in sys_turns_indices: 
            turn = root_traj['turns'][turn_idx]
            current_round = turn['round']
            
            hs = turn.get('hs')
            ha = turn.get('ha')
            
            # 只有当该轮次 *没有* 被分叉过时，才计算分数
            if hs is not None and ha is not None:
                if current_round not in existing_branches_info:
                    score, reason = self.trigger.get_score(
                        {'hs': hs, 'ha': ha}, 
                        history_hs, history_ha, 
                        turn['round']
                    )
                    
                    candidates.append({
                        'score': score,
                        'turn_idx': turn_idx,
                        'round': turn['round'],
                        'reason': reason,
                        'existing_strategy': turn.get('strategy_name')
                    })
                
                history_hs.append(hs)
                history_ha.append(ha)
            
        # 排序并选择
        valid_candidates = [c for c in candidates if c['score'] > -100.0]
        valid_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 计算还能分叉几次
        remaining_budget = self.MAX_N - current_branched_turns_count
        
        if self.trigger.mode == "exp":
             selected_points = valid_candidates 
        else:
             selected_points = valid_candidates[:remaining_budget]
        
        if selected_points:
            msg = f"New Branch Points ({len(selected_points)}):\n"
            for p in selected_points:
                msg += f"  Round {p['round']}: Score={p['score']:.4f}\n"
            if self.logger: self.logger.log(msg)
            
        # 执行新分叉
        for point in selected_points:
            self._execute_branching(
                root_traj, 
                point['turn_idx'], 
                point['existing_strategy'],
                trigger_info={'score': point['score'], 'reason': point['reason']},
                num_to_generate=None, # 默认生成 K-1 个
                override_branch_idx=None # 使用自增 ID
            )
            self.branch_count += 1
            
        return self.trajectories


    def load_existing_trajectories(self, existing_trajectories, seed_detailed_history, is_success):
        """
        加载已有的轨迹数据（用于断点续传或补充 K 值）
        existing_trajectories: 从 output_file 读取的该用户的轨迹列表
        seed_detailed_history: 原始种子数据的 history
        """
        self.trajectories = existing_trajectories
        
        # 1. 确保 Root 轨迹存在且完整
        # (因为 output_file 中可能没存 metrics，我们需要把 seed data 的 metrics 补回去或者重新计算)
        root_traj = next((t for t in self.trajectories if t['id'] == 'root'), None)
        
        if not root_traj:
            # 如果没有 Root，说明是新跑，直接调用 add_seed_trajectory
            self.add_seed_trajectory(seed_detailed_history, is_success)
        else:
            # 如果有 Root，可能需要补全 metrics 信息 (如果需要的话)
            # 这里简单处理：假设已有数据的结构是完整的，或者后续 process 不强依赖 metrics
            pass

        # 2. 计算已使用的分叉点预算 (branch_count)
        # 统计有多少个不同的 branch_at_turn
        branched_turns = set()
        for t in self.trajectories:
            if t['id'] != 'root' and t.get('branch_at_turn'):
                branched_turns.add(t['branch_at_turn'])
        
        self.branch_count = len(branched_turns)
        
        if self.logger:
            self.logger.log(f"Loaded existing state: {len(self.trajectories)} trajectories, {self.branch_count} branch points.")

    def add_seed_trajectory(self, detailed_history, is_success):
        """加载初始种子轨迹 (仅用于全新开始)"""
        traj = {
            "id": "root",
            "parent_id": None,
            "turns": [],
            "history_hs": [], 
            "metrics_log": [],
            "status": "completed",
            "success": is_success
        }
        
        history_hs = []
        
        for turn in detailed_history:
            new_turn = copy.deepcopy(turn)
            if turn['role'] == self.sys_role:   
                hs = turn.get('hs')
                if hs is not None:
                    history_hs.append(new_turn['hs'])
                    new_turn['history_hs_snapshot'] = list(history_hs[:-1]) 
                else:
                    new_turn['history_hs_snapshot'] = list(history_hs)
            traj['turns'].append(new_turn)
        
        if is_success and traj['turns']:
             traj['turns'][-1]['reward'] = 1.0

        traj['history_hs'] = history_hs
        self.trajectories.append(traj)

    def _generate_distinct_step(self, context, existing_strategies,
                            strategy_map_override=None):
        """尝试生成一个策略不同的 System Response"""
        # 温度设为 1.2 以增加多样性
        target_map = strategy_map_override if strategy_map_override is not None else self.strategies_map
        PrintLogger.print_step("Existing Strategy:", existing_strategies)
        PrintLogger.print_step("Branching Strategy Map:", target_map)

        max_retries = 5
        
        for i in range(max_retries):
            # [策略升级]：前3次靠随机，后2次靠强制约束
            current_forbidden = None
            if i >= 3 and existing_strategies:
                current_forbidden = list(existing_strategies)
                PrintLogger.print_info(f"Retry {i+1}: Injecting Negative Constraint -> {current_forbidden}")

            # 调用生成函数，传入 forbidden_strategies
            res = generate_system_response(
                self.env, context, target_map, 
                self.order_change, 
                temperature=self.branching_temperature, # 建议设为 1.0 或 1.2
                enable_thinking=self.enable_thinking,
                forbidden_strategies=current_forbidden # <--- 传入约束
            )
            
            if not res: continue

            strat = res['strategy_name'].strip()
            
            # --- 查重逻辑 ---
            is_duplicate = False
            for exist in existing_strategies:
                # 双向包含检测，防止 "Logical Appeal" 和 "Logical" 被算作不同
                # 同时忽略大小写
                if strat.lower() in exist.lower() or exist.lower() in strat.lower():
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # 成功生成了新策略
                turn_metrics = calculate_turn_metrics(res['metrics'])
                
                return {
                    "role": self.sys_role,
                    "content": res['response'],
                    "strategy_name": strat,
                    "strategy_reason": res['strategy_reason'],
                    "state_analysis": res['state_analysis'],
                    "metrics": res['metrics'], 
                    "hs": turn_metrics['hs'],
                    "ha": turn_metrics['ha'],
                    "round": len(context) // 2 + 1 
                }
            else:
                if current_forbidden:
                    print(f"    [Warning] Model ignored constraint. Generated duplicate: {strat}")
        
        return None

    def _execute_branching(self, parent_traj, split_index, existing_strategy, 
                        trigger_info=None, num_to_generate=None, override_branch_idx=None):
        """
        执行分叉：生成 K-1 条新路径
        override_branch_idx: 如果不为None，强制使用该索引生成ID (用于补齐旧数据)
        """
        # 1. 构建 Context (分叉点之前的历史)
        actual_turn_num = parent_traj['turns'][split_index]['round']
        print(f"  Actual Turn Num: {actual_turn_num}")
        context_turns = parent_traj['turns'][:split_index]
        
        # 2. 提取已有的策略（用于去重）
        existing_strategies = set()
        if existing_strategy: 
            existing_strategies.add(existing_strategy.strip())

        generated_strategies_log = [] # 用于日志记录

        # 计算需要生成的数量
        count = num_to_generate if num_to_generate is not None else (self.K - 1)
        
        # 计算已存在的偏移量 (用于生成唯一的 ID 后缀)
        if num_to_generate is not None:
            # 补齐模式：假设总共要有 K-1 个，现在生成剩下的
            # 偏移量 = (目标总数) - 本次生成数 = 已存在的数量
            # 例如 K=5 (目标4个), 本次生成2个, 说明已有2个, offset=2 (0,1 已占, 新增 2,3)
            existing_count = (self.K - 1) - count
        else:
            # 全量模式：从 0 开始
            existing_count = 0

        # 确定 ID 中使用的 batch index (例如 root_b{0}_x)
        current_branch_idx = override_branch_idx if override_branch_idx is not None else self.branch_count

        # 3. 循环生成指定数量的新路径
        for k in range(count):
            # ID 后缀索引
            suffix_idx = k + existing_count
            
            PrintLogger.print_step(f"  > Generating Branch {k+1}/{count} (ID: b{current_branch_idx}_{suffix_idx})...")
            
            filtered_strategies_map = {
                k: v for k, v in self.strategies_map.items() 
                if k not in existing_strategies
            }

            if not filtered_strategies_map:
                print("    All strategies exhausted. Using full set.")
                filtered_strategies_map = self.strategies_map

            # A. Rejection Sampling 生成新的一步
            # 强制使用高温度 (1.2) 确保多样性
            new_turn_data = self._generate_distinct_step(context_turns, 
                                existing_strategies, #
                                strategy_map_override=filtered_strategies_map)
            
            if not new_turn_data:
                print("    Failed to generate distinct strategy.")
                continue
                
            existing_strategies.add(new_turn_data['strategy_name'])
            generated_strategies_log.append(new_turn_data['strategy_name'])
            
            # B. Rollout 到结束
            is_succ, full_new_turns = self._rollout_to_end(context_turns + [new_turn_data])
            
            # C. 保存新轨迹
            new_traj = {
                "id": f"{parent_traj['id']}_b{current_branch_idx}_{suffix_idx}",
                "parent_id": parent_traj['id'],
                "branch_at_turn": actual_turn_num,
                "turns": full_new_turns,
                "metrics_log": [], 
                "success": is_succ,
            }
            self.trajectories.append(new_traj)
            
        # 记录日志 (仅当不是补齐或者有 trigger info 时)
        if trigger_info:
            self.logger.log_branching(
                user_id="Current User",
                mode=self.trigger.mode,
                turn_num=actual_turn_num,
                score=trigger_info.get('score', 0),
                reason=trigger_info.get('reason', 'Supplement' if override_branch_idx is not None else 'New'),
                old_strat=existing_strategy,
                new_strats=generated_strategies_log
            )

    def _rollout_to_end(self, current_turns):
        """让对话继续直到结束"""
        # 最大轮数限制
        MAX_TURNS = 10
        
        # 复制当前路径
        rollout_turns = copy.deepcopy(current_turns)
        
        # 此时 current_turns 最后一步是 System (刚生成的)
        # 所以下一步应该是 User
        
        while True:
            current_round = len(rollout_turns) // 2 + 1
            if current_round > MAX_TURNS: break
            
            # 1. User Turn
            # 构造 Chat History 用于 API 调用
            # history_for_api = [{"role": t['role'], "content": t['content']} for t in rollout_turns]
            
            user_res = get_user_response(self.env, rollout_turns, self.user_profile, self.sys_role, 
                                        self.user_role, use_local_api=self.use_local_api)
            rollout_turns.append({
                "role": self.user_role,
                "content": user_res,
                "round": current_round
            })
            
            # Check Reward / End
            reward = get_reward_response(self.env, rollout_turns, self.user_profile, 
                                        self.sys_role, self.user_role, use_local_api=self.use_local_api)
            if reward >= 1.0:
                # 成功，结束
                rollout_turns[-1]['reward'] = 1.0
                break
            else:
                rollout_turns[-1]['reward'] = reward
                
            if current_round >= MAX_TURNS: break

            # 2. System Turn
            # 更新 history
            # history_for_api = [{"role": t['role'], "content": t['content']} for t in rollout_turns]

            sys_res = generate_system_response(
                self.env, rollout_turns, self.strategies_map, self.order_change,
                enable_thinking=self.enable_thinking
            )
            
            
            turn_metrics = calculate_turn_metrics(sys_res['metrics'])
            
            rollout_turns.append({
                "role": self.sys_role,
                "content": sys_res['response'],
                "strategy_name": sys_res['strategy_name'],
                "metrics": sys_res['metrics'],
                "hs": turn_metrics['hs'],
                "ha": turn_metrics['ha'],
                "round": current_round + 1
            })
        
        if rollout_turns[-1]['reward'] >= 1.0:
            is_succ = True
        else:
            is_succ = False
        return is_succ, rollout_turns

def evaluate_results(structure_data, output_file):
    """
    对 Rollout 结果进行统计分析
    structure_data: run_rollout_experiment 生成的结构化数据
    """
    PrintLogger.print_step(f"    > Evaluating Results")
    
    total_episodes = len(structure_data)
    if total_episodes == 0:
        print("No data to evaluate.")
        return

    # 1. 基础计数
    count_org_success = 0
    count_branched_success = 0 # Oracle Success (Org Success OR Any Branch Success)
    count_rescue_success = 0 # Org Fail -> Branch Success
    count_org_fail = 0
    
    # 2. 轮数统计 (仅统计成功的 Case)
    turns_org_success = []
    turns_branched_best = [] # 分叉后取最快成功的轮数
    
    # 3. 分叉统计
    total_branches = 0
    branch_turns_dist = {} # {turn_num: count}
    
    for user_data in structure_data:
        trajectories = user_data['trajectories']
        
        # A. 找到原始轨迹 (id='root')
        root_traj = next((t for t in trajectories if t['id'] == 'root'), None)
        if not root_traj: continue
        
        # 判断原始是否成功
        # 注意：这里我们假设最后一个 turn 如果有 reward=1.0 则成功
        # 或者我们需要在 rollout 时显式记录 success 标记
        # 现在的代码在 rollout_worker 返回 True/False，但在这里我们需要从 turns 推断
        # 简单起见，检查最后一轮是否有 reward=1.0 标记，或者我们重新检查 reward 函数
        # 为了稳健，我们检查最后一轮 User Response 是否包含 'yes' (模拟 get_reward)
        # 或者更简单的，我们在 rollout 生成时就把 result 写入了 traj
        
        # Hack: 重新检查 Root 的成功状态 (假设数据里没有显式字段)
        # 你的 rollout_worker 会返回 (is_succ, history)，但在 process 中只存了 history
        # 我们检查最后一轮是否 System 说 "Success" 或者有特殊标记
        # 暂时用一个简单的 Heuristic: 检查 Turns 里的 'reward' 字段
        # (需要在 rollout_worker 里确保写入了 'reward': 1.0)
        
        def _check_success(traj):
            if not traj['turns']: return False
            last_turn = traj['turns'][-1]
            return last_turn.get('reward', 0.0) >= 1.0

        is_org_success = _check_success(root_traj)
        
        if is_org_success:
            count_org_success += 1
            turns_org_success.append(len(root_traj['turns']) // 2) # System turns
        else:
            count_org_fail += 1
            
        # B. 检查分叉轨迹 (Oracle)
        # 收集所有成功的轨迹（包括 Root）
        all_success_turns = []
        if is_org_success:
            all_success_turns.append(len(root_traj['turns']) // 2)
            
        has_any_branch_success = False
        
        # 统计分叉点
        # 遍历所有非 root 轨迹
        for traj in trajectories:
            if traj['id'] == 'root': continue
            
            total_branches += 1
            b_turn = traj.get('branch_at_turn', -1)
            # System turn index to Round number (0->1, 1->2)
            # split_index 是 list index，round = index + 1
            if b_turn != -1:
                r_num = b_turn + 1
                branch_turns_dist[r_num] = branch_turns_dist.get(r_num, 0) + 1
            
            if _check_success(traj):
                has_any_branch_success = True
                all_success_turns.append(len(traj['turns']) // 2)
        
        # C. 综合统计
        if is_org_success or has_any_branch_success:
            count_branched_success += 1
            if all_success_turns:
                turns_branched_best.append(min(all_success_turns)) # 取最快成功的
        
        if not is_org_success and has_any_branch_success:
            count_rescue_success += 1

    # 计算指标
    metrics = {
        "total_episodes": total_episodes,
        "success_rate": {
            "original": count_org_success / total_episodes,
            "branched_oracle": count_branched_success / total_episodes,
            "improvement": (count_branched_success - count_org_success) / total_episodes
        },
        "rescue_rate": count_rescue_success / count_org_fail if count_org_fail > 0 else 0.0,
        "efficiency": {
            "avg_turns_org": np.mean(turns_org_success) if turns_org_success else 0.0,
            "avg_turns_branched_best": np.mean(turns_branched_best) if turns_branched_best else 0.0,
            "avg_branches_per_episode": total_branches / total_episodes
        },
        "branch_distribution": branch_turns_dist
    }

    print(json.dumps(metrics, indent=2))
    
    # 保存
    eval_file = output_file.replace(".json", "_eval.json")
    with open(eval_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation saved to {eval_file}")

# ==================== 主流程 ====================
def run_rollout_experiment(input_dir, output_file, mode, K=3, N=5, 
                         use_local_api=False, order_change=False, branching_temperature=1.0, logger=None,
                         enable_thinking=False, target_users=None):
    PrintLogger.print_step(f"    > Running Rollout Experiment")
    
    # 1. 加载源数据 (Seed Data)
    summary_file = os.path.join(input_dir, "summary_stats.json")
    with open(summary_file, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
        
    global_stats = summary_data['global_stats']
    all_episodes = summary_data['episodes']
    
    # 2. 初始化结果容器与断点续传
    final_results = []
    
    # 建立 user_id -> existing_result 的索引，用于快速查找
    existing_user_map = {}
    
    if os.path.exists(output_file):
        print(f"Resuming from existing file: {output_file}")
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                final_results = existing_data
                for item in existing_data:
                    existing_user_map[item['user_id']] = item
            print(f"Loaded {len(final_results)} existing user records.")
        except json.JSONDecodeError:
            print("Existing file corrupted or empty, starting fresh.")

    # 3. 过滤待处理用户
    # 注意：这里我们不跳过已存在的用户，而是把它们加入处理列表，
    # 让 Manager 决定是“跳过”（如果 K/N 满足）还是“增量补齐”（如果 K 变大了）
    episodes_to_run = []
    for ep in all_episodes:
        uid = ep['user_id']
        
        # 过滤: 如果指定了 target_users 且当前用户不在其中，彻底跳过
        if target_users and uid not in target_users:
            continue
            
        episodes_to_run.append(ep)
    
    if not episodes_to_run:
        print("No episodes matched criteria. Exiting.")
        return

    # 初始化 Trigger 和其他配置
    trigger = TriggerDetector(mode, global_stats)
    env = "P4G"
    strategies_map = human_persuader_strategy_instruction_map

    # 4. 主循环
    for ep_idx, episode in enumerate(tqdm(episodes_to_run, desc="Processing Episodes")):
        user_id = episode['user_id']
        PrintLogger.print_step(f"Processing Episode {ep_idx+1}/{len(episodes_to_run)}", user_id)
        
        user_profile = episode['user_profile']
        
        # 初始化 Manager
        # 注意：这里传入的是当前的配置 K 和 N
        manager = RolloutManager(
            env=env,
            user_profile=user_profile,
            sys_role="Persuader",
            user_role="Persuadee",
            strategies_map=strategies_map,
            trigger_detector=trigger,
            max_n_branches=N,
            k_branches_per_node=K,
            use_local_api=use_local_api,
            order_change=order_change,
            branching_temperature=branching_temperature,
            logger=logger,
            enable_thinking=enable_thinking
        )
        
        # 检查是否存在历史记录
        existing_res = existing_user_map.get(user_id)
        
        if existing_res:
            # [关键] 传入已有的轨迹数据，Manager 会检查是否需要补齐 K 值
            manager.load_existing_trajectories(
                existing_res['trajectories'], 
                episode['episode'],      # 原始种子详情（用于补全 context）
                episode.get('success')   # 原始成功状态
            )
        else:
            # [关键] 全新开始，加载种子
            manager.add_seed_trajectory(episode['episode'], episode.get('success'))
        
        # 执行处理 (扫描 -> 补齐 K -> 寻找新分叉)
        try:
            all_trajectories = manager.process()
            
            # 构建结果对象
            result_entry = {
                "user_id": user_id,
                "trajectories": all_trajectories
            }
            
            # 更新结果列表 (如果已存在则替换，不存在则追加)
            found = False
            for i, res in enumerate(final_results):
                if res['user_id'] == user_id:
                    final_results[i] = result_entry
                    found = True
                    break
            if not found:
                final_results.append(result_entry)
            
            # [关键] 增量保存 (每次处理完一个用户就写盘，防止崩溃丢失)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"[Error] Failed to process user {user_id}: {e}")
            traceback.print_exc()
            # 出错时不中断整个程序，继续下一个用户
            continue

    # 5. 保存最终的大文件 (Metrics 分离优化)
    # 这一步是为了生成方便分析的轻量级文件 (_results.json) 和包含大体积 Metrics 的文件 (_metrics.json)
    
    print("Finalizing data structure split...")
    metrics_only_data = []
    structure_data = []
    
    for res in final_results:
        # 深拷贝一份用于结构存储
        struct_res = copy.deepcopy(res)
        metrics_res = {"user_id": res['user_id'], "trajectories": []}
        
        for traj in struct_res['trajectories']:
            traj_metrics = []
            # 将 metrics 从 struct_res 中剥离出来，放入 metrics_res
            for turn in traj['turns']:
                if 'metrics' in turn:
                    # 提取并移除 metrics 字段，减小主文件体积
                    m = turn.pop('metrics')
                    traj_metrics.append({"round": turn.get('round'), "metrics": m})
            
            metrics_res['trajectories'].append({"id": traj['id'], "metrics": traj_metrics})
            
        structure_data.append(struct_res)
        metrics_only_data.append(metrics_res)

    # 保存结构文件
    print(f"Saving final structure to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(structure_data, f, indent=2, ensure_ascii=False)
        
    # 保存 Metrics 文件
    metrics_file = output_file.replace(".json", "_metrics.json")
    existing_metrics_map = {}
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
                for item in metrics_data:
                    existing_metrics_map[item['user_id']] = item
        except json.JSONDecodeError:
            print("Existing metrics file corrupted.")
    print(f"Saving final metrics to {metrics_file}...")
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_only_data, f, indent=2, ensure_ascii=False)
        
    # 6. 调用评估脚本
    evaluate_results(structure_data, output_file)
def main():
    parser = argparse.ArgumentParser()
    # input_dir 应该是包含 summary_stats.json 的目录
    parser.add_argument('--input_dir', type=str, required=True, help="Path to processed data directory")
    parser.add_argument('--mode', type=str, default="dissonance", choices=["random", "action_entropy", "state_entropy", "dissonance", "exp"], help="Trigger mode")
    parser.add_argument('--use_local_api', type=str2bool, default=True)
    parser.add_argument('--order_change', type=str2bool, default=False)
    parser.add_argument('--enable_thinking', type=str2bool, default=False, help='Whether to enable thinking')
    # 实验参数
    parser.add_argument('--K', type=int, default=3, help='Total branches (including original)')
    parser.add_argument('--N', type=int, default=3, help='Max branch points per episode')
    
    # 采样参数
    parser.add_argument('--branching_temperature', type=float, default=1.0, help='Temperature for nucleus sampling')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p for nucleus sampling')

    args = parser.parse_args()
    
    from utils.Prompt import EXP_ALL_USERS, EXP_ALL_USERS_TEST
    target_users = EXP_ALL_USERS_TEST # 导入实验用户集合

    version = "v1"  # 版本号，可根据需要修改
    output_file = os.path.join(args.input_dir, f"rollout_{version}_t_{args.branching_temperature}_thinking_{args.enable_thinking}/results_{args.mode}_t_{args.branching_temperature}_test.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    beijing_tz = pytz.timezone('Asia/Shanghai')
    now = datetime.datetime.now(beijing_tz).strftime("%Y-%m-%dT%H-%M-%S")
    log_file = os.path.join(os.path.dirname(output_file), f"rollout_log_{args.mode}_{now}.txt")
    logger = ExperimentLogger(log_file) # 初始化 Logger

    run_rollout_experiment(
        args.input_dir, output_file, args.mode, 
        K=args.K, N=args.N,
        use_local_api=args.use_local_api, 
        order_change=args.order_change,
        enable_thinking=args.enable_thinking,
        branching_temperature=args.branching_temperature,
        logger=logger,
        target_users=target_users
    )

if __name__ == "__main__" :
    main()