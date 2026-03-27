import json
import os, sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import difflib
# 设置绘图风格
sns.set_theme(style="whitegrid")

# from utils.api_call import call_llm_chat_api_openai 
from tqdm import tqdm
import concurrent.futures

import warnings
warnings.filterwarnings("ignore")

# 标准策略集合
HUMAN_PERSUADER_STRATEGIES = sorted([
    "Greeting",
    "Logical Appeal",
    "Emotion Appeal",
    "Credibility Appeal",
    "Self Modeling",
    "Foot in the Door",
    "Personal Story",
    "Donation Information",
    "Source Related Inquiry",
    "Task Related Inquiry",
    "Personal Related Inquiry"
])

USER_RESISTANCE_STRATEGIES = sorted([
    "Donate",
    "Source Derogation",
    "Counter Argument",
    "Personal Choice",
    "Information Inquiry",
    "Self Pity",
    "Hesitance",
    "Self-assertion",
    "Others"
])

current_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_path)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from utils.api_call import call_llm_chat_api_openai 

def generate_response(messages, model_name="gpt-4o", temperature=1.1, max_tokens=50, n=10, use_local_api=False, local_api_url="http://localhost:8093/inference"):
    generation_payload = {
        "model_name": model_name,
        "messages": messages,
        "temperature": temperature,
        "n": n,
        "max_tokens": 25
    }
    local_api_url = "http://localhost:8093/inference"
    generation_payload['use_local_api'] = use_local_api
    generation_payload['local_api_url'] = local_api_url
    try:
        results = call_llm_chat_api_openai(**generation_payload)
        print(results)
        return results
    except:
        return 0

class UserStrategyLabeler:
    """
    负责调用 LLM 为用户的回复打上策略标签
    """
    def __init__(self, api_url, model_name="gpt-4o", max_workers=10, num_votes=10):
        self.api_url = api_url
        self.model_name = model_name
        self.max_workers = max_workers
        self.num_votes = num_votes # 多数投票次数
        
    def _construct_prompt(self, history, current_utterance):
        """构造用于分类的 Prompt"""
        # 将历史对话拼接成文本
        history_text = ""
        for turn in history[-3:]: # 只取最近几轮作为上下文，节省 tokens
            role = "Persuader" if turn['role'] == "Persuader" else "Persuadee"
            history_text += f"{role}: {turn['content']}\n"
            
        prompt = f"""
You are an expert annotator for persuasion dialogues.
Your task is to classify the **Persuadee's Response** into one of the following resistance strategies:

1. "Donate": show your willingness to donate. 
2. "Source Derogation": attacks or doubts the organisation’s credibility. 
3. "Counter Argument": argues that the responsibility is not on them or refutes a previous statement. 
4. "Personal Choice": Attempts to saves face by asserting their personal preference such as their choice of charity and their choice of donation. 
5. "Information Inquiry": Ask for factual information about the organisation for clarification or as an attempt to stall. 
6. "Self Pity": Provides a self-centred reason for not being willing to donate at the moment.
7. "Hesitance": Attempts to stall the conversation by either stating they would donate later or is currently unsure about donating. 
8. "Self-assertion": Explicitly refuses to donate without even providing a personal reason. 
9. "Others": Do not explicitly foil the persuasion attempts.  

Dialogue Context:
{history_text}

Persuadee's Response to Classify:
"{current_utterance}"

Output ONLY the strategy name from the list above. Do not output anything else.
Strategy:
"""
        return [{"role": "user", "content": prompt}]

    def _get_label_single(self, history, utterance):
        messages = self._construct_prompt(history, utterance)

        print(messages)
        # 投票多次
        votes = []
        temp = 1.1
        results = generate_response(messages, model_name=self.model_name, temperature=temp, n=self.num_votes)
        for r in results:
            clean_res = self._clean_strategy(r)
            votes.append(clean_res)

        # 多数投票
        counter = Counter(votes)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _clean_strategy(self, text):
        """清洗 LLM 输出，匹配到标准集合"""
        text = text.strip().lower()
        # 模糊匹配
        best_match = "Other"
        max_ratio = 0.0
        
        for strat in USER_RESISTANCE_STRATEGIES:
            ratio = difflib.SequenceMatcher(None, text, strat.lower()).ratio()
            if ratio > max_ratio:
                max_ratio = ratio
                best_match = strat
        
        # 阈值过滤，如果匹配度太低归为 Other
        if max_ratio < 0.3: return "Other"
        return best_match

    def label_dataset(self, episodes):
        """
        并行处理整个数据集
        episodes: list of dict (包含 trajectories 或 detailed_history)
        """
        tasks = []
        skipped_count = 0  # [新增] 统计跳过的数量
        
        print("Preparing labeling tasks...")
        # 1. 收集所有需要标注的 Turn
        for ep_idx, ep in enumerate(episodes):
            # 兼容两种数据格式
            if 'trajectories' in ep:
                trajs = ep['trajectories']
            elif 'detailed_history' in ep:
                # 为了统一处理，将其包装成一个伪 Trajectory 对象
                trajs = [{'id': 'root_flat', 'turns': ep['detailed_history']}]
            else:
                continue
                
            for traj in trajs:
                history_accum = []
                for t_idx, turn in enumerate(traj['turns']):
                    role = turn['role']
                    content = turn['content']
                    
                    if role == "Persuadee": # 只标注用户的回复
                        # [新增] 检查是否已存在标注
                        # 如果 user_strategy 存在且不为空，则跳过
                        if turn.get('user_strategy'):
                            skipped_count += 1
                        else:
                            # 只有未标注的才加入任务队列
                            hist_snap = list(history_accum) 
                            tasks.append({
                                "ep_idx": ep_idx,
                                "traj_id": traj['id'],
                                "turn_idx": t_idx,
                                "history": hist_snap,
                                "content": content
                            })
                    
                    # [关键] 无论是否跳过标注，都必须将当前轮次加入历史上下文
                    # 否则后续轮次会丢失 Context
                    history_accum.append(turn)

        print(f"Total user turns to label: {len(tasks)}")
        print(f"Skipped existing labels: {skipped_count}") # [新增] 打印跳过信息
        
        if len(tasks) == 0:
            print("All turns are already labeled. Returning directly.")
            return episodes
        
        # 2. 并发执行 (保持不变)
        results = {} 
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._get_label_single, t['history'], t['content']): t 
                for t in tasks
            }
            
            # 使用 tqdm 显示进度
            for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Labeling Strategies"):
                task = future_to_task[future]
                try:
                    label = future.result()
                    key = (task['ep_idx'], task['traj_id'], task['turn_idx'])
                    results[key] = label
                except Exception as e:
                    print(f"Task failed: {e}")

        print(f"Labeling finished. Updating {len(results)} turns...")

        # 3. 回填标签到原始数据 (保持不变)
        for (ep_idx, traj_id, t_idx), label in results.items():
            ep = episodes[ep_idx]
            target_turns = None
            
            if 'trajectories' in ep:
                for t in ep['trajectories']:
                    if t['id'] == traj_id:
                        target_turns = t['turns']
                        break
            elif 'detailed_history' in ep and traj_id == 'root_flat':
                 target_turns = ep['detailed_history']
            
            if target_turns is not None and t_idx < len(target_turns):
                target_turns[t_idx]['user_strategy'] = label
                 
        return episodes

class StrategyNormalizer:
    def __init__(self, valid_strategies):
        self.valid_strategies = list(valid_strategies)
        # 预处理：生成小写到标准名的映射，方便快速查找
        self.lower_map = {s.lower(): s for s in self.valid_strategies}

    def normalize(self, raw_strat):
        if not raw_strat:
            return "Unknown"
        
        # 1. 简单清洗
        clean = raw_strat.replace('[', '').replace(']', '').replace('"', '').replace("'", "").strip()
        
        # 2. 精确匹配 (忽略大小写)
        if clean.lower() in self.lower_map:
            return self.lower_map[clean.lower()]
        
        # 3. 模糊匹配
        # cutoff=0.6 表示相似度阈值，越低越宽松
        matches = difflib.get_close_matches(clean, self.valid_strategies, n=1, cutoff=0.6)
        if matches:
            return matches[0]
            
        # 4. 尝试部分匹配 (例如 "Use Foot-in-the-Door" -> "Foot in the Door")
        for valid in self.valid_strategies:
            # 移除空格和连字符进行比较
            v_norm = valid.lower().replace(" ", "").replace("-", "")
            c_norm = clean.lower().replace(" ", "").replace("-", "")
            if v_norm in c_norm or c_norm in v_norm:
                return valid
                
        return "Unknown"

class InteractionAnalyzer:
    def __init__(self, exp_data):
        self.data = exp_data
        
    def _normalize_strategy(self, text, strategy_set):
        """简单的模糊匹配标准化"""
        if not text: return None
        text = text.strip()
        # 精确匹配
        if text in strategy_set: return text
        # 包含匹配
        for s in strategy_set:
            if s.lower() in text.lower(): return s
        return "Others" if "Others" in strategy_set else None

    def analyze_context_triplets(self):
        """
        分析上下文三元组: (S_{t-1}, U_{t-1}) -> S_t
        输出: 
        1. 该组合导致的最终胜率 (Win Rate)
        2. 该组合引发的下一轮用户反应 (Next User Strategy Distribution)
        """
        triplet_stats = defaultdict(lambda: {'total': 0, 'success': 0, 'next_user_responses': Counter()})

        for ep in self.data:
            root_traj = next((t for t in ep['trajectories'] if t['id'] == 'root'), None)
            if not root_traj: continue
            
            is_succ = root_traj.get('success', False) or \
                      (root_traj['turns'] and root_traj['turns'][-1].get('reward', 0) >= 1.0)
            
            turns = root_traj['turns']
            
            # 使用滑动窗口提取 S_prev, U_prev, S_curr, U_curr
            # 假设标准顺序: S0, U0, S1, U1, S2, U2...
            
            # 提取所有有效的策略序列
            sequence = []
            for turn in turns:
                role = turn['role']
                if role == 'Persuader':
                    s = self._normalize_strategy(turn.get('strategy_name'), HUMAN_PERSUADER_STRATEGIES)
                    if s: sequence.append({'type': 'S', 'val': s})
                elif role == 'Persuadee':
                    u = self._normalize_strategy(turn.get('user_strategy'), USER_RESISTANCE_STRATEGIES)
                    if u: sequence.append({'type': 'U', 'val': u})
            
            # 遍历序列寻找 S-U-S 模式
            # 索引 i: S_prev, i+1: U_prev, i+2: S_curr, i+3: U_curr (Next reaction)
            for i in range(len(sequence) - 2):
                if sequence[i]['type'] == 'S' and \
                   sequence[i+1]['type'] == 'U' and \
                   sequence[i+2]['type'] == 'S':
                    
                    s_prev = sequence[i]['val']
                    u_prev = sequence[i+1]['val']
                    s_curr = sequence[i+2]['val']
                    
                    key = (s_prev, u_prev, s_curr)
                    
                    triplet_stats[key]['total'] += 1
                    if is_succ:
                        triplet_stats[key]['success'] += 1
                    
                    # 记录下一轮用户的反应 (Immediate Feedback)
                    if i + 3 < len(sequence) and sequence[i+3]['type'] == 'U':
                        u_next = sequence[i+3]['val']
                        triplet_stats[key]['next_user_responses'][u_next] += 1

        return triplet_stats
    def analyze_transitions(self, filter_success=None):
        """
        计算转移矩阵
        filter_success: True (只看成功), False (只看失败), None (全部)
        """
        # 1. System -> User (Effectiveness)
        sys_to_user = pd.DataFrame(0, index=HUMAN_PERSUADER_STRATEGIES, columns=USER_RESISTANCE_STRATEGIES)
        # 2. User -> System (Adaptability)
        user_to_sys = pd.DataFrame(0, index=USER_RESISTANCE_STRATEGIES, columns=HUMAN_PERSUADER_STRATEGIES)
        
        for ep in self.data:
            # 过滤轨迹
            root_traj = next((t for t in ep['trajectories'] if t['id'] == 'root'), None)
            if not root_traj: continue
            
            is_succ = root_traj.get('success', False) or (root_traj['turns'][-1].get('reward', 0) >= 1.0)
            if filter_success is not None and is_succ != filter_success:
                continue

            turns = root_traj['turns']
            
            # 遍历轮次
            # 假设 turns 顺序是: Sys(1), User(1), Sys(2), User(2)...
            # 或者是混合列表，需要按顺序解析
            
            last_sys_strat = None
            last_user_strat = None
            
            for i, turn in enumerate(turns):
                role = turn['role']
                
                if role == 'Persuader':
                    curr_sys = self._normalize_strategy(turn.get('strategy_name'), HUMAN_PERSUADER_STRATEGIES)
                    if not curr_sys: continue
                    
                    # 记录 User -> Sys 转移 (上一轮用户导致了这一轮系统)
                    if last_user_strat:
                        user_to_sys.loc[last_user_strat, curr_sys] += 1
                    
                    last_sys_strat = curr_sys
                    
                elif role == 'Persuadee':
                    curr_user = self._normalize_strategy(turn.get('user_strategy'), USER_RESISTANCE_STRATEGIES)
                    if not curr_user: continue
                    
                    # 记录 Sys -> User 转移 (上一轮系统导致了这一轮用户)
                    if last_sys_strat:
                        sys_to_user.loc[last_sys_strat, curr_user] += 1
                        
                    last_user_strat = curr_user

        return sys_to_user, user_to_sys

    def analyze_pivotal_context(self, pivot_df):
        """
        分析关键节点发生时的上下文 (用户策略)
        pivot_df: 之前计算出的包含 Phi Score 的 DataFrame
        """
        # 将 Pivot Data 映射回具体的 User Strategy
        # 这需要 exp_data 和 pivot_df 能够 join
        # 简单起见，我们重新遍历 exp_data，如果该轮次是高 Phi，记录 User Strategy
        
        context_scores = defaultdict(list) # {User_Strat: [Phi_Score, ...]}
        
        # 建立快速查找表: (User, Turn) -> Phi
        pivot_map = {}
        for _, row in pivot_df.iterrows():
            if row['Scenario'] == 'Rescue': # 只关注挽救潜力
                pivot_map[(row['User'], row['Turn'])] = row['Phi_Score']
        
        for ep in self.data:
            user_id = ep.get('user_id')
            root_traj = next((t for t in ep['trajectories'] if t['id'] == 'root'), None)
            if not root_traj: continue
            
            # 找到每一轮的前置 User Strategy
            # Pivot 发生在 System Turn T。我们需要的是 Turn T 之前的 User Turn T-1
            # 或者是当前 System Turn 之后 User 的反应？
            # 修正：关键节点是 System 的决策点。决策依据是“上一轮用户的反应”。
            
            # 提取 User 策略序列
            user_strats = {} # {turn_num: strat}
            current_turn_num = 0
            
            for turn in root_traj['turns']:
                if turn['role'] == 'Persuadee':
                    # User 回复是在 System Turn X 之后，通常算作 Turn X 的结果
                    # 或者作为 Turn X+1 的输入
                    # 假设 turns 列表顺序: Sys(1), User(1), Sys(2), User(2)
                    # Sys(2) 的输入是 User(1). User(1) 的 round 标记通常是 1
                    pass 
            
            # 更简单的遍历：
            # 在 Sys Turn T 时，取前一个 User Turn
            last_user_s = "Start"
            for turn in root_traj['turns']:
                if turn['role'] == 'Persuadee':
                    last_user_s = self._normalize_strategy(turn.get('user_strategy'), USER_RESISTANCE_STRATEGIES)
                
                elif turn['role'] == 'Persuader':
                    turn_num = turn['round']
                    phi = pivot_map.get((user_id, turn_num))
                    
                    if phi is not None and last_user_s:
                        context_scores[last_user_s].append(phi)

        return context_scores

class Visualizer:
    def __init__(self, name="Model", labeler=None):
        self.model_name = name
        self.labeler = labeler

    def plot_interaction_heatmap(self,df, title, output_path):
        # 归一化 (Row-wise: P(Col | Row))
        # 加上 epsilon 防止除零，或者 fillna
        df_norm = df.div(df.sum(axis=1), axis=0).fillna(0)
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(df_norm, cmap="YlGnBu", annot=True, fmt=".2f", cbar=True,
                    xticklabels=df.columns, yticklabels=df.index)
        plt.title(title, fontsize=14)
        plt.xlabel("Next Action")
        plt.ylabel("Condition (Previous Action)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_pivotal_context(self, context_scores, output_path):
        # 转换数据格式用于绘图
        data = []
        for strat, scores in context_scores.items():
            for s in scores:
                data.append({'User Resistance': strat, 'Rescue Potential': s})
        
        df = pd.DataFrame(data)
        
        # 按平均分排序
        order = df.groupby('User Resistance')['Rescue Potential'].mean().sort_values(ascending=False).index
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='User Resistance', y='Rescue Potential', order=order, palette="viridis")
        plt.xticks(rotation=45)
        plt.title("Which User Resistance Offers the Best Rescue Opportunity?")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_triplet_analysis(self, triplet_stats, output_dir, top_n_contexts=5):
        """
        可视化三元组分析结果。
        由于组合太多，我们只挑选出现频率最高的 Top N 个 Context (S_prev, U_prev) 进行展示。
        展示在这些 Context 下，不同 S_curr 的胜率。
        """
        # 1. 聚合 Context: (S_prev, U_prev) -> Total Count
        context_counts = defaultdict(int)
        for (sp, up, sc), stat in triplet_stats.items():
            context_counts[(sp, up)] += stat['total']
        
        # 选出 Top N Contexts
        top_contexts = sorted(context_counts.items(), key=lambda x: x[1], reverse=True)[:top_n_contexts]
        
        if not top_contexts:
            print("No triplet data found.")
            return

        # 2. 为每个 Top Context 画一张热力图或条形图
        for (sp, up), count in top_contexts:
            context_str = f"Prev: {sp} -> User: {up}"
            
            # 收集该 Context 下所有 S_curr 的数据
            data = []
            for sc in HUMAN_PERSUADER_STRATEGIES:
                key = (sp, up, sc)
                if key in triplet_stats:
                    stat = triplet_stats[key]
                    win_rate = stat['success'] / stat['total'] if stat['total'] > 0 else 0
                    
                    # 找出最常见的用户反应
                    most_common_next_u = "None"
                    if stat['next_user_responses']:
                        most_common_next_u = stat['next_user_responses'].most_common(1)[0][0]
                    
                    data.append({
                        'Current Strategy': sc,
                        'Win Rate': win_rate,
                        'Count': stat['total'],
                        'Most Likely Reaction': most_common_next_u
                    })
            
            if not data: continue
            
            df = pd.DataFrame(data)
            # 过滤掉 Count 为 0 的显示，或者保留但 Rate 为 0
            df = df[df['Count'] > 0].sort_values('Win Rate', ascending=False)
            
            # 绘图：胜率条形图
            plt.figure(figsize=(12, 6))
            barplot = sns.barplot(data=df, x='Current Strategy', y='Win Rate', palette='viridis')
            
            # 在柱子上标注 (Count | Next User Reaction)
            for index, row in df.iterrows():
                # 这里 index 是 DataFrame 的 index，需要用 enumerate 获取 bar 的位置
                pass 
            
            # 重新遍历添加标签
            for i, p in enumerate(barplot.patches):
                if i < len(df):
                    row = df.iloc[i]
                    label = f"N={row['Count']}\n->{row['Most Likely Reaction']}"
                    barplot.annotate(label, 
                                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                                    ha='center', va='bottom', fontsize=9, rotation=0, xytext=(0, 5), 
                                    textcoords='offset points')

            plt.title(f"Context: [{sp}] -> User Said [{up}] (Total N={count})", fontsize=14)
            plt.ylabel("Dialogue Success Rate")
            plt.ylim(0, 1.1)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            safe_name = f"{sp}_{up}".replace(" ", "_")
            save_path = os.path.join(output_dir, f"{self.model_name}_triplet_context_{safe_name}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved triplet plot: {save_path}")

    def export_triplet_stats(self, triplet_stats, output_dir):
        """
        导出所有三元组统计数据到 CSV
        """
        rows = []
        for (sp, up, sc), stat in triplet_stats.items():
            total = stat['total']
            success = stat['success']
            win_rate = success / total if total > 0 else 0.0
            
            # 获取最常见的用户反应及比例
            next_user_reaction = "None"
            reaction_count = 0
            if stat['next_user_responses']:
                most_common = stat['next_user_responses'].most_common(1)[0]
                next_user_reaction = most_common[0]
                reaction_count = most_common[1]
            
            reaction_prob = reaction_count / total if total > 0 else 0

            rows.append({
                "Prev_System_Strategy": sp,
                "Prev_User_Resistance": up,
                "Curr_System_Strategy": sc,
                "Total_Count": total,
                "Success_Count": success,
                "Win_Rate": win_rate,
                "Most_Common_Next_User_Reaction": next_user_reaction,
                "Next_User_Reaction_Prob": reaction_prob
            })
            
        df = pd.DataFrame(rows)
        # 按出现频率和胜率排序
        df = df.sort_values(by=['Total_Count', 'Win_Rate'], ascending=[False, False])
        
        output_path = os.path.join(output_dir, f"{self.model_name}_triplet_stats.csv")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"Full statistics exported to: {output_path}")
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return json.load(f)
    
def process_data(raw_data):
    pass

def is_traj_success(traj):
    if traj.get('success'): return True
    if not traj.get('turns'): return False
    return traj['turns'][-1].get('reward', 0.0) >= 1.0

class StrategyAnalyzer:
    def __init__(self, data, name="Model", labeler=None):
        self.name = name
        self.data = data
        self.normalizer = StrategyNormalizer(HUMAN_PERSUADER_STRATEGIES)

        if labeler:
            print(f"[{name}] Starting User Strategy Labeling...")
            self.data = labeler.label_dataset(self.data)

        self.strategies = self._extract_strategies()
        
    def _extract_strategies(self):
        """
        提取 Persuader 和 Persuadee 的双边策略序列
        支持处理 Rollout 产生的多条轨迹
        """
        extracted = []
        
        for ep in self.data:
            user_id = ep.get('user_id', 'unknown')
            
            # 统一处理：将所有数据视为 trajectory 列表
            trajs = []
            if 'trajectories' in ep:
                trajs = ep['trajectories']
            elif 'detailed_history' in ep:
                trajs = [{'id': 'root', 'turns': ep['detailed_history'], 'success': ep.get('success', False)}]
            
            for traj in trajs:
                traj_id = traj.get('id', 'root')
                success = is_traj_success(traj) # 使用统一的判断函数
                
                turn_pairs = [] # [(round, sys_strat, user_strat), ...]
                
                # 遍历 turns，配对 (System, User)
                # 假设 turns 是有序的 Sys, User, Sys, User...
                current_sys_strat = None
                current_round = 0
                
                for turn in traj['turns']:
                    role = turn['role']
                    
                    if role == 'Persuader':
                        current_round = turn.get('round', 0)
                        raw_strat = turn.get('strategy_name') or turn.get('strategy') or "Unknown"
                        # 标准化 System Strategy
                        current_sys_strat = self.normalizer.normalize(raw_strat)
                        
                    elif role == 'Persuadee':
                        # 获取标注好的 User Strategy
                        user_strat = turn.get('user_strategy', "Unknown")
                        
                        if current_sys_strat:
                            turn_pairs.append({
                                'round': current_round,
                                'sys_strat': current_sys_strat,
                                'user_strat': user_strat
                            })
                            current_sys_strat = None # Reset
                
                extracted.append({
                    'user_id': user_id,
                    'traj_id': traj_id,
                    'success': success,
                    'turn_pairs': turn_pairs
                })
                
        return extracted

    def analyze_temporal_distribution(self):
        """D1: 分析前中后期的策略分布差异 (使用频率/Proportion)"""
        stages = {'Early (1-3)': [], 'Mid (4-6)': [], 'Late (7-10)': []}
        
        # 统计总数用于归一化
        total_counts = {'Success': defaultdict(int), 'Fail': defaultdict(int)}
        
        # 收集数据
        raw_data = []
        for ep in self.strategies:
            group = 'Success' if ep['success'] else 'Fail'
            for r, s in ep['turns']:
                if r <= 3: stage = 'Early (1-3)'
                elif r <= 6: stage = 'Mid (4-6)'
                else: stage = 'Late (7-10)'
                
                raw_data.append({'Stage': stage, 'Group': group, 'Strategy': s})
                total_counts[group][stage] += 1
        
        df = pd.DataFrame(raw_data)
        
        if df.empty:
            print(f"Warning: No valid strategy data for {self.name}")
            return plt.figure()

        # 计算频率 (Proportion)
        # GroupBy: Stage, Group, Strategy -> Count
        df_counts = df.groupby(['Stage', 'Group', 'Strategy']).size().reset_index(name='Count')
        
        # 归一化：除以该 Stage 该 Group 的总策略数
        def calc_prop(row):
            total = total_counts[row['Group']][row['Stage']]
            return row['Count'] / total if total > 0 else 0
            
        df_counts['Proportion'] = df_counts.apply(calc_prop, axis=1)
        
        # 绘图
        g = sns.catplot(
            data=df_counts, x='Strategy', y='Proportion', col='Stage', hue='Group', 
            kind='bar', col_wrap=3, height=5, aspect=1.4,
            palette="muted"
        )
        g.set_xticklabels(rotation=45, ha='right')
        g.fig.suptitle(f"{self.name}: Strategy Distribution per Stage (Normalized)", y=1.02)
        return g

    def analyze_transition_matrix(self, group_filter='Success'):
        """D2: 分析策略转移矩阵 (Bigram)"""
        transitions = []
        for ep in self.strategies:
            if group_filter == 'Success' and not ep['success']: continue
            if group_filter == 'Fail' and ep['success']: continue
            
            turns = ep['turns']
            for i in range(len(turns) - 1):
                curr_s = turns[i][1]
                next_s = turns[i+1][1]
                transitions.append((curr_s, next_s))
        
        if not transitions: return None

        src, tgt = zip(*transitions)
        # 使用标准集合作为索引，保证矩阵完整
        unique_strats = sorted(list(HUMAN_PERSUADER_STRATEGIES))
        matrix = pd.DataFrame(0, index=unique_strats, columns=unique_strats)
        
        for s, t in transitions:
            if s in unique_strats and t in unique_strats:
                matrix.loc[s, t] += 1
            
        # 归一化 (Row-wise normalization)
        matrix_norm = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix_norm, annot=False, cmap="YlGnBu", xticklabels=True, yticklabels=True)
        plt.title(f"{self.name} ({group_filter}): Strategy Transition Probabilities")
        plt.xlabel("Next Strategy")
        plt.ylabel("Current Strategy")
        plt.tight_layout()
        return plt

    def analyze_diversity(self):
        """D3: 分析策略多样性"""
        diversity_data = []
        for ep in self.strategies:
            strats = [t[1] for t in ep['turns']]
            if not strats: continue
            unique_ratio = len(set(strats)) / len(strats)
            diversity_data.append({
                'Group': 'Success' if ep['success'] else 'Fail',
                'Diversity': unique_ratio
            })
            
        df = pd.DataFrame(diversity_data)
        
        order = ['Fail', 'Success']
        # 指定颜色
        palette = {'Fail': sns.color_palette("Set2")[0], 'Success': sns.color_palette("Set2")[1]}
        plt.figure(figsize=(6, 5))
        sns.boxplot(
            data=df, 
            x='Group', 
            y='Diversity', 
            order=order,      # <--- 固定 X 轴顺序
            palette=palette   # <--- 固定颜色映射
        )
        plt.title(f"{self.name}: Strategy Diversity Score")
        return plt

    def analyze_pivot_strategies(self):
        """D4: 寻找胜负手策略"""
        strat_stats = defaultdict(lambda: {'succ_count': 0, 'fail_count': 0})
        
        for ep in self.strategies:
            used_strats = set([t[1] for t in ep['turns']])
            for s in used_strats:
                if ep['success']:
                    strat_stats[s]['succ_count'] += 1
                else:
                    strat_stats[s]['fail_count'] += 1
        
        plot_data = []
        for s, counts in strat_stats.items():
            total = counts['succ_count'] + counts['fail_count']
            if total < 5: continue 
            win_rate = counts['succ_count'] / total
            plot_data.append({'Strategy': s, 'Win Rate': win_rate, 'Count': total})
            
        df = pd.DataFrame(plot_data).sort_values('Win Rate', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='Win Rate', y='Strategy', palette='viridis')
        plt.axvline(0.5, color='red', linestyle='--')
        plt.title(f"{self.name}: Strategy Effectiveness (Win Rate)")
        return plt

    # 例如 analyze_transition_matrix 现在可以画 Sys -> User 的反应矩阵   
    def analyze_response_matrix(self, group_filter='Success'):
        """分析：系统使用了策略 A 后，用户大概率会用什么策略回应？"""
        pairs = []
        for ep in self.strategies:
            if group_filter == 'Success' and not ep['success']: continue
            if group_filter == 'Fail' and ep['success']: continue
            
            for tp in ep['turn_pairs']:
                pairs.append((tp['sys_strat'], tp['user_strat']))
        
        if not pairs: return None

        # 绘图逻辑与之前类似，只是 X 轴是 User Strat，Y 轴是 Sys Strat
        sys_strats = sorted(list(HUMAN_PERSUADER_STRATEGIES))
        user_strats = sorted(USER_RESISTANCE_STRATEGIES)
        
        matrix = pd.DataFrame(0, index=sys_strats, columns=user_strats)
        for s, u in pairs:
            if s in sys_strats and u in user_strats:
                matrix.loc[s, u] += 1
                
        # Row-wise normalization
        matrix_norm = matrix.div(matrix.sum(axis=1), axis=0).fillna(0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(matrix_norm, annot=False, cmap="YlGnBu")
        plt.title(f"{self.name} ({group_filter}): User Response Probabilities")
        plt.xlabel("User Resistance Strategy")
        plt.ylabel("System Persuasion Strategy")
        plt.tight_layout()
        return plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs='+', required=True, help="List of result json files")
    parser.add_argument("--names", nargs='+', required=True, help="Names for each file")
    parser.add_argument("--output_dir", default="strategy_analysis_deep", help="Output directory")
    # LLM Labeling 参数
    parser.add_argument("--api_url", default="http://127.0.0.1:8093/inference", help="API URL for labeling")
    parser.add_argument("--label_model", default="gpt-4o-mini", help="Model name for labeling")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers")
    parser.add_argument("--save_labeled", action="store_true", default=True, help="Save labeled data to new files")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化 Labeler
    labeler = UserStrategyLabeler(args.api_url, args.label_model, args.workers)
    
    analyzers = []
    for fpath, name in zip(args.files, args.names):
        print(f"Processing {name}...")
        data = load_data(fpath)
        if data:
            # 传入 labeler 自动进行标注
            az = StrategyAnalyzer(data, name, labeler=labeler)
            analyzers.append(az)

            if args.save_labeled:
                print(f"Updating original file with user strategies: {fpath}")
                try:
                    with open(fpath, 'w', encoding='utf-8') as f:
                        # az.data 已经是包含了 user_strategy 的完整数据
                        json.dump(az.data, f, indent=4, ensure_ascii=False)
                    print(f"Successfully updated {fpath}")
                except Exception as e:
                    print(f"Error saving file {fpath}: {e}")  
        
        ## 标注完成后进行分析
        ite_anaylyzer = InteractionAnalyzer(data)
        
        # 2. 分析交互 (成功组 vs 失败组)
        s2u_succ, u2s_succ = ite_anaylyzer.analyze_transitions(filter_success=True)
        s2u_fail, u2s_fail = ite_anaylyzer.analyze_transitions(filter_success=False)
        
        visualizer = Visualizer(name=name)
        # 3. 绘图：成功的交互模式
        visualizer.plot_interaction_heatmap(u2s_succ, "Success: How System Adapts to User (User -> Sys)", f"{args.output_dir}/{name}_u2s_success.png")
        # 4. 绘图：失败的交互模式 (对比用)
        visualizer.plot_interaction_heatmap(u2s_fail, "Fail: How System Adapts to User (User -> Sys)", f"{args.output_dir}/{name}_u2s_fail.png")

        print("Generating Triplet Context Analysis...")
        triplet_stats = ite_anaylyzer.analyze_context_triplets()
        visualizer.plot_triplet_analysis(triplet_stats, args.output_dir, top_n_contexts=5)
        visualizer.export_triplet_stats(triplet_stats, args.output_dir)
    # # 绘图
    # for az in analyzers:
    #     print(f"Plotting for {az.name}...")
    #     p = az.analyze_response_matrix('Success')
    #     if p: 
    #         p.savefig(os.path.join(args.output_dir, f"{az.name}_response_success.png"))
    #         plt.close()
        
    #     p = az.analyze_response_matrix('Fail')
    #     if p: 
    #         p.savefig(os.path.join(args.output_dir, f"{az.name}_response_fail.png"))
    #         plt.close()

if __name__ == "__main__":
    main()