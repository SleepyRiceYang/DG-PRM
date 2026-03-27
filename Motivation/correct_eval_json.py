import json, os

def add_success_state(episodes, all_user_data):
    for episode in episodes:
        for user_data in all_user_data:
            user_id = user_data["user_id"]
            if episode["user_id"] == user_id:
                assert episode["trajectories"][0]['id'] == "root"
                episode["trajectories"][0]['success'] = user_data["success"]
                break
    return episodes

# 合并两次十个用户的结果
def merge_user_results():
    mode = [
        "dissonance",
        "state_entropy",
        "action_entropy",
        "random"
    ]
    exp_1_dir = "/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0/rollout_v1"
    exp_2_dir = "/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0/rollout_v2"
    all_user_data_file = "/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0/results_unified_2025-12-22T15-55-18_False_False_t_0.0.json"

    # 处理 exp_1_dir 下的所有模式文件   
    for m in mode:
        episode_file = f"{exp_1_dir}/results_{m}_t_1.0.json"
        with open(all_user_data_file, "r") as f:
            all_user_data = json.load(f)
        with open(episode_file, "r") as f:
            episodes = json.load(f)
        new_episodes = add_success_state(episodes, all_user_data)
        with open(episode_file, "w") as f:
            json.dump(new_episodes, f, indent=4)
        print(f"✅ {m} done")
    
    result_dir = "/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0/rollout_v1_v2_merged"
    os.makedirs(result_dir, exist_ok=True)
    # 合并文件
    for m in mode:
        episode_file_1 = f"{exp_1_dir}/results_{m}_t_1.0.json"
        episode_file_2 = f"{exp_2_dir}/results_{m}_t_1.0.json"
        merged_file = f"{result_dir}/results_{m}_t_1.0.json"

        with open(episode_file_1, "r") as f:
            episodes_1 = json.load(f)
        with open(episode_file_2, "r") as f:
            episodes_2 = json.load(f)
        episodes = episodes_1 + episodes_2
        print(len(episodes_1), len(episodes_2), len(episodes))
        with open(merged_file, "w") as f:
            json.dump(episodes, f, indent=4)
            print(f"✅ {m} done")

# 修改函数
def correct_main():
    all_user_data_file = "/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0/results_unified_2025-12-22T15-55-18_False_False_t_0.0.json"
    # rollout_result_file = "/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0/rollout_v2/results_dissonance_t_1.0.json"
    # rollout_result_file = "/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0/rollout_v2/results_state_entropy_t_1.0.json"
    # rollout_result_file = "/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0/rollout_v2/results_action_entropy_t_1.0.json"
    # rollout_result_file = "/root/EvolvingAgent-master/EvolvingAgentTest_wym/model/Motivation/Experiments/exp_2025-12-22T15-55-18_Order_False_User_False_t_0.0/rollout_v2/results_random_t_1.0.json"

    with open(all_user_data_file, "r") as f:
        all_user_data = json.load(f)
    with open(rollout_result_file, "r") as f:
        episodes = json.load(f)
    new_episodes = add_success_state(episodes, all_user_data)

    with open(rollout_result_file, "w") as f:
        json.dump(new_episodes, f, indent=4)    

if __name__ == '__main__':
    merge_user_results()
    