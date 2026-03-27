def generate_counterfactual_states(original_state, history_summary, use_local_api=True):
    """
    使用 LLM 生成反事实的状态分析
    返回: {'negative': str, 'positive': str}
    """
    prompt = [
        {"role": "system", "content": "You are an expert data augmentor for dialogue systems."},
        {"role": "user", "content": f"""
        Original State Analysis: "{original_state}"
        Dialogue Context Summary: "{history_summary}"
        
        Please generate two counterfactual state analyses that are plausible but semantically opposite to the original one:
        1. [Negative]: The user is extremely resistant, angry, or refuses to communicate.
        2. [Positive]: The user is extremely compliant, happy, and ready to donate immediately.
        
        Output strictly in JSON format:
        {{
            "negative": "...",
            "positive": "..."
        }}
        """}
    ]
    
    # 这里可以使用你现有的 call_llm_chat_api_openai 或其他 API 调用函数
    # 为了演示，假设我们调用一个生成函数
    # 注意：这里需要确保返回的是 JSON 格式
    try:
        if use_local_api:
             # 使用本地模型生成，需要适配 prompt 格式
             # 这里简化处理，直接用规则构造假数据或者调用较强的模型（如 GPT-4）来生成 Attack
             # 建议反事实生成这一步使用 GPT-4，保证攻击质量
             pass 
        
        # 模拟 GPT-4 调用
        # result = call_gpt4(prompt) 
        # return json.loads(result)
        
        # 临时 Mock 数据 (实际运行时请替换为真实 LLM 调用)
        return {
            "negative": "The user appears extremely hostile and explicitly refuses to engage further, questioning the legitimacy of the charity.",
            "positive": "The user is very enthusiastic about the cause and seems ready to make a donation immediately without further persuasion."
        }
    except Exception as e:
        print(f"Error generating counterfactuals: {e}")
        return None