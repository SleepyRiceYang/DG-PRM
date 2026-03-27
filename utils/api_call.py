import requests
import json
import time
import copy
from typing import List, Dict, Optional, Any
from openai import OpenAI


# --- 配置服务地址 (保持不变) ---
SERVER_IP = "127.0.0.1"
PORT = 8991  # 假设你用 8888 端口启动
API_URL = f"http://{SERVER_IP}:{PORT}/generate"

GPT_URL = "https://yunwu.ai/v1"
GPT_API_KEY = "sk-mBIpuzxBGQ4B4e2ropbDvjYNOxI84borqWuHTtYM1tNAheq4"

try:
    openai_client = OpenAI(api_key=GPT_API_KEY, base_url=GPT_URL)
    print(f"OpenAI 客户端初始化成功，基础 URL: {GPT_URL}")
except Exception as e:
    print(f"OpenAI 客户端初始化失败: {e}")
    openai_client = None # 标记为 None，后续调用会失败

def call_llm_chat_api(messages: list, temperature: float = 0.7, sample: int =1, model: str ="Qwen3-8B_11-12-15-00"):
    """
    发送一个包含 messages 列表的 POST 请求。
    """
    headers = {"Content-Type": "application/json"}
    payload = {"messages": messages, "model": "qwen-3.8b", "sample": 1}  # 假设模型名称

    try:
        start_time = time.time()
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        end_time = time.time()
        
        response.raise_for_status()
        
        result = response.json()
        print(f"--- 请求成功 ---")
        # 打印最后一条用户消息和模型的回复
        print(f"用户: {messages[-1]['content']}")
        print(f"模型: {result.get('generated_text')}")
        print(f"耗时: {end_time - start_time:.2f} 秒")
        print("--------------------")
        
        return result, result.get("generated_text")
        
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return None

import time
import requests
import json
from typing import List, Dict, Any

# 假设 openai_client 在外部已初始化，或者你可以在这里进行初始化检查
# openai_client = OpenAI(...) 

def call_llm_chat_api_openai(
    model_name: str,
    messages: List[Dict[str, str]], 
    temperature: float = 0.7,
    n: int = 1,
    max_tokens: int = 1024,
    # --- 新增参数 ---
    use_local_api: bool = False,
    local_api_url: str = "http://localhost:8092/inference",
    logprobs: bool = False, # 控制是否计算 logprobs (默认 False，即不返回 logits 相关计算)
    **kwargs: Any 
) -> List[str]:
    """
    统一调用接口：支持 OpenAI 官方 SDK 和 自定义本地 HTTP API。
    """
    start_time = time.time()

    # ==================== 分支 1: 调用本地 API ====================
    if use_local_api:
        headers = {"Content-Type": "application/json"}
        
        # 构建本地 API 需要的 payload
        payload = {
            "task_type": "generate",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": n,
            "logprobs": logprobs,
            "top_logprobs": 5 if logprobs else None # 如果需要 logprobs，默认取 top 5
        }
        for k, v in kwargs.items():
            if k not in payload:
                payload[k] = v

        try:
            # 发送请求
            responses_results = []
            for i in range(n):
                print(f"第 {i+1} 次请求:")
                current_payload = copy.deepcopy(payload)
                current_payload['n'] = 1 
                # print(f"发送的 payload: {current_payload}")
                response = requests.post(local_api_url, json=current_payload, headers=headers)
                response.raise_for_status() # 检查 HTTP 错误
                
                result = response.json()
                duration = time.time() - start_time
            
                # 提取生成的文本
                responses = result.get("generated_texts", [])
                if not responses:
                    print(f"⚠️ 第 {i+1} 次请求成功但未返回文本")
                    continue
                responses_results.extend(responses)
                print(f"✅ 第 {i+1} 次请求成功，耗时: {duration:.2f}秒")
            return responses_results

        except Exception as e:
            raise ValueError(f"本地 API 调用失败: {e}")

    # ==================== 分支 2: 调用 OpenAI SDK ====================
    else:
        if 'openai_client' not in globals() or openai_client is None:
            # 这里假设 openai_client 是全局变量，或者你可以改成传入 client
            raise ValueError("OpenAI 客户端未成功初始化，无法进行 SDK 调用。")
            
        try:
            # 处理 logprobs 参数适配 OpenAI 的格式
            # OpenAI 的 logprobs 是布尔值，top_logprobs 是整数
            api_kwargs = kwargs.copy()
            if logprobs:
                api_kwargs['logprobs'] = True
                api_kwargs['top_logprobs'] = 5
            
            completion = openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                n=n,
                max_tokens=max_tokens,
                **api_kwargs # 传递处理后的额外参数
            )
            
            duration = time.time() - start_time
            responses = []
            for choice in completion.choices:
                if choice.message and choice.message.content:
                    responses.append(choice.message.content.strip())
            
            if not responses:
                raise ValueError("API 调用成功但未收到任何有效的回复内容。")
            
            if len(responses) < n / 2:
                print(f"警告: 仅收到 {len(responses)} 个回复，少于请求的 {n} 个。")
                raise ValueError("请求少于要求的采样数量")
                
            print(f"OpenAI SDK 请求成功，耗时: {duration:.2f}秒")
            return responses
            
        except Exception as e:
            raise ValueError(f"OpenAI API 调用失败: {e}")



# --- 主程序：演示如何进行多轮对话 ---
if __name__ == "__main__":
    print("开始调用 LLM 聊天服务...")
    
    # 模拟一次多轮对话
    
    # 第一轮
    conversation_history = [
        {"role": "user", "content": "你好，请你用中文介绍一下什么是大型语言模型？"}
    ]
    api_response = call_llm_chat_api(conversation_history)
    
    # 如果调用成功，将模型的回复加入历史记录
    if api_response:
        conversation_history.append(
            {"role": "assistant", "content": api_response.get("generated_text")}
        )

        # 第二轮：基于上一轮的上下文提问
        conversation_history.append(
            {"role": "user", "content": "听起来很强大！那它和传统的机器学习模型有什么本质区别呢？"}
        )
        api_response_2 = call_llm_chat_api(conversation_history)
        
        # 继续对话...