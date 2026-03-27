# uvicorn utils.local_model_client:app --host 0.0.0.0 --port 8093


# 如何设置当前目录为
cd /root/EvolvingAgent-master/EvolvingAgentTest_wym/utils

# LOCAL_MODEL="/data/pretrained_models/Qwen2.5-7B-Instruct"
LOCAL_MODEL="/data/pretrained_models/Qwen3-8B"

# # 使用默认设置
# python local_model_client.py

# # 指定端口
# python local_model_client.py --port 8080

# # 指定模型路径
# python local_model_client.py --model-path /path/to/your/model

# # 指定GPU
# python local_model_client.py --gpus 0,1

# 组合使用所有参数
python local_model_client.py --port 8092 --model-path $LOCAL_MODEL --gpus 4,