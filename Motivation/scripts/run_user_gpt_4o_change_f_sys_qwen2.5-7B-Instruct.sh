cd /root/EvolvingAgent-master/EvolvingAgentTest_wym

python model/Motivation/test_start_step4.py \
    --use_local_api False \
    --order_change False \
    --sys_llm_model "qwen2.5-7B-Instruct" \
    --sys_local_api True \
    --temperature 0.0