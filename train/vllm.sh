vllm serve "/root/Q-Insight/src/open-r1-multimodal/output/Q-Insight-eval-score-rl" \
    --host "0.0.0.0" \
    --port 8000 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 16384 \
    --dtype bfloat16 \
    --trust-remote-code \
    --served-model-name "qwen-vl" \