uv venv vllm_env
source vllm_env/bin/activate
uv pip install vllm
uv pip install qwen-vl-utils
MODEL_PATH="/root/Q-Insight/train/output/eval_multi_ep1_compare"

uv run python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host "0.0.0.0" \
  --port 8000 \
  --max-model-len 4096 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 8 \
  # --enforce-eager \
  # --tensor-parallel-size 8 \