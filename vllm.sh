uv venv vllm_env
source vllm_env/bin/activate
uv pip install vllm
MODEL_PATH="/Q-Insight/src/open-r1-multimodal/output/eval_deficiency_f1_0.6_ep_1_new"

uv run python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host "0.0.0.0" \
  --port 8000 \
  --max-model-len 4096 \
  --enforce-eager \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 8