MODEL_PATH="TobyYang7/eval_deficiency_ckpt1000_0827"

uv run python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host "0.0.0.0" \
  --port 8000 \
  --max-model-len 4096 \
  --enforce-eager \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 8