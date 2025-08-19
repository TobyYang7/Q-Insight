set -x

export DEBUG_MODE="true"
RUN_NAME="Q-Insight-eval-score-rl"
export LOG_PATH="./debug_log_${RUN_NAME}.txt"

# Dist args (single node by default)
nproc_per_node=${ARNOLD_WORKER_GPU:-8}
nnodes=${ARNOLD_WORKER_NUM:-1}
node_rank=${ARNOLD_ID:-0}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-12345}

echo "[nproc_per_node: ${nproc_per_node}]"
echo "[nnodes: ${nnodes}]"
echo "[node_rank: ${node_rank}]"
echo "[master_addr: ${master_addr}]"
echo "[master_port: ${master_port}]"

# Envs
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

uv run torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    src/open_r1/eval_score_rl.py \
    --output_dir output/${RUN_NAME} \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name None \
    --dataset_score data_config/iqa_score_custom.yaml \
    --score_prompt_file /root/Q-Insight/src/open-r1-multimodal/prompts/overall_1to5.txt \
    --image_root /root/Q-Insight/paper-slide-crawler \
    --max_prompt_length 4096 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 3 \
    --run_name ${RUN_NAME} \
    --save_steps 500 \
    --save_only_model true \
    --score_reward_threshold 0.4 \
    --beta 0.001 \
    --deepspeed local_scripts/zero2.json \


