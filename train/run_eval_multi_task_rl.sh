set -x

export DEBUG_MODE="true"
RUN_NAME="eval_deficiency_ep2_f1_0.7_s_0.25"
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

# rm logs
rm -rf $LOG_PATH

uv run torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    src/open_r1/eval_multi_task.py \
    --output_dir output/${RUN_NAME} \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name None \
    --max_prompt_length 4096 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name ${RUN_NAME} \
    --save_steps 200 \
    --score_reward_threshold 0.25 \
    --beta 0.001 \
    --deepspeed local_scripts/zero2.json \
    --dataset_deficiency data_config/slide_deficiency.yaml \
    --dataset_score data_config/slide_score.yaml \
    --deficiency_f1_threshold 0.7 \


