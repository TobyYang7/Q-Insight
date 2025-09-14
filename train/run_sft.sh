set -x

# Script arguments
RUN_NAME="qwen-vl-sft-multi-task"
OUTPUT_DIR="output/${RUN_NAME}"

# --- Dataset Configs ---
# Comment out or set to "" any dataset you want to exclude
DATASET_DEFICIENCY="train/data_config/slideaudit.yaml"
DATASET_SCORE="train/data_config/slide_quality.yaml"
DATASET_COMPARISON="train/data_config/slide_compare.yaml"

# Dist args
NPROC_PER_NODE=${ARNOLD_WORKER_GPU:-8}
NNODES=${ARNOLD_WORKER_NUM:-1}
NODE_RANK=${ARNOLD_ID:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-12346} # Use a different port from the RL script

# Envs
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

# Clean output dir
rm -rf ${OUTPUT_DIR}

# Build dataset arguments
# Only add dataset args to the command if the path is set
DATASET_ARGS=""
if [ -n "$DATASET_DEFICIENCY" ]; then
    DATASET_ARGS="$DATASET_ARGS --dataset_deficiency $DATASET_DEFICIENCY"
fi
if [ -n "$DATASET_SCORE" ]; then
    DATASET_ARGS="$DATASET_ARGS --dataset_score $DATASET_SCORE"
fi
if [ -n "$DATASET_COMPARISON" ]; then
    DATASET_ARGS="$DATASET_ARGS --dataset_comparison $DATASET_COMPARISON"
fi

uv run torchrun --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/open_r1/train_sft.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path Qwen/Qwen2.5-VL-7B-Instruct \
    $DATASET_ARGS \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --report_to wandb \
    --run_name ${RUN_NAME} \
    --save_steps 200 \
    --deepspeed local_scripts/zero2.json
