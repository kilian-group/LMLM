#!/bin/bash

export WANDB_PROJECT=mem_pretrain
source ./scripts/account/wandb_config.sh

# Unique port per job
export MASTER_PORT=$((29501 + RANDOM % 1000))

export NCCL_TIMEOUT=18000
export NCCL_ASYNC_ERROR_HANDLING=1

# Ensure GPU isolation
export CUDA_VISIBLE_DEVICES=0,1

# Prevent NCCL conflicts
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export TORCH_USE_CUDA_DSA=1

# Set arguments
NUM_GPUs=2

ADD_DBLOOKUP_TOKENS=True
MODEL_NAME_OR_PATH="tiny-llama2-176M"
NUM_TRAIN_EPOCHS=8

DATASET_PATH=kilian-group/LMLM-dwiki6.1M

OUTPUT_DIR=./checkpoints/pretrain_test/
#

if [ "$NUM_GPUs" -eq 2 ]; then
    PER_DEVICE_TRAIN_BATCH_SIZE=128
    GRADIENT_ACCUMULATION_STEPS=1
elif [ "$NUM_GPUs" -eq 1 ]; then
    PER_DEVICE_TRAIN_BATCH_SIZE=128
    GRADIENT_ACCUMULATION_STEPS=2
elif [ "$NUM_GPUs" -eq 4 ]; then
    PER_DEVICE_TRAIN_BATCH_SIZE=128
    GRADIENT_ACCUMULATION_STEPS=1
fi


EFFECTIVE_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUs))

OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME_OR_PATH}_${DATASET_PATH##*/}_ep${NUM_TRAIN_EPOCHS}_bsz${EFFECTIVE_BATCH_SIZE}"
if [ "$ADD_DBLOOKUP_TOKENS" = "True" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_new"
fi


accelerate launch --config_file ./configs/accelerate/multi_gpu.yaml --main_process_port $MASTER_PORT --num_processes $NUM_GPUs \
    -m lmlm.training.pretrain \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_name ${DATASET_PATH} \
    --packing \
    --learning_rate 5.0e-4 \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --logging_steps 50 \
    --eval_strategy steps \
    --gradient_checkpointing \
    --eval_steps 1000 \
    --save_steps 5000 \
    --save_total_limit 30 \
    --dataset_text_field annotated_text \
    --output_dir ${OUTPUT_DIR} \
    --use_special_dblookup_tokens ${ADD_DBLOOKUP_TOKENS} \
    --plain_baseline False \
    --lr_scheduler_type cosine \
    --warmup_steps 2000 \