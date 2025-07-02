#!/bin/bash

export WANDB_PROJECT=mem_tune
source ./scripts/account/wandb_config.sh

# Set arguments
NUM_TRAIN_EPOCHS=10
OUTPUT_DIR="./output/tune_annotator_test"
DATASET=squad-train1k_dwiki-train1k_chatgpt_gpt4o-v7.1
#

MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B-Instruct"

DATASET_PATH=./data/ft/${DATASET}.json
OUTPUT_DIR="${OUTPUT_DIR}/${MODEL_NAME_OR_PATH##*/}_${DATASET}_ep${NUM_TRAIN_EPOCHS}"

python -m lmlm.training.finetune \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_name ${DATASET_PATH} \
    --learning_rate 2.0e-4 \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --save_total_limit 2 \
    --eval_strategy steps \
    --per_device_eval_batch_size 4 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --dataset_text_field formatted_text \
    --output_dir ${OUTPUT_DIR} \
    --max_seq_length 2048 \
    --logging_steps 1 \
    --save_steps 100 \
    --eval_steps 10 \
    --eval_accumulation_steps 1 \