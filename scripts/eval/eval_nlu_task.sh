#!/bin/bash

OUTPUT_DIR="./output/eval/nlu_tasks"

CHECKPOINTS=(
    "kilian-group/LMLM-llama2-382M"
    "kilian-group/LMLM-llama2-176M"
    "kilian-group/Standard-llama2-382M"
    "kilian-group/Standard-llama2-176M"
)

CUSTOM_TASKS="./experiment/eval/lighteval_tasks.py"
MAX_SAMPLES=100000

# Define tasks
TASKS=(
    "custom|hellaswag|0|1"
    "custom|winogrande|0|1"
    "custom|piqa|0|1"
    "custom|siqa|0|1"
    "custom|openbookqa|0|1"
    "custom|arc:easy|0|1"
    "custom|arc:challenge|0|1"
    "custom|commonsense_qa|0|1"
)

# Join tasks into a comma-separated string
TASKS_STR=$(IFS=,; echo "${TASKS[*]}")

# Loop through each checkpoint and run evaluation
for ckpt in "${CHECKPOINTS[@]}"; do
    echo "Evaluating checkpoint: ${ckpt}"
    lighteval accelerate \
        --model_args="pretrained=${ckpt}" \
        --custom_tasks "${CUSTOM_TASKS}" \
        --output_dir "${OUTPUT_DIR}" \
        --max_samples "${MAX_SAMPLES}" \
        --tasks "${TASKS_STR}"
done
