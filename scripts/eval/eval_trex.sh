#!/bin/bash

CHECKPOINTS=(
    "kilian-group/LMLM-llama2-382M"
    "kilian-group/Standard-llama2-382M"
    "kilian-group/LMLM-llama2-176M"
    "kilian-group/Standard-llama2-176M"
)

OUTPUT_DIR=./output/eval/eval_trex/
DATABASE_PATH=./data/database/trex11k-annotator_database.json
DATASET=./data/trex11k.json
NUM_SAMPLES=100000

THRESHOLD=0.6


DBLOOKUP_OPTIONS=("--enable_dblookup")

# Loop through all combinations
for ENABLE_DBLOOKUP in "${DBLOOKUP_OPTIONS[@]}"; do
        for MODEL_PATH in "${CHECKPOINTS[@]}"
        do
            echo "Running ${MODEL_PATH} with ${ENABLE_DBLOOKUP}"

            python ./experiment/eval/eval_trex.py \
                --model_name_or_path "${MODEL_PATH}" \
                --dataset_name "${DATASET}" \
                --per_device_eval_batch_size 1 \
                --max_seq_length 1024 \
                --output_dir "${OUTPUT_DIR}" \
                --database_path "${DATABASE_PATH}" \
                --threshold "${THRESHOLD}" \
                ${ENABLE_DBLOOKUP} \
                --num_samples ${NUM_SAMPLES} \
                --top_k 0
        done
    done
done



