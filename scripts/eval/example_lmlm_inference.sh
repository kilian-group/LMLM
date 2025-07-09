#!/bin/bash

models=(
    "kilian-group/LMLM-llama2-382M"
    "kilian-group/Standard-llama2-382M"
)

DATABASE_PATH=./data/database/dwiki_bio17k-annotator_database.json  # Will load cached index if available, otherwise download from Hugging Face, or build and cache it locally
OUTPUT_DIR=./output/eval/examples/

for model in "${models[@]}"; do


    echo "Running model=$model"
    python ./experiment/eval/example_lmlm_inference.py \
        --model_name "$model" \
        --model_path "$model" \
        --database_path "$DATABASE_PATH" \
        --output_dir "$OUTPUT_DIR"
done
