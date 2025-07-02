#!/bin/bash

source ./scripts/account/openai_key.sh

SAVE_DIR=./output/annotation

ANNOTATOR=chatgpt
MODEL_ID=gpt-4o
PROMPT_ID=gpt4o-v7.1
CONFIG_FILE=gpt4o/default

FORMAT=json
SEED=42

DATASET=squad
MANAGER=${DATASET}-eval100
SUBSET=ids/${MANAGER}-ids

python -m lmlm.annotate.annotate \
    --annotator ${ANNOTATOR} \
    --model-id ${MODEL_ID} \
    --prompt-id ${PROMPT_ID} \
    --manager ${MANAGER} \
    --config-file ${CONFIG_FILE} \
    --dataset ${DATASET} \
    --format ${FORMAT} \
    --seed ${SEED} \
    --save-every 500 \
    --save-dir ${SAVE_DIR}

