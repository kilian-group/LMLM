#!/bin/bash

SAVE_DIR=./output/annotation
ANNOTATOR=llama
MODEL_ID=kilian-group/LMLM-Annotator

PROMPT_ID=llama-v6.1
FORMAT=json
SEED=42

DATASET=conflictsbench
MANAGER=${DATASET}-train1000
SUBSET=ids/${MANAGER}-ids

CONFIG_FILE=llama/default
    
echo "Processing MODEL_ID: $MODEL_ID"

python -m lmlm.annotate.annotate \
    --annotator ${ANNOTATOR} \
    --model-id ${MODEL_ID} \
    --prompt-id ${PROMPT_ID} \
    --manager ${MANAGER} \
    --config-file ${CONFIG_FILE} \
    --dataset ${DATASET} \
    --format ${FORMAT} \
    --seed ${SEED} \
    --save-every 100 \
    --save-dir ${SAVE_DIR} \
    # --subset ${SUBSET}
