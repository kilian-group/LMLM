#!/bin/bash

export PYTHONPATH=../FActScore
source ./scripts/account/openai_key.sh

CHECKPOINTS=(
    "kilian-group/LMLM-llama2-382M"
    # "kilian-group/LMLM-llama2-176M"
    # "kilian-group/Standard-llama2-382M"
    # "kilian-group/Standard-llama2-176M"
)

## Change this to "" to disable dblookup
# ENABLE_DBLOOKUP=""
ENABLE_DBLOOKUP="--enable_dblookup"

OUTPUT_DIR=./output/eval/eval_factscore/
DATABASE_PATH=./data/database/dwiki_bio17k-annotator_database.json  # Will load cached index if available, otherwise download from Hugging Face, or build and cache it locally
DATASET=factscore
NUM_SAMPLES=100

TEMPERATURE=0.0 # greedy decoding
TOP_P=0.9
SEED=42
MAX_NEW_TOKENS=256
THRESHOLD=0.6
REPITITION_PENALTY=1.2


for MODEL in "${CHECKPOINTS[@]}"
do
    echo "Running ${MODEL} with ${ENABLE_DBLOOKUP}"

    # in the LMLM directory``
    python ./experiment/eval/eval_factscore.py \
        --save-dir $OUTPUT_DIR \
        --model $MODEL \
        --dataset $DATASET \
        --num-samples $NUM_SAMPLES \
        --temperature $TEMPERATURE \
        --top-p $TOP_P \
        --seed $SEED \
        --max-new-tokens $MAX_NEW_TOKENS \
        --world-size 1 \
        --database-path $DATABASE_PATH \
        --entity-path "./experiment/eval/factscore_labeled_prompt_entities.txt" \
        --threshold $THRESHOLD \
        --repetition-penalty $REPITITION_PENALTY \
        ${ENABLE_DBLOOKUP} \
    

    FORMATTED_TEMP=$(printf "%.1f" $TEMPERATURE)
    ARGS_POSTFIX="t${FORMATTED_TEMP}_p${TOP_P}_s${SEED}_rep${REPITITION_PENALTY}_th${THRESHOLD}_len${MAX_NEW_TOKENS}"

    MODEL_BASENAME=$(basename "$MODEL")

    if [[ "$MODEL_BASENAME" == *"checkpoint"* ]]; then
        # When "checkpoint" is in the last part of the path
        PARENT_DIR=$(basename "$(dirname "$MODEL")")
        CHECKPOINT_NUM=$(echo "$MODEL_BASENAME" | awk -F'-' '{print $NF}')
        MODEL_NAME="${PARENT_DIR}_ckpt${CHECKPOINT_NUM}"
    else
        # Just use the basename if no checkpoint in the name
        MODEL_NAME="$MODEL_BASENAME"
    fi

    if [ -z "${ENABLE_DBLOOKUP}" ]; then
        INPUT_PATH="${OUTPUT_DIR}/${MODEL_NAME}_${ARGS_POSTFIX}.jsonl"
    else
        INPUT_PATH="${OUTPUT_DIR}/${MODEL_NAME}_dblookup_${ARGS_POSTFIX}.jsonl"
    fi

    if [ ! -f "${INPUT_PATH}" ]; then
        echo "Error: Input file ${INPUT_PATH} does not exist."
    fi
    
    python ./experiment/eval/factscorer.py \
    --input_path ${INPUT_PATH} \
    --model_name retrieval+ChatGPT \
    --cost_estimate consider_cache \
    --verbose \
    --n_samples ${NUM_SAMPLES}
done