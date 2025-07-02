#!/bin/bash

# Extract knowledge triplets from the annotation.

ANNOTATION_PATH=./output/annotation/squad-eval100_llama-LMLM-Annotator_llama-v6.1.json
SAVE_PATH=./output/database/squad-eval100-Annotator_database.json

python -m lmlm.database.extract_database \
    --annotation_path $ANNOTATION_PATH \
    --save_path $SAVE_PATH \