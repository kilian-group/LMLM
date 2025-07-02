# Copyright 2024 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0

import os
import json
import math
import logging
from datetime import datetime
from dataclasses import dataclass

import argparse
import wandb
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
    ScriptArguments,
)

from lmlm.constants import DATA_DIR
from lmlm.training.utils.load_model import load_lora_model, load_tiny_llama2_tokenizer
from lmlm.training.utils.load_sft_dataset import prepare_pretrain_data
from lmlm.training.utils.utils_metrics import (
    set_tokenizer,
    dataset_stats,
    set_wandb,
    set_dataset_name,
    set_is_cleaned_dataset,
    convert_th_config_to_name,
    compute_metrics,
    preprocess_logits_for_metrics,
)
from lmlm.training.utils.utils_filter import (
    clean_high_loss_triplets,
    filter_length,
    convert_to_raw_dataset,
    convert_to_special_db_tokens_format,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class EvaluationConfig:
    eval_dataset_strategy: str = "no"  # options: "no", "perplexity", "stats", "both"
    save_dir: str = None
    clean_dataset: bool = False
    enable_length_filter: bool = False
    use_llama2_tokenizer: bool = False
    add_special_tokens: bool = False
    eval_raw_dataset: bool = False


def evaluate_in_batches(model, tokenizer, full_dataset, batch_size=20, trainer_args=None):
    total_samples = 0
    weighted_metrics = {}
    num_samples = len(full_dataset)
    num_batches = math.ceil(num_samples / batch_size)

    for i in range(num_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, num_samples)
        batch_subset = full_dataset.select(range(start, end))

        trainer = SFTTrainer(
            model=model,
            args=trainer_args,
            eval_dataset=batch_subset,
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args),
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        results = trainer.evaluate()

        if i == 0:
            weighted_metrics = {k: 0.0 for k in results}
        for k in results:
            weighted_metrics[k] += results[k] * (end - start)

        total_samples += end - start

    for k in weighted_metrics:
        weighted_metrics[k] /= total_samples

    return weighted_metrics


def prepare_triplets_save_path(eval_args, dataset_name):
    save_dir = eval_args.save_dir or "./output/dataset/high_loss_triplets"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{dataset_name}_{convert_th_config_to_name()}.json"
    return os.path.join(save_dir, filename)


def save_cleaned_dataset(dataset, dataset_name, model_name):
    save_path = os.path.join(DATA_DIR, "cleaned", f"{dataset_name}_cleaned.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metadata = {
        "source_datasets": [dataset_name],
        "corrector_model": [model_name],
        "correction_th": convert_th_config_to_name(),
        "last_modified": datetime.now().isoformat()
    }
    with open(save_path, "w") as f:
        json.dump({"examples": dataset.to_list(), "metadata": metadata}, f, indent=4)
    logger.info(f"Saved cleaned dataset to {save_path}")


def main(script_args, training_args, model_args, eval_args):
    set_wandb()
    tokenizer_only = eval_args.eval_dataset_strategy in ["stats", "no"]
    model, tokenizer = load_lora_model(model_args, tokenizer_only=tokenizer_only)

    if eval_args.use_llama2_tokenizer:
        tokenizer = load_tiny_llama2_tokenizer(add_special_tokens=eval_args.add_special_tokens)
        logger.info("Loaded TinyLLaMA2 tokenizer with special tokens: %s", eval_args.add_special_tokens)

    train_dataset, _ = prepare_pretrain_data(
        script_args,
        use_special_dblookup_tokens=eval_args.add_special_tokens,
        is_plain_baseline=eval_args.eval_raw_dataset,
    )
    dataset_name = script_args.dataset_name.split("/")[-1].split(".json")[0]

    if eval_args.eval_raw_dataset:
        train_dataset = convert_to_raw_dataset(train_dataset)
        dataset_name += "_raw"

    if eval_args.add_special_tokens:
        train_dataset = convert_to_special_db_tokens_format(train_dataset)
        dataset_name += "_special_db"

    if eval_args.eval_dataset_strategy in ["stats", "both"]:
        dataset_stats(dataset_name, train_dataset, tokenizer, visualize=True)

    if eval_args.enable_length_filter:
        train_dataset = train_dataset.filter(filter_length)
        logger.info("Filtered dataset by length: %d samples remaining", len(train_dataset))

    subset = (
        train_dataset.shuffle(seed=42)
        if eval_args.clean_dataset
        else train_dataset.shuffle(seed=42).select(range(min(100, len(train_dataset))))
    )

    if eval_args.eval_dataset_strategy in ["perplexity", "both"]:
        set_tokenizer(tokenizer)
        set_dataset_name(dataset_name)
        set_is_cleaned_dataset(eval_args.clean_dataset)

        eval_results = evaluate_in_batches(
            model=model,
            tokenizer=tokenizer,
            full_dataset=subset,
            batch_size=50,
            trainer_args=training_args,
        )

        logger.info("Evaluation Results:\n%s", json.dumps(eval_results, indent=4))
        if wandb.run:
            wandb.log(eval_results)

    if eval_args.clean_dataset:
        cleaned = clean_high_loss_triplets(subset, triplets_save_path=prepare_triplets_save_path(eval_args, dataset_name))
        save_cleaned_dataset(cleaned, dataset_name, model_args.model_name_or_path)

        if eval_args.eval_dataset_strategy in ["stats", "both"]:
            dataset_stats(f"{dataset_name}_cleaned", cleaned, tokenizer, visualize=False)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, EvaluationConfig)
    if subparsers:
        return subparsers.add_parser("eval", help="Run LMLM evaluation", dataclass_types=dataclass_types)
    return TrlParser(dataclass_types)


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, eval_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args, eval_args)
