# Copyright 2024 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from accelerate import Accelerator
from trl import (
    SFTTrainer,
    TrlParser,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    get_peft_config,
)

from lmlm.training.utils.utils_metrics import (
    compute_loss_func,
    set_wandb,
    set_tokenizer,
    compute_metrics,
    set_use_special_dblookup_tokens,
)
from lmlm.training.utils.load_model import initialize_model_for_pretraining
from lmlm.training.utils.load_sft_dataset import prepare_pretrain_data

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class PretrainConfig:
    use_special_dblookup_tokens: bool = False
    plain_baseline: bool = False
    eval_only: bool = False


def set_random_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, PretrainConfig)
    if subparsers is not None:
        return subparsers.add_parser("pretrain", help="Run LMLM pretraining", dataclass_types=dataclass_types)
    return TrlParser(dataclass_types)


def main(script_args, training_args, model_args, pretrain_args):
    accelerator = Accelerator()
    set_random_seed(getattr(training_args, "seed", 42))

    if accelerator.is_main_process:
        set_wandb()

    if pretrain_args.plain_baseline and pretrain_args.use_special_dblookup_tokens:
        raise ValueError("Cannot enable both `plain_baseline` and `use_special_dblookup_tokens`.")

    if accelerator.is_main_process:
        logger.info(f"use_special_dblookup_tokens = {pretrain_args.use_special_dblookup_tokens}")

    if training_args.resume_from_checkpoint:
        model, tokenizer = initialize_model_for_pretraining(
            model_args,
            resume_from_checkpoint=training_args.resume_from_checkpoint,
            use_special_dblookup_tokens=pretrain_args.use_special_dblookup_tokens,
        )
    else:
        model, tokenizer = initialize_model_for_pretraining(
            model_args,
            use_special_dblookup_tokens=pretrain_args.use_special_dblookup_tokens,
        )

    set_use_special_dblookup_tokens(pretrain_args.use_special_dblookup_tokens)
    set_tokenizer(tokenizer)

    training_args.remove_unused_columns = False
    training_args.max_seq_length = 1024
    training_args.compute_loss_func = compute_loss_func

    logger.info("Preparing training and evaluation datasets...")
    train_dataset, eval_dataset = prepare_pretrain_data(
        script_args,
        use_special_dblookup_tokens=pretrain_args.use_special_dblookup_tokens,
        is_plain_baseline=pretrain_args.plain_baseline,
    )

    if accelerator.is_main_process:
        logger.info(f"Example training sample: {train_dataset[10][training_args.dataset_text_field]}")
        logger.info(f"Training set size: {len(train_dataset)}")
        logger.info(f"Evaluation set size: {eval_dataset}")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        compute_metrics=compute_metrics,
        compute_loss_func=compute_loss_func,
    )

    if not pretrain_args.eval_only:
        if training_args.resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        else:
            trainer.train()

    eval_results = trainer.evaluate()
    logger.info("Evaluation results:\n" + json.dumps(eval_results, indent=4))

    logger.info(f"Saving model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        logger.info("Pushing model to HuggingFace Hub...")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, pretrain_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args, pretrain_args)
