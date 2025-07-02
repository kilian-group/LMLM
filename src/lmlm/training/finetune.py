# Copyright 2024 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0

import argparse
import json
import logging
from trl import (
    SFTTrainer,
    TrlParser,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    get_peft_config,
    get_kbit_device_map,
    get_quantization_config,
    DataCollatorForCompletionOnlyLM,
)

from lmlm.training.utils.load_model import load_llama3_for_instruction_tuning
from lmlm.training.utils.load_sft_dataset import prepare_instruction_tuning_data
from lmlm.training.utils.utils_metrics import (
    set_wandb,
    set_tokenizer,
    preprocess_logits_for_metrics,
    compute_metrics,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig)
    if subparsers is not None:
        return subparsers.add_parser("sft", help="Run SFT training", dataclass_types=dataclass_types)
    return TrlParser(dataclass_types)


def main(script_args, training_args, model_args):
    set_wandb()

    logger.info("Initializing model and tokenizer...")
    quant_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=not training_args.gradient_checkpointing,
        device_map=get_kbit_device_map() if quant_config else None,
        quantization_config=quant_config,
    )

    model, tokenizer = load_llama3_for_instruction_tuning(model_args, model_kwargs)
    set_tokenizer(tokenizer)

    logger.info("Loading training and evaluation datasets...")
    use_prompt = True
    train_dataset, eval_dataset = prepare_instruction_tuning_data(script_args, tokenizer, use_prompt=use_prompt)

    response_template = "<|start_header_id|>assistant<|end_header_id|>" if use_prompt else "### Output:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    logger.info(f"First training sample preview:\n{train_dataset[0]['formatted_text'][:200]}...")

    logger.info("Starting supervised fine-tuning...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=collator,
    )

    trainer.train()

    logger.info("Evaluating model on validation set...")
    eval_results = trainer.evaluate()
    logger.info("Evaluation results:\n" + json.dumps(eval_results, indent=4))

    logger.info(f"Saving model to: {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        logger.info("Pushing model to HuggingFace Hub...")
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
