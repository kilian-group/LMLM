import os
import json
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    GPT2TokenizerFast,
)
from peft import PeftModel
from lmlm.constants import CONFIGS_DIR
from lmlm.constants import DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, DB_END_TOKEN


def initialize_model_for_pretraining(model_args, resume_from_checkpoint=None, use_special_dblookup_tokens=False):
    """
    Load a pretrained model and tokenizer based on the provided model arguments.
    """
    model_name_or_path = model_args.model_name_or_path

    if resume_from_checkpoint:
        return load_model_from_checkpoint(resume_from_checkpoint, model_args)

    if "gpt2" in model_name_or_path:
        return load_gpt2_model(model_name_or_path, use_special_dblookup_tokens)
    elif "tiny-llama2" in model_name_or_path:
        return load_tiny_llama2_model(model_name_or_path, model_args, use_special_dblookup_tokens)
    else:
        return load_custom_model(model_name_or_path, model_args)


def load_model_for_ft_baseline(model_args, resume_from_checkpoint=None, use_special_dblookup_tokens=False):
    """
    Load a LLaMA3 model and tokenizer. Ensures the pad token is set and embeddings are resized if needed.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        resume_from_checkpoint,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )

    if use_special_dblookup_tokens:
        tokenizer, _ = add_dblookup_special_tokens(tokenizer)
        print(f"vocab_size: {len(tokenizer)}")

    model = AutoModelForCausalLM.from_pretrained(
        resume_from_checkpoint,
        trust_remote_code=model_args.trust_remote_code,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        print(f"Model vocab size updated: {model.config.vocab_size}, Tokenizer vocab size: {len(tokenizer)}")

    print_model_info(model_args.model_name_or_path, model)
    return model, tokenizer


def load_model_from_checkpoint(resume_from_checkpoint, model_args):
    """
    Load model and tokenizer from a checkpoint.
    """
    print(f"Loading model and tokenizer from checkpoint: {resume_from_checkpoint}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            resume_from_checkpoint,
            trust_remote_code=model_args.trust_remote_code,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            resume_from_checkpoint,
            trust_remote_code=model_args.trust_remote_code,
        )
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    except Exception as e:
        raise ValueError(f"Failed to load from checkpoint {resume_from_checkpoint}: {e}")


def load_gpt2_model(model_name_or_path, use_special_dblookup_tokens=False):
    """
    Load GPT-2 model and tokenizer. Optionally add special dblookup tokens.
    """
    config = GPT2Config.from_pretrained(model_name_or_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    if use_special_dblookup_tokens:
        tokenizer, config = add_dblookup_special_tokens(tokenizer, config)

    model = GPT2LMHeadModel(config)

    print_model_info(model_name_or_path, model)
    return model, tokenizer


def add_dblookup_special_tokens(tokenizer, config=None):
    """
    Add special dblookup tokens to tokenizer and optionally update config.
    """
    db_tokens = {
        "entity": DB_START_TOKEN,
        "relationship": DB_SEP_TOKEN,
        "return": DB_RETRIEVE_TOKEN,
        "end": DB_END_TOKEN,
    }

    new_tokens = list(db_tokens.values())
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    print(f"Added {num_added} tokens to the vocabulary")

    if config:
        config.vocab_size += num_added
        print(f"Updated vocab_size to {config.vocab_size}")

    return tokenizer, config


def load_tiny_llama2_tokenizer(add_special_tokens=False):
    """
    Load tokenizer for Tiny LLaMA2, optionally with special tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer/tiny-llama2")
    tokenizer.pad_token = tokenizer.eos_token

    if add_special_tokens:
        tokenizer, _ = add_dblookup_special_tokens(tokenizer)
    return tokenizer


def load_tiny_llama2_model(model_name_or_path, model_args, use_special_dblookup_tokens=False):
    """
    Load a Tiny LLaMA2 model from CONFIGS_DIR.
    """
    model_path = os.path.join(CONFIGS_DIR, f"tiny-llama/{model_name_or_path}")
    if not os.path.exists(model_path):
        raise ValueError(f"Model {model_name_or_path} not found in {CONFIGS_DIR}")

    tokenizer = AutoTokenizer.from_pretrained("./tokenizer/tiny-llama2")
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_path)

    if use_special_dblookup_tokens:
        tokenizer, config = add_dblookup_special_tokens(tokenizer, config)

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)

    print_model_info(model_name_or_path, model)
    return model, tokenizer


def load_custom_model(model_name_or_path, model_args):
    """
    Fallback for unsupported models. Currently raises NotImplementedError.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            use_fast=True,
        )
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer for {model_name_or_path}: {e}")

    raise NotImplementedError(f"Model {model_name_or_path} initialization not implemented.")


def load_lora_model(model_args, tokenizer_only=False):
    """
    Load a LoRA-wrapped model. If tokenizer_only is True, only load the tokenizer.
    """
    model_name_or_path = model_args.model_name_or_path

    if "ft" in model_name_or_path or "tune" in model_name_or_path:
        config_file = "llama-8b-ft/lora-ft-hf" if "8b" in model_name_or_path.lower() else None
        if not config_file:
            raise ValueError(f"Model {model_name_or_path} not found in {CONFIGS_DIR}")

        with open(os.path.join(CONFIGS_DIR, f"{config_file}.json"), "r") as f:
            configs = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer_only:
            return None, tokenizer

        base_model = AutoModelForCausalLM.from_pretrained(configs['base_model'], device_map="auto")

        if len(tokenizer) != base_model.config.vocab_size:
            if tokenizer.pad_token_id is None:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            base_model.config.pad_token_id = tokenizer.pad_token_id
            base_model.resize_token_embeddings(len(tokenizer))

            print(f"Added pad token and resized embeddings to {len(tokenizer)}")

        model = PeftModel.from_pretrained(base_model, model_name_or_path)
        print_model_info(model_name_or_path, model)
        return model, tokenizer

    raise ValueError(f"Model {model_name_or_path} not implemented for LoRA")


def load_llama3_for_instruction_tuning(model_args, model_kwargs):
    """
    Load LLaMA3 model and tokenizer for instruction tuning.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        print(f"EOS Token ID: {tokenizer.eos_token_id}")
        print(f"PAD Token ID: {tokenizer.pad_token_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
        print(f"Model vocab size updated: {model.config.vocab_size}, Tokenizer vocab size: {len(tokenizer)}")

    print_model_info(model_args.model_name_or_path, model)
    return model, tokenizer


def load_llama3_for_sft(model_args, model_kwargs, training_args):
    """
    Load LLaMA3 model and tokenizer for supervised fine-tuning (SFT).
    """
    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    print("==== SFT ====")
    tokenizer.pad_token = tokenizer.eos_token
    model = model_args.model_name_or_path

    print_model_info(model_args.model_name_or_path, model)
    return model, tokenizer


def print_model_info(model_name_or_path, model):
    """
    Print basic info on model parameter counts, distinguishing non-embedding params.
    """
    total_params = sum(p.numel() for p in model.parameters())
    embedding_keywords = ['embed', 'embedding', 'wte']
    embedding_params = sum(
        p.numel()
        for name, p in model.named_parameters()
        if any(kw in name.lower() for kw in embedding_keywords) and p.requires_grad
    )
    non_embedding_params = total_params - embedding_params

    def pretty(num):
        return f"{num / 1e6:.1f}M" if num < 1e9 else f"{num / 1e9:.1f}B"

    print("==== Model ====")
    print(f"Model path: {model_name_or_path}")
    print(f"Total parameters: {pretty(total_params)}")
    print(f"Non-embedding parameters: {pretty(non_embedding_params)}")


def merge_model_save(model, tokenizer, save_dir):
    """
    Merge LoRA weights and save the full model and tokenizer to disk.
    """
    merged_model = model.merge_and_unload()
    print(merged_model)
    print("LoRA model weights merged successfully.")

    os.makedirs(save_dir, exist_ok=True)
    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Merged model and tokenizer saved to {save_dir}")