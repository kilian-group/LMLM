"""
Metrics and utilities for evaluating language models with database lookup functionality.
This module provides functions for computing losses, perplexity, and statistics for models
that perform database lookups within generated text.
"""

import os
import json
import time

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import wandb
from transformers import EvalPrediction, AutoTokenizer
from lmlm.training.utils.utils_mask import extract_dblookup_masks, MASK_CATEGORIES


# --------------------
# Global Configurations
# --------------------

TOKENIZER = None
DATASET_NAME = None
IS_CLEANED_DATASET = False
USE_SPECIAL_DBLOOKUP_TOKENS = False


def set_tokenizer(tokenizer):
    global TOKENIZER
    if tokenizer:
        TOKENIZER = tokenizer
    else: 
        TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
        TOKENIZER.pad_token = TOKENIZER.eos_token

def set_wandb():
    if wandb.run is None:
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
        )


def set_dataset_name(name=None):
    global DATASET_NAME
    if name:
        DATASET_NAME = name
    else:
        DATASET_NAME = time.strftime("%Y%m%d_%H%M%S")  # Provide a default name if none is given.


def set_is_cleaned_dataset(is_cleaned_dataset=False):
    global IS_CLEANED_DATASET
    IS_CLEANED_DATASET = is_cleaned_dataset


def set_use_special_dblookup_tokens(use_special_dblookup_tokens=False):
    global USE_SPECIAL_DBLOOKUP_TOKENS
    USE_SPECIAL_DBLOOKUP_TOKENS = use_special_dblookup_tokens


# --------------------
# Loss Functions
############################################

def compute_loss_func(outputs, labels, num_items_in_batch, include_eos=False):

    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    pretrained_mask = compute_pretrain_mask(shift_labels, include_eos=include_eos)

    loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
    per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(labels.size(0), -1)
    if pretrained_mask.shape != per_token_loss.shape:
        pretrained_mask = pretrained_mask.view(per_token_loss.size(0), -1)

    weighted_loss = per_token_loss[pretrained_mask != 0]

    if num_items_in_batch is None:
        return weighted_loss.mean()
    else:
        return weighted_loss.sum() / num_items_in_batch


def compute_pretrain_mask(shift_labels, include_eos=False):
    mask_batch = extract_dblookup_masks(shift_labels, TOKENIZER, pretrain_mask_only=True, include_eos=include_eos)
    valid_mask = shift_labels != -100
    pretrain_mask = mask_batch["pretrain"] & valid_mask
    return pretrain_mask # same shape as shift_labels

def compute_org_mask(shift_labels, include_eos=False):   
    mask_batch = extract_dblookup_masks(shift_labels, TOKENIZER, pretrain_mask_only=False, include_eos=include_eos)
    valid_mask = shift_labels != -100
    org_mask = mask_batch["org"] & valid_mask

    return org_mask # same shape as shift_labels

# --------------------
# Perplexity Metrics
# --------------------
def compute_metrics(eval_preds: EvalPrediction):
    """
    Compute metrics for language modeling, specifically loss and perplexity.

    Args:
        eval_preds (EvalPrediction): Contains predictions and labels for evaluation.

    Returns:
        dict: Dictionary with the computed loss and perplexity.
    """

    logits, labels = eval_preds

    if isinstance(logits, tuple):
        logits = logits[0]
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    predictions = logits.argmax(dim=-1)
    
    results = {}
    results.update(compute_ppl(predictions, logits, labels))
    results.update(compute_mask_ppl(predictions, logits, labels))    
    return results


def compute_ppl(predictions, logits, labels):   

    if logits.dim() == 3: 
        shift_logits = logits[..., :-1, :].contiguous()  # Exclude the last token prediction
        shift_labels = labels[..., 1:].contiguous()  # Exclude the first token label
        
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # [batch_size * (sequence_length - 1), vocab_size]
        shift_labels = shift_labels.view(-1)  # [batch_size * (sequence_length - 1)]
        
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits, shift_labels)

    elif logits.shape == labels.shape:
        shift_logits = logits[:, :-1].contiguous()  # [batch_size, seq_length - 1]
        shift_labels = labels[:, 1:].contiguous()
        # Mask out -100 tokens
        valid_mask = shift_labels != -100  # Exclude tokens that should not contribute to loss

        # Compute loss only on valid tokens
        masked_logits = shift_logits[valid_mask]
        loss = -masked_logits.sum() / valid_mask.sum()  # Normalize by number of valid tokens
    else:
        raise ValueError(f"Invalid shapes for logits and labels: {logits.shape} vs {labels.shape}") 
    
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    results = {
        "loss": loss.item(),
        "ppl": perplexity.item(),
    }
    return results
    

def compute_mask_ppl(predictions, logits, labels, include_eos=False):
    """
    Computes masked perplexity and loss for different categories.

    Args:
        predictions (torch.Tensor): Predicted token IDs.
        logits (torch.Tensor): Model output logits.
        labels (torch.Tensor): Ground-truth token labels.

    Returns:
        dict: Dictionary containing loss and perplexity values for each mask category.
    """
    shift_labels = labels[..., 1:]
    
    if logits.dim() == 3:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = shift_labels.contiguous()

        loss_fct = CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(labels.size(0), -1)
    
    elif logits.shape == labels.shape:
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = shift_labels.contiguous()
        
        # Mask out `-100` tokens
        valid_mask = shift_labels != -100  # Create a mask for valid labels
        per_token_loss = torch.full_like(shift_logits, 0.0)  # Initialize per-token loss tensor
        per_token_loss[valid_mask] = -shift_logits[valid_mask]  # Apply loss only on valid tokens
    
    else:
        raise ValueError(f"Invalid shapes for logits and labels: {logits.shape} vs {labels.shape}")
    
    mask_batch = extract_dblookup_masks(shift_labels, TOKENIZER, include_eos=include_eos)

    # Compute loss and perplexity for each mask category
    results = {}
    token_counts = {}
    for key in MASK_CATEGORIES:

        valid_mask = shift_labels != -100
        mask = mask_batch[key] != 0
        mask = mask & valid_mask
        token_count = mask.sum().item()

        if token_count > 0:
            losses = per_token_loss[mask]
            masked_loss = losses.mean()
            assert torch.isclose(masked_loss, losses.sum() / token_count)

            masked_std = losses.std(unbiased=True)  # population std; use unbiased=True for sample std
            ppl = torch.exp(masked_loss) if masked_loss != 0 else torch.tensor(0.0, device=per_token_loss.device)
        else:
            masked_loss = torch.tensor(0.0, device=per_token_loss.device)
            masked_std = torch.tensor(0.0, device=per_token_loss.device)
            ppl = torch.tensor(0.0, device=per_token_loss.device)

        results[f"loss_{key}"] = masked_loss.item()
        results[f"loss_std_{key}"] = masked_std.item()
        results[f"ppl_{key}"] = ppl.item()
        token_counts[f"tokens_{key}"] = token_count
    
    results.update(token_counts)

    # add NLL
    if "loss_pretrain" in results and token_counts["tokens_pretrain"] > 0:
        pretrain_loss = results["loss_pretrain"]
        pretrain_tokens = token_counts["tokens_pretrain"]
        total_valid_tokens = token_counts["tokens_org"] 
        
        normalized_nll = pretrain_loss * (pretrain_tokens / total_valid_tokens)
        results["normalized_nll"] = normalized_nll

        if "loss_std_pretrain" in results and pretrain_tokens > 0:
            results["normalized_nll_std"] = results["loss_std_pretrain"] * (pretrain_tokens ** 0.5) / total_valid_tokens

    return results


def preprocess_logits_for_metrics(logits, labels):
    """
    Extracts relevant logits for perplexity calculation while keeping the original shape (batch_size, seq_len),
    and pads the first token position with 0.

    Args:
        logits (torch.Tensor): Model output logits of shape (batch_size, seq_len, vocab_size)
        labels (torch.Tensor): Tokenized labels of shape (batch_size, seq_len)

    Returns:
        torch.Tensor: Log probabilities for the target tokens, restored to shape (batch_size, seq_len).
    """
    # Shift logits and labels
    shift_logits = logits[..., :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
    shift_labels = labels[..., 1:].contiguous()      # (batch_size, seq_len-1)

    # Compute log probabilities
    log_probs = torch.log_softmax(shift_logits, dim=-1)  # (batch_size, seq_len-1, vocab_size)

    # Create mask for valid labels
    valid_mask = (shift_labels != -100)
    safe_labels = shift_labels.clone()
    safe_labels[~valid_mask] = 0  # Temporary safe index

    # Gather log probabilities for valid labels
    selected_logits = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)

    # Mask out invalid positions
    selected_logits[~valid_mask] = 0.0  # Or float('nan') if preferred

    # Pad back to original shape (batch_size, seq_len)
    pad = torch.zeros((logits.shape[0], 1), device=logits.device)
    restored_logits = torch.cat([selected_logits, pad], dim=-1)  # (batch_size, seq_len)

    return restored_logits


# --------------------
# Dataset Statistics
# --------------------

def dataset_stats(dataset_name, dataset, tokenizer=None, visualize=False):
    stats = {}
    stats["dataset"] = {"num_examples": len(dataset)}

    num_subset = min(1000, len(dataset))
    ratio = len(dataset) / num_subset
    subset = dataset.shuffle(seed=42).select(range(num_subset))

    text_column_dict = {"annotated_text": "output", "text": "input"}

    for text_column, display_key in text_column_dict.items():
        if subset and text_column in subset.column_names:
            unique_texts = set(subset[text_column])
            words = [word for text in subset[text_column] for word in text.split()]
            vocab_size = len(set(words))
            type_token_ratio = vocab_size / len(words)
            redundancy_ratio = 1 - (len(unique_texts) / len(subset))

            if tokenizer:
                def tokenize_example(example):
                    tokens = tokenizer(example[text_column], return_tensors="pt", truncation=False, padding=False)
                    example["num_tokens"] = tokens["input_ids"].shape[1]
                    return example

                tokenized_subset = subset.map(tokenize_example)
                token_counts = tokenized_subset["num_tokens"]

                stats[display_key] = {
                    "unique_count": len(unique_texts) * ratio,
                    "total_tokens_million": sum(token_counts) * ratio / 1e6,
                    "max_tokens": max(token_counts),
                    "min_tokens": min(token_counts),
                    "avg_tokens": sum(token_counts) / len(token_counts),
                    "median_tokens": np.median(token_counts),
                    "std_tokens": np.std(token_counts),
                    "vocab_size": vocab_size,
                    "type_token_ratio": type_token_ratio,
                    "redundancy_ratio": redundancy_ratio
                }

    if "annotated_text" in subset.column_names and "text" in subset.column_names:
        num_shorter = sum(len(e["annotated_text"]) < len(e["text"]) for e in subset)
        avg_length_ratio = sum(len(e["annotated_text"]) / len(e["text"]) for e in subset) / len(subset)

        stats["compare"] = {
            "compression_ratio": sum(len(e["annotated_text"]) for e in subset) / sum(len(e["text"]) for e in subset),
            "longer_than_text_ratio": sum(len(e["annotated_text"]) > len(e["text"]) for e in subset) / len(subset),
            "modification_ratio": sum(e["annotated_text"] != e["text"] for e in subset) / len(subset),
            "shorter_than_text_ratio": round(num_shorter / len(subset), 2),
            "to_text_avg_length_ratio": avg_length_ratio
        }

    try:
        from lmlm.database.database_manager import extract_database
        from lmlm.training.utils.utils_filter import clean_dataset

        subset = extract_database(subset)
        db_calls = [len(e["atomic_knowledge"]) for e in subset]

        stats["db_calls"] = {
            "total": sum(db_calls) * ratio,
            "avg_per_example": np.mean(db_calls),
            "max_per_example": max(db_calls),
            "min_per_example": min(db_calls),
            "std": np.std(db_calls),
            "total_cleaned": sum(len(e["atomic_knowledge"]) for e in clean_dataset(subset)) * ratio
        }

    except (ImportError, KeyError) as e:
        print(f"Could not extract database statistics: {e}")

    stats["dataset_name"] = dataset_name
    print("Dataset Stats:")
    print(json.dumps(stats, indent=4))

    if wandb.run:
        wandb.log(stats)

    if visualize and tokenizer:
        plt.hist(token_counts, bins=50, edgecolor='black')
        plt.xlabel("Number of Tokens")
        plt.ylabel("Frequency")
        wandb.log({"input_length_distribution": wandb.Image(plt)})
        plt.close()
