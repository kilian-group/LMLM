import re
import string
import torch
from collections import defaultdict, Counter
from typing import List, Tuple


def truncate_sample_length(example, max_length=1024):
    words = example['text'].split()
    if len(words) > max_length:
        example['text'] = ' '.join(words[:max_length])
    return example


def add_shared_context_ids(dataset):
    context_to_ids = defaultdict(list)
    for ex in dataset:
        context_to_ids[ex['context']].append(ex['id'])
    return dataset.add_column("shared_ids", [context_to_ids[ex['context']] for ex in dataset])


def chunk_wiki_text(texts: List[str], ids: List[str], max_len: int = 750) -> Tuple[List[str], List[str]]:
    chunks, chunk_ids = [], []
    for text, pid in zip(texts, ids):
        tokens = text.split()
        for i in range(0, len(tokens), max_len):
            chunk = " ".join(tokens[i:i + max_len])
            chunks.append(chunk)
            chunk_ids.append(f"{pid}_chunk{i // max_len}")
    return chunks, chunk_ids

def truncate_prompt(prompt: str, tokenizer, max_tokens: int = 2048) -> str:
    """
    Truncates the input prompt to ensure it does not exceed the max token limit.

    Args:
        prompt (str): The input prompt text.
        tokenizer: The tokenizer used for tokenizing the prompt.
        max_tokens (int, optional): The maximum allowed token length. Defaults to 2048.

    Returns:
        str: The truncated prompt.
    """
    tokens = tokenizer.encode(prompt, add_special_tokens=False)

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]  # Truncate to max length

    return tokenizer.decode(tokens, skip_special_tokens=True)


import re

def get_save_name(args):
    """
    Generate a filename for saving based on model and annotator info.
    """
    # Extract model name from path
    model_name = args.model_id.rstrip('/').split('/')[-1]

    # Extract model size (e.g., '7B') if present
    size_match = re.search(r'(\d+B)', model_name, re.IGNORECASE)
    model_size = size_match.group(1).lower() if size_match else ""

    # Format annotator name
    if args.annotator in {"llama", "llama-lora-ft", "llama-lora-ft-hf"}:
        suffix = model_name.split('_', 1)[-1]
        if args.annotator == "llama":
            annotator = f"llama{model_size}-{suffix}"
        else:
            annotator = f"llama{model_size}-lora-ft-{suffix}"
    else:
        annotator = args.annotator

    return f"{args.manager}_{annotator}_{args.prompt_id}"
