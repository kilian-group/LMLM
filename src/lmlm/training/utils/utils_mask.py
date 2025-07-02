from typing import List, Tuple, Dict
from itertools import chain

import torch
import numpy as np
from transformers import PreTrainedTokenizer
from lmlm.constants import DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, DB_END_TOKEN

MASK_CATEGORIES = ["entity", "relationship", "value", "org", "pretrain"]
USE_SPECIAL_DBLOOKUP_TOKENS = True  # Set True if using special tokens for dblookup


def match_spans_single_sequence(
    s_pos: torch.Tensor,  # sorted 1D tensor of start token indices
    e_pos: torch.Tensor   # sorted 1D tensor of end token indices
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fully vectorized span matcher using searchsorted.

    Matches (s, e) such that:
    - e is the first end strictly after s
    - there is no intermediate s' with s < s' < e

    Returns:
        matched_starts: (M,) tensor of matched start indices
        matched_ends: (M,) tensor of matched end indices
    """
    if s_pos.numel() == 0 or e_pos.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    e_idx = torch.searchsorted(e_pos, s_pos, right=True)
    valid = e_idx < len(e_pos)
    s_valid = s_pos[valid]
    e_valid = e_pos[e_idx[valid]]

    s_idx = torch.arange(len(s_pos), device=s_pos.device)[valid]
    s_next_idx = torch.searchsorted(s_pos, e_valid, right=False)
    no_nested = s_next_idx <= s_idx + 1
    return s_valid[no_nested], e_valid[no_nested]

def match_spans_with_eos_wildcard(
    s_pos: torch.Tensor,
    e_pos: torch.Tensor,
    eos_pos: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs span matching treating EOS as a wildcard that can serve as a valid start or end.

    Enables:
    - start → end
    - start → EOS (if end missing)
    - EOS → end (if start missing)

    Assumes:
    - All input tensors are 1D and sorted

    Returns:
        matched_s: matched start indices
        matched_e: matched end indices
    """
    if eos_pos.numel() == 0:
      s_aug = s_pos
      e_aug = e_pos
    else:
      # Treat EOS as both valid start and valid end token
      s_aug = torch.cat([s_pos, eos_pos]).sort().values
      e_aug = torch.cat([e_pos, eos_pos]).sort().values

    s_all, e_all = match_spans_single_sequence(s_aug, e_aug)

    if s_all.numel() == 0:
        return s_all, e_all

    # Remove EOS → EOS spans
    eos_set = set(eos_pos.tolist())
    is_eos_eos = torch.tensor(
        [(s.item() in eos_set and e.item() in eos_set) for s, e in zip(s_all, e_all)],
        dtype=torch.bool,
        device=s_all.device
    )
    return s_all[~is_eos_eos], e_all[~is_eos_eos]

def extract_valid_span_indices(
    start_positions: torch.Tensor,  # (N1, 2) = [batch_idx, token_idx]
    end_positions: torch.Tensor,     # (N2, 2) = [batch_idx, token_idx]
    eos_positions: torch.Tensor     # (N3, 2) = [batch_idx, token_idx]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched span matcher using vectorized grouping per batch.

    Returns:
        batch_ids: (M,) tensor of batch indices
        matched_starts: (M,) tensor of matched start positions
        matched_ends: (M,) tensor of matched end positions
    """
    matched_batches = []
    matched_starts = []
    matched_ends = []

    all_batch_ids = torch.cat([
        start_positions[:, 0],
        end_positions[:, 0],
        eos_positions[:, 0]
    ])
    unique_batches = torch.unique(all_batch_ids)

    for b in unique_batches.tolist():
        s_pos = start_positions[start_positions[:, 0] == b][:, 1].sort()[0]
        e_pos = end_positions[end_positions[:, 0] == b][:, 1].sort()[0]
        eos_pos = eos_positions[eos_positions[:, 0] == b][:, 1].sort()[0]

        matched_s, matched_e = match_spans_with_eos_wildcard(s_pos, e_pos, eos_pos)

        if matched_s.numel() == 0:
            continue

        matched_batches.append(torch.full_like(matched_s, b))
        matched_starts.append(matched_s)
        matched_ends.append(matched_e)

    if not matched_batches:
        return (torch.empty(0, dtype=torch.long),) * 3

    return (
        torch.cat(matched_batches, dim=0),
        torch.cat(matched_starts, dim=0),
        torch.cat(matched_ends, dim=0),
    )

def create_mask_from_spans(
    batch_ids: torch.Tensor,
    start_indices: torch.Tensor,
    end_indices: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros((batch_size, seq_len), dtype=torch.int32, device=device)

    if len(batch_ids) == 0:
        return mask.bool()

    span_starts = start_indices + 1
    span_ends = end_indices

    mask.index_put_((batch_ids, span_starts), torch.ones_like(span_starts, dtype=mask.dtype, device=device), accumulate=True)
    mask.index_put_((batch_ids, span_ends), -torch.ones_like(span_ends, dtype=mask.dtype, device=device), accumulate=True)

    mask = torch.cumsum(mask, dim=1)
    return mask > 0



def get_span_mask(
    tokens: torch.Tensor,
    start_token_id: int,
    end_token_id: int,
    eos_token_id: int
) -> torch.Tensor:
    """
    High-level API: extracts span mask where each start is valid iff it is
    closed by the next end with no intervening start.

    Args:
        tokens: (B, T) tensor of token IDs
        start_token_id: ID marking span starts
        end_token_id: ID marking span ends

    Returns:
        (B, T) boolean mask
    """
    B, T = tokens.shape

    assert start_token_id and end_token_id and eos_token_id, "Token IDs must be provided"
    start_pos = (tokens == start_token_id).nonzero(as_tuple=False)  # (N1, 2)
    end_pos = (tokens == end_token_id).nonzero(as_tuple=False)      # (N2, 2)
    eos_pos = (tokens == eos_token_id).nonzero(as_tuple=False)      # (N3, 2

    batch_ids, start_idx, end_idx = extract_valid_span_indices(start_pos, end_pos, eos_pos)
    return create_mask_from_spans(batch_ids, start_idx, end_idx, B, T, tokens.device)


def extract_dblookup_masks(
    tokens: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    pretrain_mask_only: bool = False,
    include_eos: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Extracts boolean masks for entity, relationship, value, and full dblookup spans
    from a tokenized batch using special dblookup tokens.

    Returns:
        A dictionary of boolean masks (each of shape B x T)
    """
    special_ids = {
        "entity": tokenizer.convert_tokens_to_ids(DB_START_TOKEN), # if not found, it will be 0
        "rel": tokenizer.convert_tokens_to_ids(DB_SEP_TOKEN),
        "return": tokenizer.convert_tokens_to_ids(DB_RETRIEVE_TOKEN),
        "end": tokenizer.convert_tokens_to_ids(DB_END_TOKEN),
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }

    B, T = tokens.shape
    device = tokens.device

    # if the tokenizer does not have the special tokens, return org_mask and pretrain_mask is the all, others are all 0
    if all(special_ids[t] == 0 for t in ["entity", "rel", "return", "end"]):
        pretrain_mask = torch.ones_like(tokens, dtype=torch.bool)
        if pretrain_mask_only:
            return {"pretrain": pretrain_mask}
        zero_mask = torch.zeros_like(tokens, dtype=torch.bool)
        return {
            "entity": zero_mask, "relationship": zero_mask, "value": zero_mask,
            "dblookup": zero_mask, "org": pretrain_mask, "pretrain": pretrain_mask
        }

    if pretrain_mask_only:  
        # Token-level masks
        pad_mask = tokens == special_ids["pad"]
        value_mask = get_span_mask(tokens, special_ids["return"], special_ids["end"], special_ids["eos"])
        
        end_token_mask = (tokens == special_ids["end"]).to(device)
        pretrain_mask = ~(value_mask | end_token_mask)
        pretrain_mask[pad_mask] = 0

        if include_eos:
            pretrain_mask[end_token_mask] = 1

        return {"pretrain": pretrain_mask}

    # Main masks
    entity_mask = get_span_mask(tokens, special_ids["entity"], special_ids["rel"], special_ids["eos"])
    rel_mask    = get_span_mask(tokens, special_ids["rel"], special_ids["return"], special_ids["eos"])
    value_mask  = get_span_mask(tokens, special_ids["return"], special_ids["end"], special_ids["eos"])
    db_span     = get_span_mask(tokens, special_ids["entity"], special_ids["end"], special_ids["eos"])

    special_token_ids = torch.tensor(
        [special_ids[k] for k in ["entity", "rel", "return", "end"]],
        device=tokens.device
    )
    special_token_mask = (tokens[..., None] == special_token_ids).any(dim=-1)
    db_span[special_token_mask] = 1  # zero out boundaries

    # Token-level masks
    pad_mask = tokens == special_ids["pad"]

    # org = everything not part of dblookup
    org_mask = ~db_span
    org_mask[pad_mask] = 0

    end_token_mask = (tokens == special_ids["end"])
    pretrain_mask = ~(value_mask | end_token_mask)
    pretrain_mask[pad_mask] = 0

    return {
        "entity": entity_mask,
        "relationship": rel_mask,
        "value": value_mask,
        "dblookup": db_span,
        "org": org_mask,
        "pretrain": pretrain_mask
    }

def indices_to_mask(text_len, results, pretrain_mask_only=False, org_mask_only=False):
    """
    Converts extracted token indices into a binary mask batch.

    Args:
        text_len (int): The length of the tokenized text.
        results (list): The extracted token indices from entity detection.

    Returns:
        dict: A dictionary containing masks for each category.
    """
    bsz = len(results)  # Batch size is simply the length of results
    mask_batch = {}

    # Define MASK_CATEGORIES based on `results` structure

    # Initialize masks for each category
    for category in MASK_CATEGORIES:
        if pretrain_mask_only and category != "pretrain":
            continue
        mask_batch[category] = torch.zeros((bsz, text_len), dtype=torch.float32)

    # Iterate over each batch and update corresponding masks
    for batch_idx, indices_group in enumerate(results):
        for category, indices in zip(MASK_CATEGORIES, indices_group):
            if pretrain_mask_only and category != "pretrain":
                continue
            if org_mask_only and category != "org":
                continue
            if indices:  # Ensure indices exist
                flat_indices = list(chain(*indices)) if isinstance(indices[0], list) else indices
                # filter out the index that is out of range
                flat_indices = [idx for idx in flat_indices if idx < mask_batch[category].shape[1]]
                mask_batch[category][batch_idx, flat_indices] = 1.0  # Set mask values to 1

    return mask_batch


def validate_mask_tokens(mask_batch, processed_token_lst_batch):
    """
    Validates the mask by replacing masked tokens with 0 while keeping unmasked tokens unchanged.

    Args:
        mask_batch (dict): A dictionary containing binary masks for different MASK_CATEGORIES.
        processed_token_lst_batch (list): List of batches, where each batch is a list of token IDs.

    Returns:
        dict: A dictionary containing masked token lists for each category.
    """
    masked_tokens = {}
    bsz = len(processed_token_lst_batch)  # Batch size
    text_len = len(processed_token_lst_batch[0])  # Assuming all sequences have the same length

    # Define MASK_CATEGORIES to process
    MASK_CATEGORIES = mask_batch.keys()

    # Initialize masked tokens for each category
    for category in MASK_CATEGORIES:
        masked_tokens[category] = []

    # Process each batch
    for batch_idx in range(bsz):
        for category in MASK_CATEGORIES:
            original_tokens = processed_token_lst_batch[batch_idx]
            mask = mask_batch[category][batch_idx]  # Get the mask for this batch

            # Replace masked positions with 0
            masked_token_list = [
                original_tokens[i] if mask[i] == 0 and i < len(original_tokens) else 0 for i in range(text_len)
            ]

            masked_tokens[category].append(masked_token_list)

    for key, value in masked_tokens.items():
        decoded_masked_tokens = []
        for token_ids in value[0]:
            if token_ids > 0: 
                decoded_masked_tokens.append(TOKENIZER.decode(token_ids, skip_special_tokens=False))
            elif token_ids == 0:
                decoded_masked_tokens.append("[TARGET]")
            elif token_ids == -100:
                decoded_masked_tokens.append("[-100]")
            
        print(f"Category: {key}")
        print(decoded_masked_tokens)  
    return masked_tokens


def mask_to_spans(mask_row: np.ndarray) -> List[Tuple[int, int]]:
    """
    Convert a 1D boolean mask to a list of (start, end) index spans.
    Each span is inclusive of start and exclusive of end.
    """
    spans = []
    in_span = False
    for i, val in enumerate(mask_row):
        if val and not in_span:
            start = i
            in_span = True
        elif not val and in_span:
            spans.append((start, i))
            in_span = False
    if in_span:
        spans.append((start, len(mask_row)))
    return spans

def mask_to_span_dict(
    mask_dict: Dict[str, torch.Tensor]
) -> Dict[str, List[List[Tuple[int, int]]]]:
    """
    Convert a dictionary of (B x T) boolean masks to a dictionary of (B x List[Tuple[int, int]]) span indices.

    Args:
        mask_dict: dictionary mapping labels to boolean masks of shape (B, T)

    Returns:
        Dictionary of the same keys mapping to lists of per-sample (start, end) index spans.
    """
    span_dict = {}

    for label, mask in mask_dict.items():
        span_dict[label] = []
        if mask is None:
            span_dict[label] = None
            continue
        for row in mask.cpu().numpy():  # convert row-wise to NumPy for scanning
            span_dict[label].append(mask_to_spans(row))

    return span_dict

def validate_extraction_from_masks(
    processed_token_lst_batch: List[List[int]],
    masks: Tuple[np.ndarray, ...]
) -> List[dict]:
    """
    Validates the extracted token spans using boolean masks, and splits spans into chunks.
    Each category will be decoded into a list of text chunks.

    Args:
        processed_token_lst_batch: List of token ID sequences (batch_size, seq_len)
        masks: Tuple of boolean masks (each of shape (batch_size, seq_len)),
               in the same order as MASK_CATEGORIES.

    Returns:
        A list of dictionaries per example, mapping each category to a list of decoded spans.
    """
    assert len(masks) == len(MASK_CATEGORIES), "Mismatch between masks and category labels"

    validation_results = []
    batch_size = len(processed_token_lst_batch)
    
    for b in range(batch_size):
        token_ids = processed_token_lst_batch[b]
        batch_result = {}

        for label in MASK_CATEGORIES:
            if masks[label] is None:
                batch_result[f"extracted_{label}"] = None
                continue
            mask = masks[label]
            span_texts = []
            spans = mask_to_spans(mask[b])
            for start, end in spans:
                span_token_ids = token_ids[start:end]
                decoded = TOKENIZER.decode(
                    span_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                span_texts.append(decoded)
            batch_result[f"extracted_{label}"] = span_texts

        validation_results.append(batch_result)

    print("Validation Results:")
    print(json.dumps(validation_results, indent=4))

    return validation_results    


def validate_extraction(processed_token_lst_batch, results: List[Tuple[List[List[int]]]]) -> List[dict]:
    """
    Validates the extracted entity, relationship, value, and other token indices by decoding them back into text.

    Args:
        processed_token_lst_batch: List of batches, where each batch is a list of token IDs.
        results: List of tuples containing lists of token indices for different MASK_CATEGORIES.

    Returns:
        A list of validation results with extracted tokens mapped to text.
    """
    validation_results = []

    for batch_idx, indices_group in enumerate(results):
        token_ids = processed_token_lst_batch[batch_idx]

        # Iterate over each set of indices dynamically
        # label_names = [
        #     "extracted_entity", "extracted_relationship", "extracted_value",
        #     "extracted_bracket_start", "extracted_bracket_end", "extracted_org", "extracted_pretrain"
        # ]
        if len(indices_group) == 5:
            label_names = [
                "extracted_entity", "extracted_relationship", "extracted_value", "extracted_org", "extracted_pretrain"
            ]
        elif len(indices_group) == 1:
            label_names = [
                "extracted_pretrain"
            ]
        elif len(indices_group) == 7:
            label_names = [
                "extracted_entity", "extracted_relationship", "extracted_value", "extracted_bracket_start", "extracted_bracket_end", "extracted_org", "extracted_pretrain"
            ]
        else:
            raise ValueError("Invalid number of indices in results.")
        
        batch_result = {label: [] for label in label_names}

        ignore_index = TOKENIZER.pad_token_id if TOKENIZER.pad_token_id is not None else TOKENIZER.eos_token_id
        # token_ids = [t if 0 <= t < len(TOKENIZER) else ignore_index for t in token_ids]

        for label, index_list in zip(label_names, indices_group):
            # Check if index list exists (some may be empty)
            if index_list:
                if isinstance(index_list[0], int):
                    batch_result[label].append(TOKENIZER.decode([token_ids[i] for i in index_list if 0<=token_ids[i]<len(TOKENIZER)], skip_special_tokens=False))
                elif isinstance(index_list[0], list):
                    for indices in index_list:
                        batch_result[label].append(TOKENIZER.decode([token_ids[i] for i in indices if 0<=token_ids[i]<len(TOKENIZER)], skip_special_tokens=False)) 
                else:
                    raise ValueError("Invalid index list format.")
        
        validation_results.append(batch_result)
    
    print("Validation Results:")    
    print(json.dumps(validation_results, indent=4))

    return validation_results
