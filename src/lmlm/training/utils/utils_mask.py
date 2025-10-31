from typing import List, Tuple, Dict
from itertools import chain

import torch
import numpy as np
from transformers import PreTrainedTokenizer
from lmlm.constants import DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, DB_END_TOKEN

MASK_CATEGORIES = ["entity", "relationship", "value", "org", "pretrain"]
USE_SPECIAL_DBLOOKUP_TOKENS = True  # Set to True if using special tokens for dblookup


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


def match_spans_with_bos_eos_wildcards(
    s_pos: torch.Tensor,   # 1D sorted tensor of start positions (in one sequence)
    e_pos: torch.Tensor,   # 1D sorted tensor of end positions (in one sequence)
    eos_pos: torch.Tensor, # 1D sorted tensor; typically tensor([T]) if you appended EOS at T
    bos_pos: torch.Tensor  # 1D sorted tensor; typically tensor([0]) to use BOS as fallback
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Greedily matches spans in a single sequence with wildcards:
      - start -> end                (normal)
      - start -> EOS (if no end)    (end fallback)
      - BOS   -> end (if no start)  (start fallback)

    BOS is only used as a start fallback, EOS only as an end fallback.
    BOS->EOS (both wildcards) is dropped.
    Assumes inputs are sorted and 1-D.
    """

    device = s_pos.device
    dtype  = s_pos.dtype

    # Augment *starts* with BOS (if provided), *ends* with EOS (if provided).
    s_aug = s_pos
    if bos_pos.numel() > 0:
        s_aug = torch.unique(torch.cat([s_aug, bos_pos.to(device=device, dtype=dtype)]), sorted=True)

    e_aug = e_pos
    if eos_pos.numel() > 0:
        e_aug = torch.unique(torch.cat([e_aug, eos_pos.to(device=device, dtype=dtype)]), sorted=True)

    # Do the greedy 1D match (assumes s_aug and e_aug sorted, returns aligned pairs)
    s_all, e_all = match_spans_single_sequence(s_aug, e_aug)  # your existing matcher

    if s_all.numel() == 0:
        return s_all, e_all


    return s_all, e_all


def extract_valid_span_indices(
    start_positions: torch.Tensor,  # (N1, 2) = [batch_idx, token_idx]
    end_positions: torch.Tensor,     # (N2, 2) = [batch_idx, token_idx]
    eos_positions: torch.Tensor,     # (N3, 2) = [batch_idx, token_idx]
    bos_positions: torch.Tensor     # (N4, 2) = [batch_idx, token_idx]
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
        eos_positions[:, 0],
        bos_positions[:, 0]
    ])
    unique_batches = torch.unique(all_batch_ids)

    for b in unique_batches.tolist():
        s_pos = start_positions[start_positions[:, 0] == b][:, 1].sort()[0]
        e_pos = end_positions[end_positions[:, 0] == b][:, 1].sort()[0]
        eos_pos = eos_positions[eos_positions[:, 0] == b][:, 1].sort()[0]
        bos_pos = bos_positions[bos_positions[:, 0] == b][:, 1].sort()[0]

        matched_s, matched_e = match_spans_with_bos_eos_wildcards(s_pos, e_pos, eos_pos, bos_pos)

        if matched_s.numel() == 0:
            continue

        matched_batches.append(torch.full_like(matched_s, b))
        matched_starts.append(matched_s)
        matched_ends.append(matched_e)

    if len(matched_batches) == 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long)
        )

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
    # +2 for the bos and eos token
    mask = torch.zeros((batch_size, seq_len+2), dtype=torch.int32, device=device)

    if len(batch_ids) == 0:
        return mask.bool()

    span_starts = start_indices + 1
    span_ends = end_indices

    mask.index_put_((batch_ids, span_starts), torch.ones_like(span_starts, dtype=mask.dtype, device=device), accumulate=True)
    mask.index_put_((batch_ids, span_ends), -torch.ones_like(span_ends, dtype=mask.dtype, device=device), accumulate=True)

    mask = torch.cumsum(mask, dim=1)
    # remove the first and last column
    mask = mask[:, 1:-1]
    return mask > 0



def get_span_mask(
    tokens: torch.Tensor,
    start_token_id: int,
    end_token_id: int,
    eos_token_id: int,
    bos_token_id: int
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

    # Append BOS and EOS token for fallback span termination
    eos_col = torch.full((B, 1), eos_token_id, dtype=torch.long, device=tokens.device)
    bos_col = torch.full((B, 1), bos_token_id, dtype=torch.long, device=tokens.device)
    tokens = torch.cat([bos_col, tokens, eos_col], dim=1)  # shape (B, T+2)

    assert start_token_id and end_token_id and eos_token_id, "Token IDs must be provided"
    start_pos = (tokens == start_token_id).nonzero(as_tuple=False)  # (N1, 2)
    end_pos = (tokens == end_token_id).nonzero(as_tuple=False)      # (N2, 2)
    eos_pos = (tokens == eos_token_id).nonzero(as_tuple=False)      # (N3, 2)
    bos_pos = (tokens == bos_token_id).nonzero(as_tuple=False)      # (N4, 2)

    batch_ids, start_idx, end_idx = extract_valid_span_indices(start_pos, end_pos, eos_pos, bos_pos)
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
        "bos": tokenizer.bos_token_id,
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
        pad_mask = pad_mask.to(device)

        value_mask  = get_span_mask(tokens, special_ids["return"], special_ids["end"], special_ids["eos"], bos_token_id=special_ids["bos"])
        
        end_token_mask = (tokens == special_ids["end"]).to(device)
        pretrain_mask = ~(value_mask | end_token_mask)
        pretrain_mask[pad_mask] = 0

        if include_eos:
            pretrain_mask[end_token_mask] = 1

        return {"pretrain": pretrain_mask}

    # Main masks
    entity_mask = get_span_mask(tokens, special_ids["entity"], special_ids["rel"], special_ids["eos"], bos_token_id=special_ids["bos"])
    rel_mask    = get_span_mask(tokens, special_ids["rel"], special_ids["return"], special_ids["eos"], bos_token_id=special_ids["bos"])
    value_mask  = get_span_mask(tokens, special_ids["return"], special_ids["end"], special_ids["eos"], bos_token_id=special_ids["bos"])
    db_span     = get_span_mask(tokens, special_ids["entity"], special_ids["end"], special_ids["eos"], bos_token_id=special_ids["bos"])

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


