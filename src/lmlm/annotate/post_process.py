import re
import Levenshtein
from datasets import load_dataset
from fuzzysearch import find_near_matches
from functools import lru_cache
import multiprocessing as mp
from typing import List, Dict, Tuple, Optional

# Global configuration
THRESHOLD = 0.9
DIST_K = 2  # Allowed edit distance (sum of insertions/deletions/substitutions)
MASK_TOKEN = ' @@@@ '

# Pre-compile regex patterns for better performance
DBLOOKUP_PATTERN = re.compile(r"\[dblookup.*? -> (.*?)\]")
BLANK_PATTERN = re.compile(r'(\[dblookup[^\]]* -> [^\]]*\]) ')

# Cache for Levenshtein ratio calculations
@lru_cache(maxsize=10000)
def cached_levenshtein_ratio(s1: str, s2: str) -> float:
    """Cached Levenshtein ratio calculation"""
    return Levenshtein.ratio(s1, s2)

# Global configuration complete - no additional global variables needed
def find_dblookup(text: str, mask_text: bool = False) -> Tuple[List[Dict], str]:
    """
    Extract dblookup patterns from text. 
    
    Args:
        text (str): Text to search
        mask_text (bool): If True, replace dblookup with mask token
        
    Returns:
        tuple: (results, text_masked)
            - results: List of extracted dblookup information
            - text_masked: Masked text (when mask_text=True)
    """
    results = []
    text_masked = text
    
    # Use pre-compiled pattern for better performance
    for match in DBLOOKUP_PATTERN.finditer(text):
        entity = match.group(1)
        start, end = match.start(), match.end()
        dbquery = text[start:end]
        results.append({
            "entity": entity,
            "start": start,
            "end": end,
            "dbquery": dbquery
        })

    if mask_text and results:
        # More efficient string replacement using list comprehension
        replacements = [(result['dbquery'], MASK_TOKEN) for result in results]
        for old, new in replacements:
            text_masked = text_masked.replace(old, new, 1)
    
    return results, text_masked



def find_by_ratio_fuzzysearch(pat: str, text: str, max_k: int = DIST_K, min_ratio: float = 0.8) -> List[Dict]:
    """
    Find pattern in text using fuzzy search. (Optimized version, case-insensitive)
    
    Args:
        pat (str): Pattern to search for
        text (str): Target text to search
        max_k (int): Maximum edit distance
        min_ratio (float): Minimum similarity ratio
        
    Returns:
        list: List of matched results
    """
    # Early return for empty pattern or text
    if not pat or not text:
        return []
    
    # Convert to lowercase for case-insensitive matching
    pat_lower = pat.lower()
    text_lower = text.lower()
    
    cand = find_near_matches(pat_lower, text_lower, max_l_dist=max_k)
    if not cand:
        return []

    # Pre-calculate pattern length for efficiency
    pat_len = len(pat)
    
    # Use list comprehension for better performance
    match_list = []
    for mobj in cand:
        start, end = mobj.start, mobj.end
        d = mobj.dist
        matched_len = max(1, end - start)
        ratio = 1 - d / max(pat_len, matched_len)
        
        if ratio >= min_ratio:
            match_list.append({
                "start": start, 
                "end": end, 
                "match": text[start:end],  # Return original case from original text
                "dist": d, 
                "ratio": ratio
            })
    
    return match_list


def find_match_idx(entity: str, following_text: str, max_l_dist: int = DIST_K) -> Optional[Dict]:
    """
    Find matching index for entity in following text. 
    
    Args:
        entity (str): Entity to match
        following_text (str): Text to search
        max_l_dist (int): Maximum edit distance
        
    Returns:
        dict or None: Matched result or None
    """
    # Early return for empty inputs
    if not entity or not following_text:
        return None
    
    hits = find_by_ratio_fuzzysearch(entity, following_text, max_l_dist)
    if not hits:
        return None
    
    # Return first match (most relevant)
    h = hits[0]
    return {
        "start": h['start'], 
        "end": h['end'], 
        "matched": h['match'], 
        "distance": h['dist']
    }

def process_single_text(text: str) -> Dict[str, any]:
    """
    Process single text to remove incorrect dblookups. 
    
    Args:
        text (str): Text to process
        
    Returns:
        dict: {'processed_text': str, 'changed': bool, 'lookup_delete': bool, 'lookup_loc_changed': bool}
    """
    # Early return for empty text
    if not text:
        return {'processed_text': text, 'changed': False, 'lookup_delete': False, 'lookup_loc_changed': False}
    
    wrong_dblookup = []
    has_issues = False
    lookup_deleted = False
    lookup_loc_changed = False
    
    # Use pre-compiled pattern
    text_af = BLANK_PATTERN.sub(r'\1', text)
    results, _ = find_dblookup(text_af)
    
    if not results:
        return {'processed_text': text_af, 'changed': False, 'lookup_delete': False, 'lookup_loc_changed': False}

    # Process results more efficiently
    for result in results:
        dbquery = result['dbquery']
        entity = result['entity']
        start, end = result['start'], result['end']
        
        # Get following text more efficiently
        following_text = text_af[end:]
        substring = following_text[:len(entity)]
        
        # Use cached Levenshtein ratio for better performance (case-insensitive)
        if cached_levenshtein_ratio(entity.lower(), substring.lower()) < THRESHOLD:
            # Mask existing dbqueries to limit search space
            _results, following_text_masked = find_dblookup(following_text, mask_text=True)
            next_dbquery = following_text_masked.find(MASK_TOKEN)
            out_of_interest = ''
            if next_dbquery>0:
                following_text_masked, out_of_interest = following_text_masked[:next_dbquery],following_text_masked[next_dbquery:]
            
            # Check if we have entity after dblookup using fuzzy matching
            matched = find_match_idx(entity, following_text_masked, max_l_dist=DIST_K)
            if not matched:
                wrong_dblookup.append(dbquery)
                has_issues = True
                lookup_deleted = True
                continue
            
            # Reconstruct text more efficiently
            prepend_text = text_af[:start]
            text_af = (prepend_text + 
                      following_text_masked[:matched['start']] + 
                      dbquery + 
                      following_text_masked[matched['start']:] +
                      out_of_interest)
            
            # Restore masked dbqueries
            for _result in _results:
                text_af = text_af.replace(MASK_TOKEN, _result['dbquery'], 1)
            has_issues = True
            lookup_loc_changed = True

    # Remove wrong dblookups more efficiently
    if wrong_dblookup:
        for dbquery in wrong_dblookup:
            text_af = text_af.replace(dbquery, '', 1)
    
    return {'processed_text': text_af, 'changed': has_issues, 'lookup_delete': lookup_deleted, 'lookup_loc_changed': lookup_loc_changed}

    
def rollback_space(text: str) -> Dict[str, any]:
    text = re.sub(r'(\[dblookup[^\]]*? -> [^\]]*?\])(?=\S)', r'\1 ', text)
    return {'processed_text': text}

def cut_single_text(text: str) -> Dict[str, any]:
    """
    Truncate text at the first dblookup if it appears after 85% of the text length.
    
    Args:
        text (str): Text to process
        
    Returns:
        dict: {'processed_text': str, 'truncated': bool}
    """
    # Mask all dblookup patterns to find their positions
    _results, text_masked = find_dblookup(text, mask_text=True)
    
    # Configuration for truncation threshold
    cut_threshold = 0.85
    substring = "[dblookup"
    
    # Find all positions of dblookup patterns in the masked text
    positions = [m.start() for m in re.finditer(re.escape(substring), text_masked)]
    
    # Return original text if no dblookup patterns found
    if not positions:
        return {'processed_text': text, 'truncated': False}
    
    # Calculate the relative position of the first dblookup
    first_db_position = positions[0] / len(text_masked)
    
    # Truncate if the first dblookup appears after the threshold
    if first_db_position >= cut_threshold:
        # Cut text at the first dblookup position
        text_masked = text_masked[:positions[0]]
        
        # Restore original dblookup patterns in the truncated text
        for _result in _results:
            text_masked = text_masked.replace(MASK_TOKEN, _result['dbquery'], 1)
        
        return {'processed_text': text_masked, 'truncated': True}
    else:
        # Return original text if truncation threshold not met
        return {'processed_text': text, 'truncated': False}

def process_texts_optimized(dataset, num_proc: int = None, batch_size: int = 1000, target_func=process_single_text):
    """
    Optimized text processing function using Dataset.map.
    dataset.map supports batch processing and parallel processing by default.
    
    Args:
        dataset: HuggingFace Dataset to process
        num_proc: Number of parallel processing processes (uses CPU core count if None)
        batch_size: Batch size for processing
        target_func: Function to apply to each text (default: process_single_text)
        
    Returns:
        Dataset: Processed Dataset with results from target_func
    """
    def process_example(example):
        """
        Process a single example from the dataset.
        
        Args:
            example: Single example from the dataset
            
        Returns:
            dict: Result from applying target_func to the text
        """
        if 'processed_text' in example:
            text = example['processed_text']
        else:
            text = example['annotated_text']
        result = target_func(text)
        return result
    
    # Set default number of processes
    if num_proc is None:
        num_proc = min(mp.cpu_count(), 8)  # Limit to 8 processes max for stability
    
    # dataset.map supports batch processing and parallel processing by default
    processed_dataset = dataset.map(
        process_example,
        desc="Processing texts",
        num_proc=num_proc,
        batch_size=batch_size
    )
    
    return processed_dataset


def main(batch_size: int = 1000, num_proc: int = None):
    """
    Main execution function 
    
    Args:
        batch_size: Batch size
        num_proc: Number of parallel processing processes
    """
    print("Loading dataset...")
    ds = load_dataset(
        "kilian-group/LMLM-pretrain-dwiki6.1M",
        split="train",
    )
    
    print(f"Processing {len(ds)} texts...")
    
    # Step 1: Fix incorrect dblookups using process_single_text
    print("Step 1: Processing with process_single_text...")
    ds = process_texts_optimized(ds, num_proc=num_proc, batch_size=batch_size, target_func=process_single_text)
    
    # Step 2: Truncate texts at late dblookups using cut_single_text
    print("Step 2: Processing with cut_single_text...")
    ds = process_texts_optimized(ds, num_proc=num_proc, batch_size=batch_size, target_func=cut_single_text)
    
    # Step 3: Rollback space using rollback_space
    print("Step 3: Processing with rollback_space...")
    ds = process_texts_optimized(ds, num_proc=num_proc, batch_size=batch_size, target_func=rollback_space)
    
    # Save the final processed dataset
    ds.save_to_disk("processed_dataset")
    
    return ds


if __name__ == "__main__":
    # Process the dataset with optimized settings
    result = main(batch_size=1000, num_proc=4)