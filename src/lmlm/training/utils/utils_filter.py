import json
import os
import re
from lmlm.database.database_manager import extract_database

USE_SPECIAL_DBLOOKUP_TOKENS = True
def set_use_special_dblookup_tokens(use_special_dblookup_tokens=True):
    global USE_SPECIAL_DBLOOKUP_TOKENS
    USE_SPECIAL_DBLOOKUP_TOKENS = use_special_dblookup_tokens


def convert_dblookup_format(example):
    dblookup_pattern = re.compile(r"\[dblookup\('(.+?)',\s*'(.+?)'\) ->\s*(.+?)\]")
    
    example["annotated_text"] = dblookup_pattern.sub(
        r"<|db_entity|> \1<|db_relationship|> \2<|db_return|> \3<|db_end|>", 
        example["annotated_text"]
    )
    
    return example

################
# Data Cleaning
################
def clean_dataset(dataset):
    # extract 'atomic knowledge' from the dataset
    dataset = extract_database(dataset)

    dataset = dataset.map(filter_out_redundant_triplets_in_example)
    return dataset


def convert_to_raw_dataset(dataset):
    dataset = dataset.map(lambda example: filter_out_specified_triplets_in_example(example, triplets_to_keep=[]))
    print(f"Filtered dataset: {dataset} after filtering all lookup triplets")

    return dataset

def clean_high_loss_triplets(dataset, triplets_save_path):

    if not os.path.exists(triplets_save_path):
        raise FileNotFoundError(f"High-loss file {triplets_save_path} not found.")

    with open(triplets_save_path, "r") as f:
        high_loss_data = json.load(f)
    print(f"Loaded {len(high_loss_data)} high-loss triplets from {triplets_save_path}")

    dataset = dataset.map(lambda example: filter_out_specified_triplets_in_example(example, triplets_to_remove=high_loss_data))
    
    return dataset

def convert_to_special_db_tokens_format(dataset):
    dataset = dataset.map(lambda example: convert_dblookup_format(example))
    return dataset

################
# Utils
################
def normalize_triplet_text(triplet):
    """Normalize triplet values by stripping unwanted quotes and whitespace."""
    return [item.strip().strip('"').strip("'") for item in triplet]


def remove_unwanted_dblookups(text, triplets_to_keep=None, triplets_to_remove=None):
    """
    Remove [dblookup( )] patterns from the text that are not in the filtered triplets.

    Args:
        text (str): The input text containing [dblookup( )] patterns.
        triplets_to_keep (list of tuples): List of (entity, relationship, return_value) to keep.
        triplets_to_remove (list of tuples): List of (entity, relationship, return_value) to remove.

    Returns:
        str: The cleaned text with unwanted [dblookup( )] patterns removed.
    """
    # Ensure at least one filter list is provided
    assert triplets_to_keep is not None or triplets_to_remove is not None, \
        "Either triplets_to_keep or triplets_to_remove must be provided."

    # Normalize triplets to remove unnecessary quotes and spaces
    if triplets_to_keep:
        triplets_to_keep = [normalize_triplet_text(triplet) for triplet in triplets_to_keep]
    if triplets_to_remove:
        triplets_to_remove = [normalize_triplet_text(triplet) for triplet in triplets_to_remove]

    # Regex pattern to match dblookup statements
    if USE_SPECIAL_DBLOOKUP_TOKENS:
        pattern_lst = [r"\s?<\|db_entity\|>(.+?)<\|db_relationship\|>(.+?)<\|db_return\|>(.+?)<\|db_end\|>"]
        # pattern = r"\s?<\|db_entity\|>(.+?)<\|db_relationship\|>(.+?)<\|db_return\|>(.+?)<\|db_end\|>"
    else:
        pattern_lst = [r'\s?\[dblookup\(([^,]+),\s*([^,]+)\)\s*->\s*(.*?)\]',
                       r"\s?\[dblookup\('(.+?)',\s*'(.+?)'\) ->\s*(.+?)\]"]

    def replacement(match):
        # Extract triplet values and normalize them
        match_triplet = normalize_triplet_text(match.groups())

        # Check if the triplet should be kept or removed
        if (triplets_to_keep and match_triplet in triplets_to_keep) or \
           (triplets_to_remove and match_triplet not in triplets_to_remove):
            return match.group(0)  # Keep the dblookup pattern
        return ""  # Remove the dblookup pattern

    # Replace the dblookup patterns in the text using the regex pattern
    cleaned_text = text
    for pattern in pattern_lst:
        cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.DOTALL)
    
    if triplets_to_keep == []:
        # need to remove the incomplete dblookups
        cleaned_text = filter_incomplete_dblookups(cleaned_text)
    return cleaned_text


def extract_last_match(text):

    if USE_SPECIAL_DBLOOKUP_TOKENS:
        pattern = r"<\|db_entity\|>(.+?)<\|db_relationship\|>(.+?)<\|db_return\|>(.+?)<\|db_end\|>"
    else:
        pattern = r"\[dblookup\('(.+?)',\s*'(.+?)'\) ->\s*(.+?)\]"

    # Find the last match for each pattern
    last_matches = []

    matches = list(re.finditer(pattern, text, re.DOTALL))
    if matches:
        last_match = matches[-1]
        last_matches.append(last_match.groups())
    
    return [
        [item.strip("'").strip('"').strip() for item in match]
        for match in last_matches
    ]

################
# Data Filtering Functions
################
def filter_length(example, max_length=750, length_ratio=0.9):
    assert 'text' in example, "The example must contain a 'text' field."
    assert 'annotated_text' in example, "The example must contain an 'annotated_text' field."           

    if len(example['text'].split()) > max_length:
        return False
    
    if len(example['annotated_text'].split()) / len(example['text'].split()) < length_ratio:
        return False
    
    return True


def filter_redundant_triplets(triplets):
    filtered_triplets = []
    for triplet in triplets:
        entity, relationship, return_value = triplet
        if entity not in return_value and relationship not in return_value:
            filtered_triplets.append(triplet)
    return filtered_triplets    


def filter_high_low_triplet_density(example, max_triplets_per_100_tokens=12):

    text = example["annotated_text"]
    triplets = example["atomic_knowledge"]
    
    # Calculate the number of tokens (roughly)
    num_tokens = len(text.split())
    
    triplets_per_100_tokens = (len(triplets) / num_tokens) * 100 if num_tokens > 0 else 0

    if triplets_per_100_tokens > max_triplets_per_100_tokens:
        # print(f"Filtered out example with too many triplets per 100 tokens: {triplets_per_100_tokens}: {text}")
        return False          

    return triplets_per_100_tokens <= max_triplets_per_100_tokens and len(triplets) > 0

def filter_out_redundant_triplets_in_example(example):
    assert 'annotated_text' in example, "The example must contain an 'annotated_text' field."
    assert 'atomic_knowledge' in example, "The example must contain an 'atomic_knowledge' field."   

    filtered_triplets = filter_redundant_triplets(example['atomic_knowledge'])
    
    if len(filtered_triplets) < len(example["atomic_knowledge"]):
        # Remove unwanted [dblookup( )] patterns from the annotated text
        cleaned_text = remove_unwanted_dblookups(example["annotated_text"], triplets_to_keep=filtered_triplets)

        example["annotated_text"] = cleaned_text

    example["atomic_knowledge"] = filtered_triplets
    
    return example

def filter_out_specified_triplets_in_example(example, triplets_to_keep=None, triplets_to_remove=None):
    assert 'annotated_text' in example, "The example must contain an 'annotated_text' field."

    # Apply the `remove_unwanted_dblookups` function on the annotated_text
    example['annotated_text'] = remove_unwanted_dblookups(example['annotated_text'], triplets_to_keep=triplets_to_keep, triplets_to_remove=triplets_to_remove)
    
    if 'atomic_knowledge' in example:
        example['atomic_knowledge'] = [triplet for triplet in example['atomic_knowledge'] if (triplets_to_keep and triplet in triplets_to_keep) or (triplets_to_remove and triplet not in triplets_to_remove)]
    return example


def is_valid_dblookup(text):
    """
    Find all valid dblookup strings in the given text.
    
    Args:
        text (str): The text to scan for valid dblookup strings.
        
    Returns:
        List of valid dblookup strings.
    """
    # Updated regex pattern to allow spaces before '[' and after ']', and handle quotes for entity and relationship
    strict_pattern = re.compile(
        r"\[dblookup\(['\"](.+?)['\"],\s*['\"](.+?)['\"]\)\s*->\s*(.+?)\]"
    )

    matches = strict_pattern.findall(text)
    
    if not matches:
        return False
    
    return True


def filter_invalid_dblookups(example, version=1):
    """
    Filter out invalid dblookups from text.
    
    Args:
        example (dict): Input dictionary containing the annotated text
        
    Returns:
        dict: Updated example with invalid dblookups removed
    """
    text = example["annotated_text"]
    
    matches = loose_match(text)
    
    invalid_dblookups = []
    
    for match in matches:
        dblookup_str = match
        if not is_valid_dblookup(dblookup_str):
            invalid_dblookups.append(dblookup_str)
    
    for invalid_dblookup in invalid_dblookups:
        text = text.replace(invalid_dblookup, "")
        print(f"Filtered out invalid dblookup: {invalid_dblookup}") 
    
    incomplete_pattern = r'\[dblookup[^\]]*$'
    
    # Remove incomplete dblookups at the end of the text
    example["annotated_text"] = re.sub(incomplete_pattern, '', text)

    return example


def loose_match(dblookup_str):
    """
    Loose match to detect dblookup-like patterns.
    It can handle dblookups with some errors like missing values, incomplete relationships, etc.

    Args:
        dblookup_str (str): The dblookup string to check.

    Returns:
        list: List of matched dblookup calls.
    """
    # List of loose patterns that handle different variations of dblookup
    loose_pattern_lst = [
        # Pattern 1: dblookup with optional value and optional relationship
        re.compile(
            r"""\s*\[\s*dblookup\(\s*(?:(['\"])(.*?)\1)?(?:\s*,\s*(?:(['\"])(.*?)\3)?)?(?:\s*\))?\s*(?:->)?\s*(.*?)\s*\]"""
        ),
        # Pattern 2: dblookup with value, optional relationship
        re.compile(
            r"""\s*\[\s*dblookup\(\s*(['\"])(.*?)\1(?:\s*,\s*(.*?)\s*)?\)\s*(?:->\s*(.*?)\s*)?\]"""
        )
    ]
    
    # Find all matches for all patterns in the list
    matches = []
    for pattern in loose_pattern_lst:
        matches.extend([match.group(0) for match in pattern.finditer(dblookup_str)])

    return matches


def filter_incomplete_dblookups(example):
    """
    Remove both well-formed and incomplete dblookup patterns (e.g., [dblookup(...)] or <|db_entity|>...)
    that appear in the annotated text. Useful for cleaning examples before training or evaluation.

    Args:
        example (dict): A dictionary with "annotated_text" key.

    Returns:
        dict: The modified example with dblookup patterns removed.
    """
    if isinstance(example, dict) and "annotated_text" in example:
        text = example["annotated_text"]
    else:
        text = example

    # Step1: Remove complete dblookup formats
    if USE_SPECIAL_DBLOOKUP_TOKENS:
        pattern_lst = [
            r"\s?<\|db_entity\|>(.+?)<\|db_relationship\|>(.+?)<\|db_return\|>(.+?)<\|db_end\|>",
        ]
    else:
        pattern_lst = [
            r'\s?\[dblookup\(([^,]+),\s*([^,]+)\)\s*->\s*(.*?)\]',
            r"\s?\[dblookup\('(.+?)',\s*'(.+?)'\)\s*->\s*(.+?)\]"
        ]
    
    for pattern in pattern_lst:
        text = re.sub(pattern, '', text)
    
    ## Step2: Remove incomplete dblookups that are not closed in either beginning or end by appending <|db_entity|> or <|db_end|>
    starts = [m.start() for m in re.finditer(r"<\|db_entity\|>", text)]
    ends   = [m.start() for m in re.finditer(r"<\|db_end\|>", text)]

    if ends and not starts or starts and ends and starts[0] > ends[0]:
        # the case where the sentence starts with <|db_end|>
        text = "<|db_entity|>" + text
    if starts and not ends or starts and ends and starts[-1] > ends[-1]:
        # the case where the sentence ends with <|db_entity|>
        text = text + "<|db_end|>"

    ## Remove incomplete [dblookup(...] or <|db_entity|>... patterns at the end
    if USE_SPECIAL_DBLOOKUP_TOKENS:
        incomplete_patterns = [
            r'\s?<\|db_entity\|>(.*?)<\|db_end\|>'                      # Unclosed angle-bracket style
        ]
    else:
        incomplete_patterns = [
            r'\[dblookup[^\]]*$',                      # Unclosed square bracket style
        ]

    for incomplete in incomplete_patterns:
        text = re.sub(incomplete, '', text)

    if isinstance(example, dict) and "annotated_text" in example:
        example["annotated_text"] = text.strip()
    else:
        example = text.strip()
    return example
