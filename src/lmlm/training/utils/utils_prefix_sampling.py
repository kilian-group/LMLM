#!/usr/bin/env python3
import os
import sys
import pickle
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, LogitsProcessorList, LogitsProcessor
import marisa_trie
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.nn import CrossEntropyLoss


from lmlm.database.database_manager import DatabaseManager, DatabaseLookupError
from lmlm.training.utils.utils_metrics import indices_to_mask, extract_dblookup_indices, validate_extraction, validate_mask_tokens
from lmlm.constants import DB_START_TOKEN, DB_SEP_TOKEN, DB_RETRIEVE_TOKEN, DB_END_TOKEN

import re
import math

########################################
# Utility functions
########################################

# We use a delimiter to convert lists of token IDs to a unique string.
DELIM = "|"

def tokens_to_key(tokens):
    """Convert a list of token IDs into a string key."""
    if not tokens:
        return "" # Return empty string if no tokens
    return DELIM.join(map(str, tokens)) + DELIM

def key_to_tokens(key):
    """Convert a string key back into a list of token IDs."""
    if not key:
        return []  # Special case for empty string
    # Remove the trailing delimiter before splitting
    return [int(x) for x in key.rstrip(DELIM).split(DELIM)]

########################################
# Database Triplets to Query Keys
########################################

def process_triplets_to_keys(triplets, tokenizer, special_relation = "<|db_relationship|>"):
    """
    Given triplets of the form (entity, relationship, value), return a list
    of query keys in the form: "entity<|db_relationship|>value".
    """
    # The special token "<|db_relationship|>" is assumed to be in the tokenizer.
    keys = []
    for entity, relationship, value in tqdm(triplets, desc="Processing triplets"):
        entity = entity.strip()
        relationship = relationship.strip()
        value = value.strip()

        # Create a key string using the special token as a delimiter.
        key_text = f" {entity}{special_relation} {relationship}"
        token_ids = tokenizer.encode(key_text, add_special_tokens=False)
        key_str = tokens_to_key(token_ids)
        keys.append(key_str)
    return keys

########################################
# Build Prefix Tree and Compute Statistics
########################################

def build_prefix_tree(keys):
    """Build a marisa‑trie from a list of string keys."""
    trie = marisa_trie.Trie(keys)
    return trie

def plot_depth_distribution(depth_distribution, filename="prefix_tree_depths.png"):
    """
    Given a depth distribution dictionary mapping depth to count,
    plot a histogram of the depths and save it to a file.
    """
    # Expand the dictionary into a list of depth values.
    depth_list = []
    for depth, count in depth_distribution.items():
        depth_list.extend([depth] * count)
        
    plt.figure(figsize=(8, 4))
    # Create histogram with appropriate bins.
    # bins = range(min(depth_list), max(depth_list) + 2)  # one bin per depth value
    bins = 30
    plt.hist(depth_list, bins=bins, edgecolor='black', align='left', color='skyblue')
    plt.xlabel("Depth")
    plt.ylabel("Frequency")
    plt.title("Prefix Tree Depth Distribution")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def compute_trie_stats(trie):
    """
    Compute and return statistics about the prefix tree:
      - raw size in bytes and MB (using pickle)
      - maximum depth (max number of tokens in any key)
      - depth distribution (count of keys for each depth)
    """
    # Get raw size via pickle
    pickled = pickle.dumps(trie)
    size_bytes = len(pickled)
    size_mb = size_bytes / (1024 * 1024)
    # Compute depths for each key.
    depths = []
    for key in tqdm(trie.keys(), desc="Computing depths"):
        token_ids = key_to_tokens(key)
        depths.append(len(token_ids))
    max_depth = max(depths) if depths else 0
    depth_distribution = dict(Counter(depths))
    return size_bytes, size_mb, max_depth, depth_distribution

########################################
# Custom Logits Processor for Prefix Tree Sampling (Phase 2)
########################################

class PrefixTreeLogitsProcessor(LogitsProcessor):
    """
    This logits processor enforces that generation follows valid query keys
    (as defined by the prefix tree).
    
    - If the current prefix (extracted from input_ids) exactly matches one of the stored keys,
      then only the special token <db_return> is allowed.
    - Otherwise, it restricts valid next tokens to those given by get_valid_next_tokens.
    """
    def __init__(self, db_return_token_id, eos_token_id, trie, tokenizer, verbose=False):
        self.eos_token_id = eos_token_id
        self.db_return_token_id = db_return_token_id
        self.trie = trie
        self.tokenizer = tokenizer
        self.prompts = None
        self.sampled_entries = set()
        self.verbose = verbose
    
    def get_valid_next_tokens(self, prefix_tokens):
        """
        Given a prefix (list of token IDs), return the set of valid next tokens –
        the token immediately following the prefix in any stored key.
        """
        prefix_key = tokens_to_key(prefix_tokens)
        next_tokens = set()

        for key in self.trie.keys(prefix=prefix_key):
            token_ids = key_to_tokens(key)
            if len(token_ids) > len(prefix_tokens):
                # Exclude candidate tokens that would lead to a lookup already sampled.
                if key in self.sampled_entries:
                    continue
                next_tokens.add(token_ids[len(prefix_tokens)])
        return next_tokens

    def print_sampled_entries(self):
        """
        Print the sampled entries for debugging.
        """
        print("Sampled entries:")
        for entry in self.sampled_entries:
            entry_token_ids = key_to_tokens(entry)
            entry_str = self.tokenizer.decode(entry_token_ids, skip_special_tokens=False)
            print(f"  - {entry_str}")
        print(f"Total sampled entries: {len(self.sampled_entries)}")
    
    def add_sampled_entry(self, prefix_tokens):
        """
        Add a prefix to the set of sampled entries.
        This is used to prevent re-sampling the same prefix in future
        generations.
        """
        prefix_key = tokens_to_key(prefix_tokens)
        assert prefix_key not in self.sampled_entries, f"Entry '{prefix_key}' already sampled."
        assert prefix_key in self.trie, f"Entry '{self.tokenizer.decode(prefix_tokens, skip_special_tokens=False)}' not in trie."
        self.sampled_entries.add(prefix_key)

    def reset(self):
        """
        Reset the processor state for a new generation.
        Call this method between different generations.
        """
        self.prompts = None
        self.sampled_entries = set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.verbose:
            print(f"Calling PrefixTreeLogitsProcessor with partial sequence '{self.tokenizer.decode(input_ids[0], skip_special_tokens=False)}'")
        
        if self.prompts is None:
            self.prompts = input_ids

        # Create a tensor for negative infinity with the same dtype and device as scores.
        batch_size = input_ids.shape[0]
        scores_processed = torch.full_like(scores, -1e9)

        for i in range(batch_size):
            prefix = input_ids[i].tolist()[len(self.prompts[i]):]
            # print(f"Processing beam {i} with prefix: {tokenizer.decode(prefix)}"
            prefix_key = tokens_to_key(prefix)

            valid_tokens = None
            # If the current prefix exactly matches a valid query key, force EOS.
            if prefix_key in self.trie and prefix_key not in self.sampled_entries:
                # print(f"Found exact match: '{tokenizer.decode(prefix, skip_special_tokens=False)}'")
                valid_tokens = {self.db_return_token_id}
            else:
                # print(f"No exact match found for: '{tokenizer.decode(prefix, skip_special_tokens=False)}'")
                valid_tokens = self.get_valid_next_tokens(prefix)
                # print(f"Valid tokens: {[self.tokenizer.decode([t], skip_special_tokens=False) for t in valid_tokens]}")

            # If no valid tokens are returned, we set a default valid token (e.g. EOS) to avoid all -inf.
            if not valid_tokens:
                valid_tokens = {self.eos_token_id}
            
            valid_token_list = list(valid_tokens)
            # Create a boolean mask: True for valid token indices.
            mask = torch.zeros_like(scores_processed[i], dtype=torch.bool)
            mask[valid_token_list] = True
            scores_processed[i, mask] = scores[i, mask]

        normalized_scores = torch.log_softmax(scores_processed, dim=-1)

        if self.verbose:
            for i in range(batch_size):
                top_10_valid_token_scores, top_10_valid_token_indices = torch.topk(normalized_scores[i], 10, largest=True)
                print(f"Top 10 most likely valid tokens = {[(self.tokenizer.decode([top_10_valid_token_indices[i]], skip_special_tokens=False), top_10_valid_token_scores[i].item()) for i in range(len(top_10_valid_token_indices))]}")

        return normalized_scores

def load_triplets_to_dict(triplets):
    """
    Load a JSON file of triplets (entity, relation, value) and build
    a dictionary mapping (entity, relation) tuples to the corresponding value.
    If there are multiple values per key, you can store them as a list.
    """
    db = {}
    for triplet in triplets:
        entity, relationship, value = triplet
        entity = entity.strip().lower()
        relationship = relationship.strip().lower()
        value = value

        key = (entity, relationship)
        # If there might be multiple values for the same key, store them in a list.
        if key in db:
            if isinstance(db[key], list):
                db[key].append(value)
            else:
                db[key] = [db[key], value]
        else:
            db[key] = value
    return db

def query_db(db, entity, relation):
    """
    Retrieve the value(s) corresponding to (entity, relation) from the database dictionary.
    Returns None if the key is not found.
    """
    return db.get((entity, relation))

def retrieve_from_database(db, prompt):
    '''
    Retrieve data from the database using a dblookup pattern in the prompt.
    '''

    # if USE_SPECIAL_DBLOOKUP_TOKENS:
        # pattern_lst = [r"<\|db_entity\|>(.+?)<\|db_relationship\|>(.+?)<\|db_return\|>"]
    # else:
    pattern_lst = [
        r"\[dblookup\('((?:[^'\\]|\\.)+)',\s*'((?:[^'\\]|\\.)+)'\)\s*->",
        r"\[dblookup\('(.+?)',\s*'(.+?)'\)\s*->",
        r"<\|db_entity\|>\s*(.+?)\s*<\|db_relationship\|>\s*(.+?)\s*<\|db_return\|>"
    ]
        # pattern_lst = [r'\[dblookup\(([^,]+),\s*([^,]+)\)\s*->',
        #             r"\[dblookup\('(.+?)',\s*'(.+?)'\) ->"]

    matches = {tuple(match) for pattern in pattern_lst for match in re.findall(pattern, prompt)}

    if not matches:
        raise DatabaseLookupError(f"dblookup_fail_5: No valid dblookup pattern found in prompt: {prompt}", "no_match_found")

    if len(matches) > 1:
        raise DatabaseLookupError(f"dblookup_fail_1: Multiple matches found: {matches} from {prompt}", "multiple_matches")

    entity_raw, relationship_raw = matches.pop()

    entity = entity_raw.strip().lower()
    relationship = relationship_raw.strip().lower()
    
    results = query_db(db, entity, relationship)

    # print(results)
    
    if not results:
        print(f"dblookup_fail_3: Retrieval failed for entity '{entity}' and relationship '{relationship}' no_retrieval_data_found")

    return results, entity_raw, relationship_raw


def compute_dblookup_mask(shift_labels, tokenizer):    
    indices = extract_dblookup_indices(shift_labels)
    print(f"Entity token '{tokenizer.decode(shift_labels[indices[0]:indices[0]+1])}'")
    print(f"Relationship token '{tokenizer.decode(shift_labels[indices[1]:indices[1]+1])}'")
    print(f"Return token '{tokenizer.decode(shift_labels[indices[2]:indices[2]+1])}'")
    print(f"Dblookup '{tokenizer.decode(shift_labels[indices[0]:indices[2]+1])}'")

    validate_extraction(shift_labels[1:2], indices[1:2])
    # import pdb; pdb.set_trace()
    
    mask_batch = indices_to_mask(shift_labels.shape[1], indices, pretrain_mask_only=True)
    # masked_tokens = validate_mask_tokens(mask_batch, shift_labels) 
    # import pdb; pdb.set_trace()
    if not torch.all((mask_batch["entity"] + mask_batch["relationship"] == torch.logical_or(mask_batch["entity"], mask_batch["relationship"]))*1):
        print("Entity and Relationship overlap")
        print(mask_batch["entity"] + mask_batch["relationship"])
        print(torch.logical_or(mask_batch["entity"], mask_batch["relationship"])*1)
    return mask_batch["entity"] + mask_batch["relationship"]

def compute_loss_func(outputs, labels, num_items_in_batch, tokenizer):

    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()  # Exclude the last token prediction
    shift_labels = labels[..., 1:].contiguous()  # Exclude the first token label
    
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # [batch_size * (sequence_length - 1), vocab_size]
    shift_labels = shift_labels.view(-1)  # [batch_size * (sequence_length - 1)]

    dblookup_mask = compute_dblookup_mask(shift_labels, tokenizer)

    loss_fct = CrossEntropyLoss(reduction='none')
    per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(labels.size(0), -1)
    weighted_loss = per_token_loss[dblookup_mask != 0]

    if num_items_in_batch is None:
        weighted_loss = weighted_loss.mean()
    else:
        weighted_loss = weighted_loss.sum()
        weighted_loss = weighted_loss / num_items_in_batch    
    return weighted_loss


def compute_conditional_log_likelihood(model, context_ids, target_ids, tokenizer):
    """
    Compute the log-likelihood of each target sequence conditioned on the context sequence.
    
    Args:
        model: A causal language model in evaluation mode.
        context_ids: List of token IDs representing the context (Phase 1 output).
        target_ids: Tensor of shape (n, T) representing n target sequences (padded with tokenizer.eos_token_id).
        tokenizer: The tokenizer corresponding to the model.
        
    Returns:
        total_log_likelihoods (torch.Tensor): Tensor of shape (n,) containing the sum of log 
            probabilities for each target sequence (ignoring padded tokens).
        per_token_log_probs (torch.Tensor): Tensor of shape (n, T) containing log probabilities for each
            target token.
    """
    device = model.device
    n, T = target_ids.shape
    C = len(context_ids)

    # Convert the context to a tensor and expand it to match the batch size.
    context_tensor = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)  # shape: (1, C)
    context_tensor = context_tensor.expand(n, -1)  # shape: (n, C)

    # Concatenate context with each target sequence.
    # full_ids will have shape (n, C + T)
    full_ids = torch.cat([context_tensor, target_ids], dim=1)

    # Prepare inputs and labels for the causal model.
    # input_ids: full_ids excluding the last token.
    # labels: full_ids excluding the first token.
    input_ids = full_ids[:, :-1]  # shape: (n, C+T-1)
    labels = full_ids[:, 1:]      # shape: (n, C+T-1)

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        logits = outputs.logits  # shape: (n, C+T-1, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1)  # shape: (n, C+T-1, vocab_size)

    # The target tokens in labels start at position C-1.
    # For example, if context = [c0, c1, ..., c_{C-1}], then the labels are:
    # [c1, c2, ..., c_{C-1}, target0, target1, ...]
    target_start_idx = C - 1

    # Extract log probabilities corresponding to target predictions.
    # This will be of shape (n, T, vocab_size)
    target_log_probs = log_probs[:, target_start_idx:, :]

    # Gather the log probabilities for the true target token IDs.
    # target_ids shape: (n, T) is unsqueezed to (n, T, 1) so that we gather along the vocab dimension.
    per_token_log_probs = target_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)  # shape: (n, T)

    # Create a mask to zero-out padded tokens (those equal to tokenizer.eos_token_id)
    valid_mask = (target_ids != tokenizer.eos_token_id).float()  # shape: (n, T)

    # Sum the log probabilities for valid (non-padding) tokens for each sequence.
    total_log_likelihoods = (per_token_log_probs * valid_mask).mean(dim=-1)

    return total_log_likelihoods

########################################
# MultiPhaseGenerator Class
########################################

class MultiPhaseGenerator:
    def __init__(self, model, tokenizer, prefix_processor, triplets, db_entity_token_id, db_relationship_token_id, db_return_token_id, db_end_token_id,
                 phase1_config, phase2_config, threshold=-0.9):
        self.model = model
        self.tokenizer = tokenizer
        self.logits_processor = LogitsProcessorList([prefix_processor])
        self.db = load_triplets_to_dict(triplets)
        
        self.db_entity_token_id = db_entity_token_id
        self.db_relationship_token_id = db_relationship_token_id
        self.db_return_token_id = db_return_token_id
        self.db_end_token_id = db_end_token_id

        self.phase1_config = phase1_config
        self.phase2_config = phase2_config

        self.threshold = threshold
        
        self.scores_list = []

    def reset_logits_processor(self):
        """
        Reset the logits processor for a new generation.
        Call this method between different generations.
        """
        for i in range(len(self.logits_processor)):
            self.logits_processor[i].reset()

    def generate_phase1(self, current_ids, return_combined=False):
        input_tensor = torch.tensor([current_ids]).to(self.model.device)
        attention_mask = torch.ones_like(input_tensor).to(self.model.device)
        phase1_output = self.model.generate(
            input_tensor,
            attention_mask=attention_mask,
            generation_config=self.phase1_config,
            tokenizer=self.tokenizer
        )

        # Remove the portion corresponding to current_ids (the prompt for this phase)
        combined_ids = phase1_output[0].tolist()
        phase1_token_ids = combined_ids[len(current_ids):] 
        
        if return_combined:
            return phase1_token_ids, combined_ids
        else:
            return phase1_token_ids

    
    def generate_phase2(self, current_ids, remove_bad_lookups=False):
        # Use the entire current_ids as the prompt.
        phase2_tensor = torch.tensor([current_ids]).to(self.model.device)
        phase2_attention_mask = torch.ones_like(phase2_tensor).to(self.model.device)
        phase2_output = self.model.generate(
            phase2_tensor,
            attention_mask=phase2_attention_mask,
            generation_config=self.phase2_config,
            logits_processor=self.logits_processor,
            tokenizer=self.tokenizer,
            return_dict_in_generate=True,
            output_scores=True
        )

        phase2_output_token_ids = phase2_output.sequences[:, len(current_ids):]
        phase2_log_probs = compute_conditional_log_likelihood(self.model, current_ids, phase2_output_token_ids, self.tokenizer)
        sorted_indices = torch.argsort(phase2_log_probs, descending=True)
        phase2_output_token_ids = phase2_output_token_ids[sorted_indices]
        phase2_log_prob = phase2_log_probs[0].item()

        filtered_sequences = [seq[seq != self.tokenizer.eos_token_id].tolist() for seq in phase2_output_token_ids]
        phase2_token_ids = filtered_sequences[0]

        bad_dblookup = False
        if phase2_log_prob < self.threshold:
            bad_dblookup = True
            for i in range(len(filtered_sequences)):
                print(f"Beam {i}: {self.tokenizer.decode(filtered_sequences[i], skip_special_tokens=False)}")
            # We picked a bad dblookup. Do normal generation without logits processor
            print(f"Bad dblookup: {self.tokenizer.decode(phase2_token_ids, skip_special_tokens=False)}, log_prob = {phase2_log_prob}")

            if remove_bad_lookups:
                # Reset the logits processor for the next text generation.
                for i in range(len(self.logits_processor)):
                    self.logits_processor[i].prompts = None

                return []
            
            # We picked a bad dblookup. Do normal generation without logits processor
            phase2_output = self.model.generate(
                phase2_tensor,
                attention_mask=phase2_attention_mask,
                generation_config=self.phase2_config,
                logits_processor=None,
                tokenizer=self.tokenizer,
                return_dict_in_generate=True,
                output_scores=True
            )

            filtered_sequences = [seq[seq != self.tokenizer.eos_token_id].tolist() for seq in phase2_output.sequences]
            phase2_token_ids = filtered_sequences[0][len(current_ids):]

        if len(phase2_token_ids) == 0:
            for i in range(len(filtered_sequences)):
                print(f"Beam {i}: {self.tokenizer.decode(filtered_sequences[i], skip_special_tokens=False)}")
            import pdb; pdb.set_trace()
            raise ValueError(f"Phase 2 produced an empty sequence. Make sure to reset the logits processor for the next generation.")

        # Verify that Phase 2 ended with the DB return token.
        if phase2_token_ids[-1] != self.db_return_token_id:
            import pdb; pdb.set_trace()
            raise ValueError(f"Phase 2 did not end with the DB return token. Instead, ended with: {self.tokenizer.decode(phase2_token_ids, skip_special_tokens=False)}")
        
        # Remove the final return token from the valid query key.
        query_key_ids = [self.db_entity_token_id] + phase2_token_ids
        query_key = self.tokenizer.decode(query_key_ids, skip_special_tokens=False)

        db_results, entity_text, relationship_text = retrieve_from_database(self.db, query_key)
        if db_results:
            try:
                if not isinstance(db_results, str):
                    db_results = str(db_results) 
                db_results_ids = self.tokenizer.encode(db_results, add_special_tokens=False)
            except Exception as e:
                print(f"Error encoding db_results: {e}")
                import pdb; pdb.set_trace()

        else:
            # db_results_ids = self.tokenizer.encode(f"Unknown", add_special_tokens=False)
            db_results_ids = self.tokenizer.encode(f"unknown", add_special_tokens=False)
        
        # Reset the logits processor for the next text generation.
        for i in range(len(self.logits_processor)):
            self.logits_processor[i].prompts = None
            # Mark this lookup as sampled so that it will not be allowed next time.
            if not bad_dblookup:
                self.logits_processor[i].add_sampled_entry(phase2_token_ids[:-1])

        return phase2_token_ids + db_results_ids + [self.db_end_token_id]

    def generate(self, prompt, verbose=False):
        # Phase 1: Normal generation until stop token is reached.
        # Use return_tensors="pt" so that we get a tensor.
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        current_ids = encoded.input_ids[0].tolist()
        
        MAX_LOOP_COUNT = 10
        loop_count = 0

        self.scores_list = []

        # Loop over the two-phase cycle until an EOS token is produced.
        while True:
            if loop_count >= MAX_LOOP_COUNT:
                break
            
            if verbose:
                print("\n##################################################")
                print(f"                  Loop {loop_count + 1} ")
                print("##################################################")

            ########################################
            # Phase 1: Normal generation.
            ########################################
            phase1_token_ids = self.generate_phase1(current_ids, return_combined=False)
            current_ids += phase1_token_ids

            if verbose:
                print(f"* After Phase 1 text: '{self.tokenizer.decode(current_ids, skip_special_tokens=False)}'")
            
            last_token = current_ids[-1]
            # If the last token is EOS, we're done.
            if last_token == self.tokenizer.eos_token_id:
                break
            
            # If the last token is the designated DB entry token, do Phase 2.
            if last_token == self.db_entity_token_id:
                ########################################
                # Phase 2: Constrained generation.
                ########################################
                phase2_token_ids = self.generate_phase2(current_ids)
                current_ids += phase2_token_ids
                print(f"After Phase 2 text: {self.tokenizer.decode(current_ids, skip_special_tokens=False)}")
            else:
                raise ValueError(f"Unexpected token at end of Phase 1: {self.tokenizer.decode([last_token], skip_special_tokens=False)}")

            loop_count += 1

        print(f"scores_list:\n {self.scores_list}")
        final_text = self.tokenizer.decode(current_ids, skip_special_tokens=False)
        return final_text

def initialize_prefix_tree(model_name, tokenizer, database_path, threshold=-0.9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    db_entity_token_id = tokenizer.convert_tokens_to_ids(DB_START_TOKEN)
    db_relationship_token_id = tokenizer.convert_tokens_to_ids(DB_SEP_TOKEN)
    db_return_token_id = tokenizer.convert_tokens_to_ids(DB_RETRIEVE_TOKEN)
    db_end_token_id = tokenizer.convert_tokens_to_ids(DB_END_TOKEN)

    ########################################
    # Process Database Triplets to Build Query Keys
    ########################################
    db_manager = DatabaseManager()
    db_manager.load_database(database_path)
    print(f"Loaded database {db_manager}")

    # Process triplets to generate keys: "entity<|db_relationship|>value"
    triplets = list(db_manager.database["triplets"])
    keys = process_triplets_to_keys(triplets, tokenizer)
    
    ########################################
    # Build the Prefix Tree and Compute Statistics
    ########################################
    prefix_tree = build_prefix_tree(keys)
    size_bytes, size_mb, max_depth, depth_distribution = compute_trie_stats(prefix_tree)

    print("Prefix Tree Statistics:")
    print(f"  Raw size: {size_bytes} bytes ({size_mb:.2f} MB)")
    print(f"  Max depth: {max_depth}")
    # print(f"  Depth distribution saved to {depth_plot_filename}")

    # Define sampling parameters for each phase.
    # logit_bias = {entity_token_id: 10.0, relationship_token_id: 2.0, return_token_id: 2.0, end_token_id: 2.0}
    logit_bias = {db_entity_token_id: 10.0}
    renormalize_logits = True
    sequence_bias = 0.0

    max_total_tokens = 128

    phase1_config = GenerationConfig(
        max_new_tokens=max_total_tokens,
        temperature=1.0,
        # top_p=0.9,
        top_k=10,
        repetition_penalty=1.2,
        eos_token_id=[db_entity_token_id, tokenizer.eos_token_id],
        num_beams=2,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    phase2_config = GenerationConfig(
        max_new_tokens=max_depth+1,
        temperature=1.0,
        # top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        eos_token_id=[db_return_token_id],
        num_beams=10,
        renormalize_logits=True,
        do_sample=True,
        num_return_sequences=10,
        pad_token_id=tokenizer.pad_token_id,
    )

    prefix_processor = PrefixTreeLogitsProcessor(db_return_token_id, tokenizer.eos_token_id, prefix_tree, tokenizer)

    # Create the multi-phase generator.
    generator = MultiPhaseGenerator(model, tokenizer, prefix_processor, triplets, db_entity_token_id, db_relationship_token_id, db_return_token_id, db_end_token_id, phase1_config, phase2_config, threshold=threshold)

    return generator

########################################
# Main function
########################################

def main():
    ########################################
    # Setup: Load model, tokenizer, and special tokens
    ########################################
    model_name = "/path/to/version4/model/tiny-llama2-176M_dwiki6.1M_ep8_bsz256_new"
    database_path = "/path/to/version4/database/dwiki_bio17k-annotator_database.json"
    # database_path = "/path/to/version4/database/dwiki6.1M_database.json"

    # # Create vLLM instance (vLLM loads model & tokenizer by name)
    # llm = LLM(
    #     model=model_name,
    #     tensor_parallel_size=1,
    #     max_model_len=1024,
    #     gpu_memory_utilization=0.85,
    #     dtype=torch.bfloat16,
    #     seed=42,
    #     tokenizer=model_name
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_name)


    # Load Hugging Face model and tokenizer.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    db_entity_token_id = tokenizer.convert_tokens_to_ids(DB_START_TOKEN)
    db_relationship_token_id = tokenizer.convert_tokens_to_ids(DB_SEP_TOKEN)
    db_return_token_id = tokenizer.convert_tokens_to_ids(DB_RETRIEVE_TOKEN)
    db_end_token_id = tokenizer.convert_tokens_to_ids(DB_END_TOKEN)

    ########################################
    # Process Database Triplets to Build Query Keys
    ########################################
    db_manager = DatabaseManager()
    db_manager.load_database(database_path)
    print(f"Loaded database {db_manager}")

    # Process triplets to generate keys: "entity<|db_relationship|>value"
    triplets = list(db_manager.database["triplets"])
    keys = process_triplets_to_keys(triplets, tokenizer)
    
    ########################################
    # Build the Prefix Tree and Compute Statistics
    ########################################
    prefix_tree = build_prefix_tree(keys)
    size_bytes, size_mb, max_depth, depth_distribution = compute_trie_stats(prefix_tree)
    depth_plot_filename = "prefix_tree_depths.png"
    plot_depth_distribution(depth_distribution, filename=depth_plot_filename)

    print("Prefix Tree Statistics:")
    print(f"  Raw size: {size_bytes} bytes ({size_mb:.2f} MB)")
    print(f"  Max depth: {max_depth}")
    print(f"  Depth distribution saved to {depth_plot_filename}")

    # Define sampling parameters for each phase.
    # logit_bias = {entity_token_id: 10.0, relationship_token_id: 2.0, return_token_id: 2.0, end_token_id: 2.0}
    logit_bias = {db_entity_token_id: 10.0}
    renormalize_logits = True
    sequence_bias = 0.0

    max_total_tokens = 128

    # Create GenerationConfig objects for each phase.
    # phase1_config = GenerationConfig(
    #     max_new_tokens=max_total_tokens,
    #     temperature=1.0,
    #     # top_p=0.9,
    #     top_k=10,
    #     repetition_penalty=1.2,
    #     eos_token_id=[db_entity_token_id, tokenizer.eos_token_id],
    #     num_beams=2,
    #     do_sample=True,
    #     pad_token_id=tokenizer.pad_token_id,
    # )

    phase2_config = GenerationConfig(
        max_new_tokens=max_depth+1,
        temperature=1.0,
        # top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        # stop_strings=["<|db_return|>"],
        # eos_token_id=[return_token_id, tokenizer.eos_token_id],
        eos_token_id=[db_return_token_id],
        num_beams=10,
        renormalize_logits=True,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    prefix_processor = PrefixTreeLogitsProcessor(db_return_token_id, tokenizer.eos_token_id, prefix_tree, tokenizer)

    # Create the multi-phase generator.
    generator = MultiPhaseGenerator(model, tokenizer, prefix_processor, triplets, db_entity_token_id, db_relationship_token_id, db_return_token_id, db_end_token_id, phase1_config, phase2_config)

    # Generate with a prompt.
    prompt = ["Tell me a brief bio of Kang Ji-hwan. Kang Ji-hwan is a "]
    final_output = generator.generate(prompt)
    print("\nFinal output:")
    print(final_output)

if __name__ == "__main__":
    main()