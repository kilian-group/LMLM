import os
import json
import yaml
from copy import deepcopy
from typing import Tuple, Dict

from datasets import load_dataset, Dataset

from lmlm.constants import PROMPTS_DIR, EVALSET_PATH
from lmlm.training.utils.utils_filter import (
    convert_to_raw_dataset,
    convert_to_special_db_tokens_format,
    filter_invalid_dblookups
)


############################################
# Datasets
############################################

def load_trainset(script_args) -> Tuple[Dataset, Dict[str, Dataset]]:
    """
    Load the dataset for supervised fine-tuning (SFT) and evaluation sets.
    
    Args:
        script_args: An object containing dataset-related arguments (e.g., dataset_name, dataset_config, dataset_train_split).
        
    Returns:
        Tuple[Dataset, Dict[str, Dataset]]: A tuple containing the training dataset and a dictionary of evaluation datasets.
    """
    if 'json' in script_args.dataset_name:
        dataset = load_dataset('json', data_files=script_args.dataset_name, field="examples")
    else:
        try:
            dataset = load_dataset(script_args.dataset_name, name=getattr(script_args, 'dataset_config', None))
        except:
            dataset = load_dataset("json", data_files=f"{script_args.dataset_name}/*.json", field="examples")
            # dataset = load_dataset(script_args.dataset_name, name=getattr(script_args, 'dataset_config', None), field="examples")
    
    train_dataset = dataset[getattr(script_args, 'dataset_train_split', 'train')]

    if 'annotated_text' not in train_dataset.column_names:
        if 'squad' in script_args.dataset_name:
            train_dataset = train_dataset.rename_column('context', 'annotated_text')
        elif 'dolmino' in script_args.dataset_name:
            train_dataset = train_dataset.rename_column('text', 'annotated_text')
        else:
            raise ValueError(f"Dataset {script_args.dataset_name} is not supported")

    return train_dataset


def load_evalsets(setting: str = 'LMLM_eval') -> Dataset:
    with open(EVALSET_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    if setting not in config:
        raise ValueError(f"Setting '{setting}' not found in YAML config.")
    
    eval_path_lst = config[setting]
    
    eval_dataset_dict = {}
    for key, eval_path in eval_path_lst.items():
        if os.path.exists(eval_path):  
            eval_dataset = load_dataset("json", data_files=eval_path, split="train", field="examples")
            eval_dataset_dict[key] = eval_dataset
        else:
            raise FileNotFoundError(f"Eval dataset not found at {eval_path}.")  
    return eval_dataset_dict


def prepare_evalsets(use_special_dblookup_tokens=False, is_plain_baseline=False) -> Tuple[Dataset, Dict[str, Dataset]]:
    eval_datasets = load_evalsets()

    eval_datasets = {key: eval_dataset.map(filter_invalid_dblookups) for key, eval_dataset in eval_datasets.items()}

    
    for sub_evalset_name, sub_evalset in list(eval_datasets.items()):
        if is_plain_baseline:
            # add convert_to_raw_dataset to evalset
            eval_datasets[sub_evalset_name + "_raw"] = convert_to_raw_dataset(sub_evalset)

        if use_special_dblookup_tokens and not is_plain_baseline:
            sub_evalset = convert_to_special_db_tokens_format(sub_evalset)
            eval_datasets[sub_evalset_name] = sub_evalset
    
    return eval_datasets

def prepare_trainset(script_args, use_special_dblookup_tokens=False, is_plain_baseline=False) -> Dataset:
    train_dataset = load_trainset(script_args)

    current_filter_version = 1 # Force reprocessing when you make changes
    train_dataset = train_dataset.map(lambda x: filter_invalid_dblookups(x, version=current_filter_version)) 
    
    if is_plain_baseline:
        train_dataset = convert_to_raw_dataset(train_dataset)
        
    if use_special_dblookup_tokens and not is_plain_baseline:
        train_dataset = convert_to_special_db_tokens_format(train_dataset)

    return train_dataset

def prepare_pretrain_data(script_args, use_special_dblookup_tokens=False, is_plain_baseline=False) -> Tuple[Dataset, Dict[str, Dataset]]:

    train_dataset = prepare_trainset(script_args, use_special_dblookup_tokens, is_plain_baseline)

    eval_datasets = prepare_evalsets(use_special_dblookup_tokens, is_plain_baseline)
    
    return train_dataset, eval_datasets

def prepare_instruction_tuning_data(script_args, tokenizer, use_prompt=True) -> Tuple[Dataset, Dataset]:
    """Loads and processes training & evaluation datasets for instruction tuning."""
    if 'json' in script_args.dataset_name:
        dataset = load_dataset('json', data_files=script_args.dataset_name, field="examples")
        train_dataset = dataset[getattr(script_args, 'dataset_train_split', 'train')]

    else:
        train_dataset = load_dataset(script_args.dataset_name, name=getattr(script_args, 'dataset_config', None))

    eval_dataset_name = getattr(script_args, 'eval_dataset_name')
    eval_dataset = load_dataset("json", data_files=eval_dataset_name, split="train", field="examples")

    # Load the instruction prompt (make configurable)
    if use_prompt:
        prompt_id = getattr(script_args, 'prompt_id', "llama-v6")
        prompt = InstructionPrompt(prompt_id)
    else:
        prompt = None

    # Process datasets
    train_dataset = train_dataset.map(lambda x: format_chat(x, tokenizer, prompt), batched=False)
    eval_dataset = eval_dataset.map(lambda x: format_chat(x, tokenizer, prompt), batched=False)

    return train_dataset, eval_dataset


def format_chat(data, tokenizer, prompt=None):
    """Formats the input text and annotation using the provided instruction prompt."""
    if prompt:
        full_text = prompt(data['text'], data['annotated_text'])
        formatted_text = tokenizer.apply_chat_template(full_text, tokenize=False, add_generation_prompt=False)
    else:
        full_text = "### Input:" + data['text'] + " ### Output:" + data['annotated_text']
        formatted_text = full_text

    return {
            "formatted_text": formatted_text,
        }


class InstructionPrompt:
    """Handles loading and formatting of instruction-based prompts."""
    
    def __init__(self, prompt_id):
        self.prompt_id = prompt_id
        path = os.path.join(PROMPTS_DIR, f"{prompt_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Prompt file not found at {path}.")
        
        with open(path, "r", encoding="utf-8") as f:
            self.prompt = json.load(f)

    def __call__(self, text, annotation):
        """
        Fills placeholders in the prompt with the provided text and annotation.

        Args:
            text (str): The text to insert into the prompt.
            annotation (str): The annotation to insert into the prompt.

        Returns:
            list: A list of dictionaries with placeholders replaced.
        """
        filled_prompt = deepcopy(self.prompt)

        for prompt_dict in filled_prompt:
            if "INSERT_TEXT" in prompt_dict['content']:
                prompt_dict['content'] = prompt_dict['content'].replace("[INSERT_TEXT]", text)
            if "INSERT_ANNOTATION" in prompt_dict['content']:
                prompt_dict['content'] = prompt_dict['content'].replace("[INSERT_ANNOTATION]", annotation)

        return filled_prompt