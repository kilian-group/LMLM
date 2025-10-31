import os
import json
import yaml
from copy import deepcopy
from typing import Tuple, Dict

from datasets import load_dataset, Dataset

from lmlm.constants import PROMPTS_DIR
from lmlm.training.utils.utils_filter import (
    convert_to_raw_dataset,
)


############################################
# Datasets
############################################

def prepare_pretrain_data(script_args, use_special_dblookup_tokens=False, is_plain_baseline=False) -> Tuple[Dataset, Dict[str, Dataset]]:

    assert use_special_dblookup_tokens == True, "Special dblookup tokens are required for pretraining. The plain version is deprecated."
    dataset = load_dataset(script_args.dataset_name, name=getattr(script_args, 'dataset_config', None))

    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation", None)
    test_dataset = dataset.get("test", None)


    if is_plain_baseline:
        train_dataset = convert_to_raw_dataset(train_dataset)
        eval_dataset = convert_to_raw_dataset(eval_dataset)
        test_dataset = convert_to_raw_dataset(test_dataset)
    
    eval_datasets = {
        "validation": eval_dataset,
        "test": test_dataset
    }
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