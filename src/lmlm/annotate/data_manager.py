import json
import os
import pickle

from dataclasses import dataclass
from datasets import Dataset as HFDataset
from datetime import datetime
from lmlm.constants import DATA_DIR
from typing import Any, Dict, List, Optional


@dataclass
class Example:
    """Class to store a single example with its metadata."""
    annotated_text: str
    text: str
    model_id: str
    prompt_id: str

    original_dataset: str
    original_dataset_ids: List[str]

    processing_timestamp: str = None

    def __post_init__(self):
        if self.processing_timestamp is None:
            self.processing_timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert Example to dictionary format for HuggingFace Dataset."""
        return vars(self)

class DataManager:
    """Class to manage a collection of annotated examples."""

    def __init__(self):
        self.examples: List[Example] = []
        self.dataset_metadata = {
            'creation_date': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'source_datasets': set(),
            'models_used': set(),
            'prompts_used': set(),
        }

    @property
    def total_examples(self) -> int:
        """Number of examples in the dataset."""
        return len(self.examples)

    def add_example(self, example: Example) -> None:
        """Add a single example to the dataset."""
        self.examples.append(example)
        self.dataset_metadata['source_datasets'].add(example.original_dataset)
        self.dataset_metadata['models_used'].add(example.model_id)
        self.dataset_metadata['prompts_used'].add(example.prompt_id)
        self.dataset_metadata['last_modified'] = datetime.now().isoformat()

    def add_examples(self, examples: List[Example]) -> None:
        """Add multiple examples to the dataset."""
        for example in examples:
            self.add_example(example)

    def get_examples_by_source(self, source: str) -> List[Example]:
        """Retrieve all examples from a specific source dataset."""
        return [ex for ex in self.examples if ex.original_dataset == source]

    def get_examples_by_model(self, model_name: str) -> List[Example]:
        """Retrieve all examples annotated by a specific model."""
        return [ex for ex in self.examples if ex.model_id == model_name]

    def get_examples_by_prompt(self, prompt_id: str) -> List[Example]:
        """Retrieve all examples annotated with a specific prompt."""
        return [ex for ex in self.examples if ex.prompt_id == prompt_id]

    def to_huggingface_dataset(self) -> HFDataset:
        """Convert to HuggingFace Dataset format."""
        # Convert examples to list of dictionaries
        examples_dict = [ex.to_dict() for ex in self.examples]
        
        # Create HuggingFace Dataset
        return HFDataset.from_dict({
            'annotated_text': [d['annotated_text'] for d in examples_dict],
            'text': [d['text'] for d in examples_dict],
        })

    def save(self, data_identifier: str, format: str = 'pickle', save_dir: str = DATA_DIR) -> None:
        """
        Save the dataset to disk.

        Args:
            path: Path to save the dataset
            format: Either 'pickle' or 'json'
        """
        extension = '.pkl' if format == 'pickle' else '.json'
        path = os.path.join(save_dir, f"{data_identifier}{extension}")

        # Convert sets to lists for serialization
        metadata = self.dataset_metadata.copy()
        metadata['source_datasets'] = list(metadata['source_datasets'])
        metadata['models_used'] = list(metadata['models_used'])
        metadata['prompts_used'] = list(metadata['prompts_used'])

        if format == 'pickle':
            with open(path, 'wb') as f:
                pickle.dump({
                    'examples': self.examples,
                    'metadata': metadata
                }, f)

        elif format == 'json':
            examples_dict = [ex.to_dict() for ex in self.examples]
            with open(path, 'w') as f:
                json.dump({
                    'examples': examples_dict,
                    'metadata': metadata
                }, f, indent=2)

    @classmethod
    def load(cls, data_identifier: str, format: str = 'pickle', save_dir: str = DATA_DIR) -> 'DataManager':
        """
        Load a dataset from disk.
        
        Args:
            path: Path to load the dataset from
            format: Either 'pickle' or 'json'
        """
        extension = '.pkl' if format == 'pickle' else '.json'
        path = os.path.join(save_dir, f"{data_identifier}{extension}")
        manager = cls()

        if format == 'pickle':
            with open(path, 'rb') as f:
                data = pickle.load(f)
                manager.examples = data['examples']
                manager.dataset_metadata = data['metadata']

        elif format == 'json':
            with open(path, 'r') as f:
                data = json.load(f)
                manager.examples = [Example(**ex) for ex in data['examples']]
                if 'metadata' in data:
                    manager.dataset_metadata = data['metadata']
                else:
                    manager.dataset_metadata = {
                        'creation_date': datetime.now().isoformat(),
                        'last_modified': datetime.now().isoformat(),
                        'source_datasets': set(),
                        'models_used': set(),
                        'prompts_used': set(),
                    }   

        # Convert lists back to sets in metadata
        manager.dataset_metadata['source_datasets'] = set(manager.dataset_metadata['source_datasets'])
        manager.dataset_metadata['models_used'] = set(manager.dataset_metadata['models_used'])
        manager.dataset_metadata['prompts_used'] = set(manager.dataset_metadata['prompts_used'])

        return manager