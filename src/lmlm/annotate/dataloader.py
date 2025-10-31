"""Core data preparation functions for LMLM training."""
from datasets import load_dataset
from .utils import truncate_sample_length, add_shared_context_ids
from typing import List, Tuple, Callable, Dict

# Global registry for dataset loaders
_DATASET_REGISTRY: Dict[str, Callable] = {}


def register_dataset(name: str):
    """Decorator to register dataset preparation functions.
    
    Args:
        name: Unique identifier for the dataset
        
    Example:
        @register_dataset("my_custom_dataset")
        def prepare_my_dataset(split="train", subset_ids=None, **kwargs):
            # ... implementation
            return texts, ids
    """
    def decorator(func):
        _DATASET_REGISTRY[name] = func
        return func
    return decorator


@register_dataset("squad")
def prepare_squad(split: str = "train", subset_ids: List[str] = None, **kwargs) -> Tuple[List[str], List[str]]:
    """Prepare SQuAD v2 dataset for training.
    
    Args:
        split: Dataset split to load
        subset_ids: Optional list of IDs to filter dataset
        
    Returns:
        Tuple of (texts, ids)
    """
    dataset = load_dataset("rajpurkar/squad_v2", split=split)
    dataset = add_shared_context_ids(dataset)
    if subset_ids:
        dataset = dataset.filter(lambda ex: ex["id"] in subset_ids)
    texts = [f"{ex['title']}: {ex['context']}" for ex in dataset]
    ids = [ex["shared_ids"] for ex in dataset]
    return texts, ids


@register_dataset("pretrain_wiki")
def prepare_dwiki(split: str = "train", subset_ids: List[str] = None, **kwargs) -> Tuple[List[str], List[str]]:
    """Prepare Dolmino Wiki dataset for training.
    
    Args:
        split: Dataset split to load
        subset_ids: Optional list of IDs to filter dataset
        
    Returns:
        Tuple of (texts, ids)
    """
    dataset = load_dataset("allenai/dolmino-mix-1124", "wiki", split=split)
    if subset_ids:
        dataset = dataset.filter(lambda ex: ex["id"] in set(subset_ids))
    dataset = dataset.map(truncate_sample_length)
    texts = [ex["text"] for ex in dataset]
    ids = [ex["id"] for ex in dataset]
    return texts, ids


def prepare_data(dataset_name: str, **kwargs) -> Tuple[List[str], List[str]]:
    """Prepare dataset for LMLM training.
    
    Supports both built-in datasets and custom registered datasets.
    To use experimental datasets, import the experimental dataloader module first:
        from experiment.annotate import dataloader_experimental
    
    Args:
        dataset_name: Name of the dataset to prepare
        **kwargs: Additional arguments passed to dataset preparation function
        
    Returns:
        Tuple of (texts, ids)
        
    Raises:
        ValueError: If dataset_name is not supported
        
    Examples:
        >>> texts, ids = prepare_data("squad", split="train")
        >>> texts, ids = prepare_data("pretrain_wiki", split="train")
    """
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(_DATASET_REGISTRY.keys())}"
        )
    return _DATASET_REGISTRY[dataset_name](**kwargs)


def get_available_datasets() -> List[str]:
    """Return list of registered dataset names.
    
    Returns:
        List of dataset names that can be used with prepare_data()
    """
    return list(_DATASET_REGISTRY.keys())