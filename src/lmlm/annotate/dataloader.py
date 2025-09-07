from datasets import load_dataset
from .utils import truncate_sample_length, chunk_wiki_text, add_shared_context_ids
from typing import List, Tuple


def prepare_squad(split="train", subset_ids=None, **kwargs):
    subset_ids = getattr(kwargs, "subset_ids", None)
    dataset = load_dataset("rajpurkar/squad_v2", split=split)
    dataset = add_shared_context_ids(dataset)
    if subset_ids:
        dataset = dataset.filter(lambda ex: ex["id"] in subset_ids)
    texts = [f"{ex['title']}: {ex['context']}" for ex in dataset]
    ids = [ex["shared_ids"] for ex in dataset]
    return texts, ids


def prepare_dwiki(split="train", subset_ids=None, **kwargs):
    subset_ids = getattr(kwargs, "subset_ids", None)
    dataset = load_dataset("allenai/dolmino-mix-1124", "wiki", split=split)
    if subset_ids:
        dataset = dataset.filter(lambda ex: ex["id"] in set(subset_ids))
    dataset = dataset.map(truncate_sample_length)
    texts = [ex["text"] for ex in dataset]
    ids = [ex["id"] for ex in dataset]
    return texts, ids


def prepare_fineweb(split="train", subset_ids=None, **kwargs):
    subset_ids = getattr(kwargs, "subset_ids", None)
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split=split)
    if subset_ids:
        dataset = dataset.filter(lambda ex: ex["id"] in set(subset_ids))
    dataset = dataset.map(truncate_sample_length)
    texts = [ex["text"] for ex in dataset]
    ids = [ex["id"] for ex in dataset]
    return texts, ids


def prepare_trex11k(split="train"):
    dataset = load_dataset("json", data_files=f"./data/trex_v4/trex11k_v4.json")[split]
    texts = [ex["input_text"] for ex in dataset]
    ids = [ex["uuid"] for ex in dataset]
    return texts, ids


def prepare_dwiki_bio(split="train", subset_ids=None, **kwargs):
    subset_ids = getattr(kwargs, "subset_ids", None)
    dataset = load_dataset("allenai/dolmino-mix-1124", "wiki", split=split)
    if subset_ids:
        dataset = dataset.filter(lambda ex: ex["id"] in set(subset_ids))
    texts = [ex["text"] for ex in dataset]
    ids = [ex["id"] for ex in dataset]
    return chunk_wiki_text(texts, ids)


def prepare_wiki_eval(split="train", subset_ids=None, **kwargs):
    dataset = load_dataset("json", data_files=f"../LLM-Shearing/datasets/eval/wiki_eval500.json")[split]
    texts = [ex["text"] for ex in dataset]
    ids = [ex["id"] for ex in dataset]
    return texts, ids


def prepare_conflictsbench(split="train", subset_ids=None, **kwargs):
    dataset = load_dataset("Warrieryes/CB_qa", split=split)
    def is_wiki(row):
        return str(row.get("default_evidence_category", "")).strip().lower() == "wikipedia" and str(row.get("semantic_conflict_evidence_category", "")).strip().lower() == "wikipedia"
    
    dataset = dataset.filter(is_wiki)
    # get the first 1000 examples
    dataset = dataset.select(range(1000))

    texts = [ex["semantic_conflict_evidence"] for ex in dataset]
    ids = [i for i in range(len(texts))]

    return texts, ids

def prepare_data(dataset_name: str, **kwargs) -> Tuple[List[str], List[str]]:
    dispatch = {
        "squad": prepare_squad,
        "allenai/dolmino-mix-1124": prepare_dwiki,
        "fineweb": prepare_fineweb,
        "trex11k": prepare_trex11k,
        "dwiki_bio": prepare_dwiki_bio,
        "wiki_eval": prepare_wiki_eval,
        "conflictsbench": prepare_conflictsbench,
    }
    if dataset_name not in dispatch:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dispatch[dataset_name](**kwargs)
