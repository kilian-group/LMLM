import argparse
import os
from datasets import load_dataset
from lmlm.database.database_manager import DatabaseManager
from lmlm.training.utils.utils_filter import filter_invalid_dblookups


def load_args():
    parser = argparse.ArgumentParser(description="Process and store dataset in a database.")
    parser.add_argument("--annotation_path", type=str, required=True, help="Path to the JSON annotation file.")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save the processed database.")
    return parser.parse_args()


def load_and_filter_dataset(path):
    """Load and optionally filter a dataset."""
    try:
        if path.endswith(".json") or path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=path, split="train", field="examples")
        else:
            dataset = load_dataset(path, split="train", field="examples")
    except Exception as e:
        print(f"Error loading dataset with 'examples' field: {e}, trying without 'examples' field")
        try:
            if path.endswith(".json") or path.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=path, split="train")
            else:
                dataset = load_dataset(path, split="train")
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")

    return dataset


def main():
    args = load_args()

    if not args.save_path:
        filename = os.path.basename(args.annotation_path) + "_cleaned_database.json"
        args.save_path = os.path.join("./database", filename)

    dataset = load_and_filter_dataset(args.annotation_path)
    dataset = dataset.map(filter_invalid_dblookups)

    print(f"Dataset size: {len(dataset)}")
    print(f"First entry:\n{dataset[0] if len(dataset) > 0 else 'Empty dataset'}")

    db = DatabaseManager()
    db.build_database(dataset)
    print(f"Database built: {db}")
    db.save_database(args.save_path)
    print(f"Saved to: {args.save_path}")


def load_database():
    """Load and print summary of all databases in './database'."""
    for file in os.listdir("./database"):
        if file.endswith(".json"):
            path = os.path.join("./database", file)
            db = DatabaseManager()
            db.load_database(path)
            print(f"Loaded {path}: {db}")


if __name__ == "__main__":
    main()
