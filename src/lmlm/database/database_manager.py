import os
import re
import json
import time
import logging
from typing import Dict, List, Optional, Union
from collections import Counter
from datasets import DatasetDict
from lmlm.database.topk_retriever import TopkRetriever

# Setup logger
logger = logging.getLogger(__name__)

def extract_lookups(text, pattern=None):
    """Extract (entity, attribute, answer) triplets from annotated text."""
    if pattern is None:
        pattern_lst = [
            r'\[dblookup\(([^,]+),\s*([^,]+)\)\s*->\s*(.*?)\]',
            r"\[dblookup\('(.+?)',\s*'(.+?)'\)\s*->\s*(.+?)\]"
        ]

    matches = []
    for p in pattern_lst:
        matches.extend(re.findall(p, text, re.DOTALL))
    matches = list({tuple(match) for match in matches})

    return [[item.strip("'").strip('"').strip() for item in match] for match in matches]

def update_atomic_knowledge(example):
    triplets = extract_lookups(example["annotated_text"])
    example.update({"atomic_knowledge": triplets})
    return example

def extract_database(dataset):
    if 'atomic_knowledge' not in dataset.column_names:
        dataset = dataset.map(update_atomic_knowledge)
    return dataset

class DatabaseLookupError(Exception):
    """Custom exception for database lookup failures, tracking failure statistics."""
    failure_statistics = Counter()

    def __init__(self, message, failure_type):
        super().__init__(message)
        self.__class__.failure_statistics[failure_type] += 1

    @classmethod
    def get_failure_statistics(cls):
        return dict(cls.failure_statistics)

class DatabaseManager:
    def __init__(self):
        self.database_name = None
        self.database_org_file = []
        self.database = {
            "entities": set(),
            "relationships": set(),
            "return_values": set(),
            "triplets": set(),
        }
        self.topk_retriever = None

    def __len__(self):
        return len(self.database["triplets"])

    def __str__(self):
        return (
            f"DatabaseManager: {len(self)} triplets, "
            f"{len(self.database['entities'])} entities, "
            f"{len(self.database['relationships'])} relationships, "
            f"{len(self.database['return_values'])} return values."
        )

    def init_topk_retriever(self, model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=None, default_threshold=None):

        if self.topk_retriever is None:
            self.topk_retriever = TopkRetriever(
                self.database["triplets"], model_name, top_k, default_threshold, database_name=self.database_name
            )
            logger.info(f"Top-k retriever initialized with {len(self)} triplets and threshold {self.topk_retriever.default_threshold}.")

    def retrieve_from_database(self, prompt: str, threshold: Optional[float] = None):
        """Retrieve a single top-1 database result from a prompt containing dblookup. If lookup fails, raise an error."""
        
        pattern_lst = [
            r"\[dblookup\('((?:[^'\\]|\\.)+)',\s*'((?:[^'\\]|\\.)+)'\)\s*->",
            r"\[dblookup\('(.+?)',\s*'(.+?)'\)\s*->",
            r"<\|db_entity\|>(.+?)<\|db_relationship\|>(.+?)<\|db_return\|>"
        ]

        matches = {tuple(match) for pattern in pattern_lst for match in re.findall(pattern, prompt)}

        if not matches:
            raise DatabaseLookupError(
                f"[dblookup_fail_5] No valid dblookup pattern found in prompt: {prompt}", 
                "no_match_found"
            )

        if len(matches) > 1:
            raise DatabaseLookupError(
                f"[dblookup_fail_1] Multiple dblookup matches found: {matches} in prompt: {prompt}", 
                "multiple_matches"
            )

        entity, relationship = matches.pop()

        self.init_topk_retriever()
        results = self.topk_retriever.retrieve_top_k(entity, relationship, threshold=threshold)

        if not results:
            raise DatabaseLookupError(
                f"[dblookup_fail_3] No retrieval results for entity='{entity}', relationship='{relationship}'",
                "no_retrieval_data_found"
            )

        return results[0]  # Return top-1 retrieval

    def build_database(self, dataset: Union[DatasetDict, List[Dict]], database_name: Optional[str] = None, database_org_file: Optional[str] = None):
        """Build database from atomic knowledge dataset."""
        self.database_name = database_name
        if database_org_file:
            self.database_org_file.append(database_org_file)

        if isinstance(dataset, DatasetDict):
            dataset = extract_database(dataset)

        if "atomic_knowledge" not in dataset.column_names:
            dataset = extract_database(dataset)
        for example in dataset:
                # raise ValueError("Example missing 'atomic_knowledge'.")
            for entity, relationship, return_value in example["atomic_knowledge"]:
                self.database["entities"].add(entity)
                self.database["relationships"].add(relationship)
                self.database["return_values"].add(return_value)
                self.database["triplets"].add((entity, relationship, return_value))

        logger.info(f"Built database with {len(self)} triplets.")

    def load_database(self, load_path: str):
        """Load database from JSON file."""
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Database file not found: {load_path}")

        with open(load_path, "r") as f:
            data = json.load(f)

        if not all(k in data for k in ["entities", "relationships", "return_values", "triplets"]):
            raise ValueError(f"Invalid database format in {load_path}.")
        
        self.database_name = os.path.basename(load_path).split(".json")[0]
        self.database_org_file.append(load_path)

        self.database["entities"].update(data["entities"])
        self.database["relationships"].update(data["relationships"])
        self.database["return_values"].update(data["return_values"])
        self.database["triplets"].update(tuple(triplet) for triplet in data["triplets"])

        logger.info(f"Loaded database from {load_path}.")

    def save_database(self, save_path: str):
        """Save current database to a JSON file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        serializable = {k: list(v) for k, v in self.database.items()}
        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=4)
        logger.info(f"Database saved at {save_path}.")

def load_current_database(database_dir: str) -> DatabaseManager:
    """Load all database files from directory."""
    db_manager = DatabaseManager()
    for file in os.listdir(database_dir):
        if file.endswith(".json"):
            db_manager.load_database(os.path.join(database_dir, file))
    return db_manager


if __name__ == "__main__":
    import time
    import psutil
    import os

    def run_retrieval_tests(db_manager):
        """Run a series of retrieval tests using the provided knowledge triplets"""
        
        # Test entity-relationship pairs from provided knowledge triplets
        test_entity_lst = [
            "Thomas Macdonald-Paterson",
            "Trivium",
            "Runar Berg",
            "Prattville",
            "Izzy Asper",
            "Josef Paldus"
        ]
        
        test_relationship_lst = [
            "Political Office",
            "Components",
            "Sport",
            "State",
            "Role in Media",
            "Affiliation"
        ]
        
        # Additional variant test cases (variants of the same entities and relationships)
        variant_entity_lst = [
            "Macdonald-Paterson, Thomas",  # Name format variation
            "The Trivium",                 # With article
            "Berg, Runar",                 # Last name first format
            "Prattville, Alabama",         # Combined with state
            "Israel Asper",                # Full name instead of nickname
            "Joseph Paldus"                # Spelling variation
        ]
        
        # Variations in relationship formatting/naming
        variant_relationship_lst = [
            "political_office",            # Underscore format
            "elements",                    # Synonym for components
            "played_sport",                # Verb form
            "located_in",                  # Alternative relationship
            "profession",                  # Alternative relationship
            "university"                   # More specific relationship
        ]
        
        # Incorrect/non-existent entity-relationship pairs for negative testing
        incorrect_entity_lst = [
            "Thomas Macdonald",            # Incomplete name
            "Quadrivium",                  # Related but different concept
            "Runar Smith",                 # Wrong last name
            "Prattville",                  # Correct entity
            "Izzy Aspers",                 # Misspelled
            "Josef Einstein"               # Wrong person
        ]
        
        incorrect_relationship_lst = [
            "Political Party",             # Wrong but related relationship
            "Inventors",                   # Unrelated relationship
            "Team",                        # More specific than just sport
            "Country",                     # Higher level geographical unit
            "Birth Date",                  # Unrelated relationship
            "Publications"                 # Related but different relationship
        ]
        
        print("\n=== RUNNING EXACT MATCH TESTS ===")
        start_time = time.time()
        db_manager.init_topk_retriever()
        end_time = time.time()

        print(f"Top-k Retriever Initialization Time: {end_time - start_time:.2f}s")
        print_system_usage()

        for entity, relationship in zip(test_entity_lst, test_relationship_lst):
            try:
                start_time = time.time()
                results = db_manager.topk_retriever.retrieve_top_k(entity, relationship, threshold=0.8)
                end_time = time.time()
                print(f"Query Results for '{entity}' and '{relationship}':", results)
                print(f"Query Time: {end_time - start_time:.4f}s")
                print_system_usage()
            except Exception as e:
                print(f"Query Error for '{entity}' and '{relationship}':", e)
        
        print("\n=== RUNNING VARIANT TESTS ===")
        for entity, relationship in zip(variant_entity_lst, variant_relationship_lst):
            try:
                start_time = time.time()
                results = db_manager.topk_retriever.retrieve_top_k(entity, relationship, threshold=0.8)
                end_time = time.time()
                print(f"Query Results for '{entity}' and '{relationship}':", results)
                print(f"Query Time: {end_time - start_time:.4f}s")
                print_system_usage()
            except Exception as e:
                print(f"Query Error for '{entity}' and '{relationship}':", e)
        
        print("\n=== RUNNING NEGATIVE TESTS ===")
        for entity, relationship in zip(incorrect_entity_lst, incorrect_relationship_lst):
            try:
                start_time = time.time()
                results = db_manager.topk_retriever.retrieve_top_k(entity, relationship, threshold=0.8)
                end_time = time.time()
                print(f"Query Results for '{entity}' and '{relationship}':", results)
                print(f"Query Time: {end_time - start_time:.4f}s")
                print_system_usage()
            except Exception as e:
                print(f"Query Error for '{entity}' and '{relationship}':", e)
        
        # Test expected output for known triplets
        print("\n=== RUNNING VALIDATION TESTS ===")
        expected_results = {
            ("Thomas Macdonald-Paterson", "Political Office"): "Parliament of Queensland",
            ("Trivium", "Components"): "grammar, logic, rhetoric",
            ("Runar Berg", "Sport"): "football",
            ("Prattville", "State"): "Alabama",
            ("Izzy Asper", "Role in Media"): "media baron",
            ("Josef Paldus", "Affiliation"): "University of Waterloo"
        }
        
        for (entity, relationship), expected in expected_results.items():
            try:
                results = db_manager.topk_retriever.retrieve_top_k(entity, relationship)
                if results and expected in str(results):
                    print(f"✓ PASS: '{entity}' + '{relationship}' correctly returned '{expected}'")
                else:
                    print(f"✗ FAIL: '{entity}' + '{relationship}' returned {results}, expected to contain '{expected}'")
            except Exception as e:
                print(f"✗ ERROR: '{entity}' + '{relationship}' test failed with error:", e)
        

    def print_system_usage():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"Memory Usage: {mem_info.rss / (1024 ** 3):.2f} GB")
        print(f"CPU Usage: {cpu_percent}%")

    database_path = "./database_v4/trex11k-annotator_database.json"
    
    db_manager = DatabaseManager()
    
    start_time = time.time()

    db_manager.load_database(database_path)
    print(db_manager)
    db_manager.database_name = database_path.split("/")[-1].split(".json")[0]

    end_time = time.time()
    print(f"Loading Time: {end_time - start_time:.2f}s")
    print_system_usage()

    run_retrieval_tests(db_manager)