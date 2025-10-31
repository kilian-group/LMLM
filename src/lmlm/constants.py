import os

# Get package root directory (2 levels up from this file)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define paths relative to root
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompts")
DATA_DIR = os.path.join(ROOT_DIR, "data")
CONFIGS_DIR = os.path.join(ROOT_DIR, "configs")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Make sure the directories exist
os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CONFIGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- Special token format ---
DB_START_TOKEN = "<|db_entity|>"          # Begins a lookup call
DB_SEP_TOKEN = "<|db_relationship|>"                # Separates entity and relation in the query
DB_RETRIEVE_TOKEN = "<|db_return|>"   # Signals insertion point for returned value
DB_END_TOKEN = "<|db_end|>"            # Marks end of lookup block

# --- Legacy format (used for annotation) ---
LEGACY_DB_START_TOKEN = "[dblookup"
LEGACY_DB_SEP_TOKEN = "', '"
LEGACY_DB_RETRIEVE_TOKEN = "') -> "
LEGACY_DB_END_TOKEN = "]"

TINY_LLAMA2_TOKENIZER_PATH = "./tokenizer/tiny-llama2"