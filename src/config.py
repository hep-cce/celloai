# Configuration for Cello-AI

TEXT_EMBEDDING_MODEL_NAME = "Lajavaness/bilingual-embedding-large" # Max positional embeddings 8192
CODE_EMBEDDING_MODEL_NAME = "Lajavaness/bilingual-embedding-large" # Max positional embeddings 8192

# Codebase location
DIR_PATH = "/home/atif/cello-ai/SOURCE_DOCUMENTS/FastCaloSimAnalyzer/"

# Include callers/callees in the context
ENHANCE_PROMPT_WITH_LINEAGE = 1

# Number of code chunks to retrieve
NUM_CODE = 40

# Number of text documents to retrieve
NUM_TEXT = 10

# Compute performance metrics [tokens/s]
COMPUTE_METRICS = 1

# See src/config_backups.py for more options
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_BASENAME = None
CONTEXT_WINDOW_SIZE = 1280000 # 128k for Llama-3.3-70B-instruct
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

import os
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder for storing database
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/../SOURCE_DOCUMENTS"

PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/../DB"

MODELS_PATH = "/data/llms" # for dahlia CSI



