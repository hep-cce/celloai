# Configuration for Cello-AI

TEXT_EMBEDDING_MODEL_NAME = "Lajavaness/bilingual-embedding-large" # Max positional embeddings 8192
CODE_EMBEDDING_MODEL_NAME = "Lajavaness/bilingual-embedding-large" # Max positional embeddings 8192

# Codebase location
#DIR_PATH = "/home/atif/celloai-dev/SOURCE_DOCUMENTS_athena/athena/"
DIR_PATH = "/home/atif/celloai-dev/SOURCE_DOCUMENTS_fcs_cuda/FastCaloSimAnalyzer"

# Include callers/callees in the context
ENHANCE_PROMPT_WITH_LINEAGE = 1

# Number of code chunks to retrieve
NUM_CODE = 20

# Number of text documents to retrieve
NUM_TEXT = 10

# Compute performance metrics [tokens/s]
COMPUTE_METRICS = 1

# See src/config_backups.py for more options
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_BASENAME = None
CONTEXT_WINDOW_SIZE = 128000 # 128k for Llama-3.3-70B-instruct
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

import os
ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Define the folder that contains texts for RAG 
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/../SOURCE_DOCUMENTS_fcs_cuda/"

# Define the folder for storing database
#PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/../DB_athena_whole/"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/../DB_fcs_cuda/"

MODELS_PATH = "/data/llms" # for dahlia CSI



