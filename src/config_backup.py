# Configuration for Cello-AI

#TEXT_EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"  # Uses 1.5 GB of VRAM (High Accuracy with lower VRAM usage)
#TEXT_EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl" # Uses 5 GB of VRAM (Most Accurate of all models)
TEXT_EMBEDDING_MODEL_NAME = "Lajavaness/bilingual-embedding-large" # Max positional embeddings 8192
#TEXT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
#TEXT_EMBEDDING_MODEL_NAME = "intfloat/e5-mistral-7b-instruct" #SLOW
#TEXT_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"

#CODE_EMBEDDING_MODEL_NAME = "Salesforce/SFR-Embedding-Code-2B_R" # Max positional embeddings 8192
#CODE_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
#CODE_EMBEDDING_MODEL_NAME = "microsoft/codebert-base" # Max position embeddings 512
CODE_EMBEDDING_MODEL_NAME = "Lajavaness/bilingual-embedding-large" # Max positional embeddings 8192
#CODE_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
#CODE_EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"

# Codebase location
DIR_PATH = "/home/atif/cello-ai/SOURCE_DOCUMENTS/FastCaloSimAnalyzer/"
#DIR_PATH = "/home/atif/cello-ai/SOURCE_DOCUMENTS/pixeltrack-standalone/src"
#DIR_PATH = "/home/atif/cello-ai/SOURCE_DOCUMENTS/p2r-tests/"
#DIR_PATH = "/home/atif/cello-ai/SOURCE_DOCUMENTS/wire-cell-gen-kokkos/"
#DIR_PATH = "/home/atif/cello-ai/SOURCE_DOCUMENTS/wire-cell-2dtoy/"

# Include callers/callees in the context
ENHANCE_PROMPT_WITH_LINEAGE = 1

# Number of code chunks to retrieve
NUM_CODE = 40

# Number of text documents to retrieve
NUM_TEXT = 10

# Compute performance metrics [tokens/s]
COMPUTE_METRICS = 1


# Good
#MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
#MODEL_BASENAME = None
#CONTEXT_WINDOW_SIZE = 128000 # 128k for Llama-3.3-70B-instruct
#MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

#MODEL_ID = "deepseek-ai/DeepSeek-R1-0528"
#MODEL_ID = "unsloth/DeepSeek-R1-GGUF"
#MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
#MODEL_ID = "microsoft/MAI-DS-R1"
#MODEL_BASENAME = None

#MODEL_ID = "mistralai/Mistral-Large-Instruct-2411"
##MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
#MODEL_BASENAME = None
#CONTEXT_WINDOW_SIZE = 128000 # 128k for Llama-3.3-70B-instruct
#MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

# Not good but fast
#MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
#MODEL_BASENAME = None
#CONTEXT_WINDOW_SIZE = 1280000 # 128k for Llama-3.3-70B-instruct
#MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

# For Code Gen
#MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
##MODEL_ID = "google/gemma-3n-E4B-it"
###MODEL_ID = "agentica-org/DeepCoder-14B-Preview"
#CONTEXT_WINDOW_SIZE = 131072 
#MODEL_BASENAME = None
#MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

##MODEL_ID = "Menlo/Jan-nano-128k"
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
##MODEL_ID = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
##MODEL_ID = "Qwen/Qwen3-8B"
CONTEXT_WINDOW_SIZE = 131072 
MODEL_BASENAME = None
MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

#MODEL_ID = "openai/gpt-oss-120b"
#MODEL_ID = "openai/gpt-oss-20b"
#CONTEXT_WINDOW_SIZE = 131072 
#MODEL_BASENAME = None
#MAX_NEW_TOKENS = int(CONTEXT_WINDOW_SIZE/4)

#MODEL_ID = "ibm-granite/granite-3.2-8b-instruct"
##MODEL_ID = "ibm-granite/granite-8b-code-instruct-128k"
#MODEL_BASENAME = None
#CONTEXT_WINDOW_SIZE = 131072 
#MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

#bigcode/starcoder2-15b
#phi-3

#MODEL_ID = "microsoft/Phi-4-reasoning"
#MODEL_ID = "microsoft/Phi-4-reasoning-plus"
#MODEL_BASENAME = None
#CONTEXT_WINDOW_SIZE = 128000 
#MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)


#MODEL_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
#MODEL_BASENAME = None
#CONTEXT_WINDOW_SIZE = 10485760
#MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)

#MODEL_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
#MODEL_BASENAME = None
#CONTEXT_WINDOW_SIZE = 1048576
#MAX_NEW_TOKENS = CONTEXT_WINDOW_SIZE  # int(CONTEXT_WINDOW_SIZE/4)


####
#### (FOR GGUF MODELS)
####

# MODEL_ID = "TheBloke/Llama-2-13b-Chat-GGUF"
# MODEL_BASENAME = "llama-2-13b-chat.Q4_K_M.gguf"

# MODEL_ID = "TheBloke/Llama-2-7b-Chat-GGUF"
# MODEL_BASENAME = "llama-2-7b-chat.Q4_K_M.gguf"

#MODEL_ID = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF"
#MODEL_BASENAME = "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"

# LLAMA 3 # use for Apple Silicon
# Good
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_BASENAME = None

# Code LLAMA 3 # use for Codes
# MODEL_ID = "meta-llama/CodeLlama-7b-hf"
# MODEL_BASENAME = None

# LLAMA 3 # use for NVIDIA GPUs
# MODEL_ID = "unsloth/llama-3-8b-bnb-4bit"
# MODEL_BASENAME = None

# MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
# MODEL_BASENAME = "mistral-7b-instruct-v0.1.Q8_0.gguf"

# MODEL_ID = "TheBloke/Llama-2-70b-Chat-GGUF"
# MODEL_BASENAME = "llama-2-70b-chat.Q4_K_M.gguf"

####
#### (FOR HF MODELS)
####

# MODEL_ID = "NousResearch/Llama-2-7b-chat-hf"
# MODEL_BASENAME = None
# MODEL_ID = "TheBloke/vicuna-7B-1.1-HF"
# MODEL_BASENAME = None
# MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-HF"
# MODEL_ID = "TheBloke/guanaco-7B-HF"
# MODEL_ID = 'NousResearch/Nous-Hermes-13b' # Requires ~ 23GB VRAM. Using STransformers
# alongside will 100% create OOM on 24GB cards.
# llm = load_model(device_type, model_id=model_id)

####
#### (FOR GPTQ QUANTIZED) Select a llm model based on your GPU and VRAM GB. Does not include Embedding Models VRAM usage.
####

##### 48GB VRAM Graphics Cards (RTX 6000, RTX A6000 and other 48GB VRAM GPUs) #####

### 65b GPTQ LLM Models for 48GB GPUs (*** With best embedding model: hkunlp/instructor-xl ***)
# MODEL_ID = "TheBloke/guanaco-65B-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/Airoboros-65B-GPT4-2.0-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/gpt4-alpaca-lora_mlp-65B-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/Upstage-Llama1-65B-Instruct-GPTQ"
# MODEL_BASENAME = "model.safetensors"

##### 24GB VRAM Graphics Cards (RTX 3090 - RTX 4090 (35% Faster) - RTX A5000 - RTX A5500) #####

### 13b GPTQ Models for 24GB GPUs (*** With best embedding model: hkunlp/instructor-xl ***)
# MODEL_ID = "TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ"
# MODEL_BASENAME = "Wizard-Vicuna-13B-Uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
# MODEL_ID = "TheBloke/vicuna-13B-v1.5-GPTQ"
# MODEL_BASENAME = "model.safetensors"
# MODEL_ID = "TheBloke/Nous-Hermes-13B-GPTQ"
# MODEL_BASENAME = "nous-hermes-13b-GPTQ-4bit-128g.no-act.order"
# MODEL_ID = "TheBloke/WizardLM-13B-V1.2-GPTQ"
# MODEL_BASENAME = "gptq_model-4bit-128g.safetensors

### 30b GPTQ Models for 24GB GPUs (*** Requires using intfloat/e5-base-v2 instead of hkunlp/instructor-large as embedding model ***)
# MODEL_ID = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
# MODEL_BASENAME = "Wizard-Vicuna-30B-Uncensored-GPTQ-4bit--1g.act.order.safetensors"
# MODEL_ID = "TheBloke/WizardLM-30B-Uncensored-GPTQ"
# MODEL_BASENAME = "WizardLM-30B-Uncensored-GPTQ-4bit.act-order.safetensors"

##### 8-10GB VRAM Graphics Cards (RTX 3080 - RTX 3080 Ti - RTX 3070 Ti - 3060 Ti - RTX 2000 Series, Quadro RTX 4000, 5000, 6000) #####
### (*** Requires using intfloat/e5-small-v2 instead of hkunlp/instructor-large as embedding model ***)

### 7b GPTQ Models for 8GB GPUs
# MODEL_ID = "TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ"
# MODEL_BASENAME = "Wizard-Vicuna-7B-Uncensored-GPTQ-4bit-128g.no-act.order.safetensors"
# MODEL_ID = "TheBloke/WizardLM-7B-uncensored-GPTQ"
# MODEL_BASENAME = "WizardLM-7B-uncensored-GPTQ-4bit-128g.compat.no-act-order.safetensors"
# MODEL_ID = "TheBloke/wizardLM-7B-GPTQ"
# MODEL_BASENAME = "wizardLM-7B-GPTQ-4bit.compat.no-act-order.safetensors"

####
#### (FOR GGML) (Quantized cpu+gpu+mps) models - check if they support llama.cpp
####

# MODEL_ID = "TheBloke/wizard-vicuna-13B-GGML"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q4_0.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q6_K.bin"
# MODEL_BASENAME = "wizard-vicuna-13B.ggmlv3.q2_K.bin"
# MODEL_ID = "TheBloke/orca_mini_3B-GGML"
# MODEL_BASENAME = "orca-mini-3b.ggmlv3.q4_0.bin"

