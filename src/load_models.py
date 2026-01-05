# This file was originally part of PromtEngineer/localGPT and has been modified.
#
# The original code is licensed under the MIT License, a copy of which
# is available in the LICENSES/ directory.
#
# All modifications are licensed under the BSD-3-Clause License.

from config import *
import torch, json, tempfile, pathlib
import logging
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoConfig,
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,   
    pipeline,
)

def load_full_model(model_id, model_basename, device_type, logging):
    """
    Load a full model using AutoModelForCausalLM.
    """

    if device_type.lower() in ["mps", "cpu", "hpu"]:
        logging.info("Using AutoModelForCausalLM")
        # tokenizer = LlamaTokenizer.from_pretrained(model_id, cache_dir="./models/")
        # model = LlamaForCausalLM.from_pretrained(model_id, cache_dir="./models/")

        model = AutoModelForCausalLM.from_pretrained(model_id,
                                            #  quantization_config=quantization_config,
                                            #  low_cpu_mem_usage=True,
                                            #  torch_dtype="auto",
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto",
                                             cache_dir=f"{MODELS_PATH}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=f"{MODELS_PATH}")
    else:
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=f"{MODELS_PATH}")
        logging.info("Tokenizer loaded")
        gen_cfg = GenerationConfig.from_pretrained(model_id)
        logging.info("GenerationConfig loaded")
        if "nopenai" in MODEL_ID:
            NUM_GPUS   = 8                          # adjust if you use fewer cards
            FIRST_GPU  = 0                          # 0 if you don't have display-GPU issues
            LAST_GPU   = NUM_GPUS - 1
            DTYPE      = torch.bfloat16            # or torch.float16 on Ampere GPUs
            
            # -- 1. read the config so we know the exact layer count ----------------------
            cfg = AutoConfig.from_pretrained(MODEL_ID)
            n_layers = cfg.num_hidden_layers       # works for dense & MoE alike
            print(f"{MODEL_ID} has {n_layers} transformer blocks")
            
            # -- 2. split layers as evenly as possible -----------------------------------
            per_gpu = math.ceil(n_layers / NUM_GPUS) - 1
            
            #device_map = {}
            #for gpu in range(NUM_GPUS):
            #    start = gpu * per_gpu
            #    end   = min(start + per_gpu, n_layers)
            #    if start >= end:
            #        break                           # more GPUs than layers
            #    # GPT-OSS block names follow the GPT-NeoX pattern: `model.layers.<idx>`
            #    device_map.update({
            #        f"model.layers.{i}": gpu + FIRST_GPU
            #        for i in range(start, end)
            #    })
            
            # embeddings + lm_head take ~80 MB, just keep them on GPU-0
            # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            #device_map.update({
            #    "model.layers.1": 1,#"cpu",
            #    "model.layers.2": 2,#"cpu",
            #    "model.layers.3": 7,#"cpu",
            #    "model.layers.4": 4,#"cpu",
            #    "model.layers.5": 5,#"cpu",
            #    "model.layers.6": 6,#"cpu",
            #    "model.layers.7": 7,#"nvme",
            #    "model.layers.8": "cpu",
            #    "model.layers.9": "cpu",
            #    "model.layers.10": "cpu",
            #    "model.layers.11": "cpu",
            #    "model.layers.12": "cpu",
            #    "model.layers.13": "cpu",
            #    "model.layers.14": "cpu",
            #    "model.layers.15": "cpu",
            #    "model.layers.16": "cpu",
            #    "model.layers.17": "cpu",
            #    "model.layers.18": "cpu",
            #    "model.layers.19": "cpu",
            #    "model.layers.20": "cpu",
            #    "model.layers.21": "cpu",
            #    "model.layers.22": "cpu",
            #    "model.layers.23": "cpu",
            #    "model.layers.24": "cpu",
            #    "model.layers.25": "cpu",
            #    "model.layers.26": "cpu",
            #    "model.layers.27": "cpu",
            #    "model.layers.28": "cpu",
            #    "model.layers.29": "cpu",
            #    "model.layers.30": "cpu",
            #    "model.layers.31": "cpu",
            #    "model.layers.32": "cpu",
            #    "model.layers.33": "cpu",
            #    "model.layers.34": "cpu",
            #    "model.layers.35": "cpu",
            #    "model.embed_tokens":    FIRST_GPU, # input embeddings  (was missing)
            #    "model.norm":            LAST_GPU, # final RMSNorm     (Llama-style)
            #    "model.embed_in":        LAST_GPU,
            #    "model.embed_out":       LAST_GPU, # if present
            #    "lm_head":               LAST_GPU,
            #    "model.final_layernorm": LAST_GPU  # NeoX naming; adjust if OSS differs
            #})

            #print("Device map preview :")
            #for k, v in list(device_map.items())[:]:
            #    print(f"{k:<32} → {v}")


            #device_map = {
            #    # Enable Expert Parallelism
            #    "distributed_config": DistributedConfig(enable_expert_parallel=1),
            #    # Enable Tensor Parallelism
            #    #"tp_plan": "auto",
            #}


            # Load configuration from the model to avoid warnings
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                #**device_map,
                #device_map=device_map,
                device_map="auto",
                attn_implementation="sdpa",        # disable Flash-Attn
                #attn_implementation="flash_attention_2",
                #use_flash_attention_2=True,
                #load_in_4bit=True,
                #device_map="balanced_low_0",
                cache_dir=MODELS_PATH,
                trust_remote_code=True,  
                torch_dtype=torch.bfloat16,
                #generation_config = gen_cfg,
                # Uncomment this line with you encounter CUDA out of memory errors
                #max_memory={0: "46GiB", 1: "46GiB", 2: "46GiB", 3: "46GiB", 4: "46GiB", 5: "46GiB", 6: "46GiB", 7: "46GiB"},  
            ) 
        else:
            bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                    )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                cache_dir=MODELS_PATH,
                trust_remote_code=True,  # set these if you are using NVIDIA GPU
               # quantization_config=bnb_config
               # int8_kv=True,
               # load_in_4bit=True,
               # bnb_4bit_quant_type="nf4",
               # bnb_4bit_compute_dtype=torch.float16,
               # max_memory={0: "48GB", 1: "48GB", 2: "48GB", 3: "48GB", 4: "48GB", 5: "48GB", 6: "48GB", 7: "48GB"},  # Uncomment this line with you encounter CUDA out of memory errors
            )

        model.tie_weights()
    return model, tokenizer


def load_model(device_type, model_id, model_basename=None, LOGGING=logging, temperature=0.2):
    """
    Select a model for text generation using the HuggingFace library.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    
    model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        max_length=MAX_NEW_TOKENS,
        temperature=temperature,
        # top_p=0.95,
        do_sample=True,
        repetition_penalty=1.15,
    )

    logging.info("Local LLM Loaded")

    if "nopenai" in MODEL_ID:
        return model, tokenizer
    else:
        local_llm = HuggingFacePipeline(pipeline=pipe)
        return local_llm, tokenizer



