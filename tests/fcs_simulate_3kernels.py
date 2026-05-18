import sys
import os
import logging
import click
import torch
import time
from typing import List, Optional, Tuple

celloai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, celloai_path)

import re
import glob

_FENCE_RE = re.compile(
    r"""^\s*```[ \t]*([A-Za-z0-9_+\-]*)[ \t]*\r?\n  # opening fence + optional lang
        (.*?)                                            # code
        \r?\n\s*```\s*$                                   # closing fence
    """,
    re.DOTALL | re.VERBOSE,
)

def strip_triple_backticks(text: str) -> str:
    """
    If `text` is exactly one fenced code block like:
        ```cpp
        <code>
        ```
    return just <code>. Otherwise return text unchanged.
    """
    m = _FENCE_RE.match(text)
    if not m:
        return text
    code = m.group(2)
    return code

system_prompt = """
You are a helpful assistant, you will use the provided context to answer user questions.
1. You are taking the role of an expert GPU and high-performance computing software engineer who writes code.
2. The output code must compile, target new GPU architectures, and integrate cleanly into the existing code.
3. Read the given context before answering questions and think step by step. 
4. Do not use any other information for answering user. Provide a concise and relevant answer to the question.
5. If you can not answer a user question based on the provided context then give no response.
6. Do *not* wrap code in triple backticks like ```cpp code```.
7. Do *not* write anything that will not compile with a standard C++ compiler.
"""
#For example, if your response contains a function name foo, write it as ```foo```.i
#**Always use triple backquotes to capture exact strings of C/C++ symbols or functions, variables, classes names from the source code.**

from retrieval_pipeline import retrieve_docs_enhance_prompt_from_cello
from retrieval_pipeline import get_text_embeddings, get_code_embeddings, CHROMA_SETTINGS, CelloRetriever
from llamacpp_request import openai_nonstream 
from alcf_requests import query_alcf
from langchain_chroma.vectorstores import Chroma
from chromadb.config import Settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)
ENHANCE_PROMPT_WITH_LINEAGE = True

NUM_CODE = 2
NUM_TEXT = 1
PERSIST_DIRECTORY = "/home/atif/celloai-dev/DB_fcs_cuda/"
KERNEL = 1 # 0=clean, 1=count, 2=simulate

CELLOAI = True 
MAX_TOKENS = 16384 #32768
TEMPERATURE = 0.5
# Either ALCF
ALCF = False 
# Or BNL, CSD
HOST = "peony.csi.bnl.gov"
PORT = 8000
url = f"http://{HOST}:{PORT}/v1/chat/completions"


def add_rag(query: str, persist_db: str):

    text_embeddings = get_text_embeddings()
    code_embeddings = get_code_embeddings()

    # Compute query vector (user_prompt + code) embedding
    #query_vec = np.array(text_embeddings.embed_query(query))
    #query_vec = query_vec / np.linalg.norm(query_vec)
    #print("QUERYVEC", query_vec)

    db_code = Chroma(persist_directory=persist_db, 
            embedding_function=code_embeddings, 
            collection_name="code_collection", 
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=CHROMA_SETTINGS
            )
    db_text = Chroma(persist_directory=persist_db, 
            embedding_function=text_embeddings, 
            collection_name="text_collection",
            client_settings=CHROMA_SETTINGS
            )
    dbget_code = db_code.get() 
    dbget_text = db_text.get() 

    file_names = [meta["source"] for meta in dbget_code["metadatas"] if "source" in meta]
    print(persist_db, "NUM CODE DOCS in DB", len(file_names), flush=True)
    file_names = [meta["source"] for meta in dbget_text["metadatas"] if "source" in meta]
    print(persist_db, "NUM TEXT DOCS in DB", len(file_names), flush=True)

    # Retrieve documents along with their similarity scores
    num_text_ret = NUM_TEXT
    num_code_ret = NUM_CODE
    retriever_text = db_text.as_retriever(search_type="similarity", search_kwargs={"k": num_text_ret})  
    retriever_code = db_code.as_retriever(search_type="similarity", search_kwargs={"k": num_code_ret})  
    retrieved_docs = retriever_text.get_relevant_documents(query)
    retrieved_code = retriever_code.get_relevant_documents(query)

    retrieved_docs = retrieved_code + retrieved_docs

    context_str = "\n\n".join(d.page_content for d in retrieved_docs)
    prompt_text = f"""Based on the following context, please answer the question.
Context:
- {context_str}

Question: {query}
"""
    print(prompt_text)
    return prompt_text


def retrieve_docs_enhance_prompt_from_cello1(
    query: str,
    history: List[dict],
): #-> str, list[Document]:
    """
    Wrapper function to call CelloRetriever and return list of Document
    """
    text_embeddings = get_text_embeddings()
    code_embeddings = get_code_embeddings()

    db_code = Chroma(persist_directory=PERSIST_DIRECTORY, 
            embedding_function=code_embeddings, 
            collection_name="code_collection", 
            client_settings=CHROMA_SETTINGS
            )
    db_text = Chroma(persist_directory=PERSIST_DIRECTORY, 
            embedding_function=text_embeddings, 
            collection_name="text_collection", 
            client_settings=CHROMA_SETTINGS
            )
    dbget_code = db_code.get() 
    dbget_text = db_text.get() 

    file_names = [meta["source"] for meta in dbget_code["metadatas"] if "source" in meta]
    print(PERSIST_DIRECTORY, "NUM CODE DOCS in DB", len(file_names), flush=True)
    file_names = [meta["source"] for meta in dbget_text["metadatas"] if "source" in meta]
    print(PERSIST_DIRECTORY, "NUM TEXT DOCS in DB", len(file_names), flush=True)
 
    # Retrieve documents along with their similarity scores
    retriever_code = db_code.as_retriever()  
    retriever_text = db_text.as_retriever()  

    # Combine text and code retrievers
    num_code_ret = NUM_CODE
    num_text_ret = NUM_TEXT
    retriever_comb = CelloRetriever(retriever_code, retriever_text, num_code_ret, num_text_ret)
    retrieved_docs = retriever_comb._get_relevant_documents(query)

    # Collect patterns within ``` for exact matching
    patterns = retriever_comb.collect_patterns_for_matching(query)
    
    #Only taking patterns fromi the last QA
    if history:
        pulled_history = history[-2]
        concatenated_string = ", ".join(f"{k}: {v}" for k, v in pulled_history.items())
        hist_patterns = retriever_comb.collect_patterns_for_matching(concatenated_string)
        patterns.extend(hist_patterns)
        print("HISTORY1=", hist_patterns) 
        pulled_history = history[-1]
        concatenated_string = ", ".join(f"{k}: {v}" for k, v in pulled_history.items())

    # Add callgraph lineage of matched patterns
    patterns = list(set(patterns)) #deduplicate
    callgraph_patterns, callgraph_text = retriever_comb.add_callgraph_lineage(patterns)

    # Add callers and callees to pattern-match code reranking
    if ENHANCE_PROMPT_WITH_LINEAGE:
        query = query + callgraph_text

    return query, retrieved_docs


def query_llamacpp_server_with_rag(
    user_query: str,
    history: List[dict],
    system_prompt: str
) -> str:
    """
    Queries the LLM with a prompt augmented by context and conversation history.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    
    if CELLOAI:
        user_query, retrieved_docs = retrieve_docs_enhance_prompt_from_cello(user_query, history)
        #print("Enhanced user query", user_query) 
        context_str = "\n\n".join(d.page_content for d in retrieved_docs)
        prompt_text = f"""Based on the following context, please answer the question.

Context:
- {context_str}

Question: {user_query}
"""
        messages.append({"role": "user", "content": prompt_text})
    else:
        prompt_text = add_rag(user_query, PERSIST_DIRECTORY)
        messages.append({"role": "user", "content": prompt_text})

    # --- LLM Inference ---
    payload = {
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
    if ALCF:
        text = query_alcf(payload)
    else:
        text, latency, ttft, tokens = openai_nonstream(url, payload)
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))

    return text


def log_to_file(question, answer, docs):

    log_dir, log_file = "chat_history", "qa_log.txt"
    # Ensure log directory exists, create if not
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Construct the full file path
    log_path = os.path.join(log_dir, log_file)

    # Check if file exists, if not create and write headers
    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as file:
            file.write(f"INSERT MODEL NAME ETC \n")

    # Append the log entry
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{timestamp} \n Question: {question} \n Answer: {answer} \n")
        for i, doc in enumerate(docs, 1):
            retrieved_string = f"Document {i}: {doc.metadata['source']}:\n{doc.page_content}\n"
            file.write(retrieved_string)
            file.write("\n-------------\n")
        file.write("\n\n* _ * _ * _ * _ * _ * _ * _ * _ * _ * _ * _ * _ * _ *\n\n")



# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="llama3",
    type=click.Choice(
        ["llama3", "llama", "mistral", "non_llama", "deepseek-ai"],
    ),
    help="model type, llama3, llama, mistral or non_llama",
)
@click.option(
    '--temperature', 
    type=float, 
    help='0.0 < temperature <= 1, higher is more creative'
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="whether to save Q&A pairs to a CSV file (Default is False)",
)
@click.option(
    "--llamacpp_server",
    is_flag=True,
    help="Use LlamaCpp Servers for inference (Default is False)",
)



def main(device_type, show_sources, use_history, model_type, save_qa, temperature, llamacpp_server):
    """
    """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    if llamacpp_server == True:
        """
        Llamacpp server pipeline.
        """
        conversation_history = []
        print(f"\nSystem Prompt: {system_prompt}")

        if KERNEL == 0:
            user_query = "Port the function ```simulate_clean``` from CUDA to OpenMP target offload. ***Do not write anything other than the function**. Think carefully about the requirements of a C++ OpenMP code and which variables need to be mapped to GPU using is_device_pointer. Define a separate local variable for members of structures or classes if they are basic data types. For example, if you have a struct s with a member m, the define a local variable auto m = s.m and use that in the code. Do not write namespaces in the code but write the complete function. **Do not write anything other than the functions as the output will be automatically added to the existing code so it should compile**. Make sure that the output remains compilable."
            write_file = "/home/atif/FCS-GPU-benchmarks/unit-test1/FastCaloSimAnalyzer/FastCaloGpu/src/CaloGpuGeneral_omp_simulate_clean.cxx"
        elif KERNEL == 1:
            user_query = "Port the function ```simulate_ct``` from CUDA to OpenMP target offload. ***Do not write anything other than the function**. Do not write coding fence with triple backticks or anything that will not compile in C++. Think about which variables need to be mapped to GPU using is_device_pointer. Define a separate local variable for members of structures or classes if they are basic data types. For example, if you have a struct s with a member m, the define a local variable auto m = s.m and use that in the code. Make sure that the output remains compilable."
            write_file = "/home/atif/FCS-GPU-benchmarks/unit-test2/FastCaloSimAnalyzer/FastCaloGpu/src/CaloGpuGeneral_omp_simulate_ct.cxx"
        elif KERNEL == 2:
            user_query = "Port the function ```simulate_A``` from CUDA to OpenMP target offload. Do not rename functions or write helper or host functions. If atomicAdd appears anywhere, write it as a separate function. **Do not write anything other than the functions as the output will be automatically added to the existing code so it should compile**. Do not write coding fence with triple backticks or anything that will not compile in C++. Think about which variables need to be mapped to GPU using is_device_pointer. *Always define a separate local variable for members of structures or classes if they are basic data types.* For example, if you have a struct s with a member m, the define a local variable auto m = s.m and use that in the code. Make sure that the output remains compilable. ```HitCellMapping_d```"
            write_file = "/home/atif/FCS-GPU-benchmarks/unit-test3/FastCaloSimAnalyzer/FastCaloGpu/src/CaloGpuGeneral_omp_simulate_A.cxx"

        assistant_response = query_llamacpp_server_with_rag(
            user_query=user_query,
            history=conversation_history,
            system_prompt=system_prompt
        )

        assistant_response = strip_triple_backticks(assistant_response)

        print(f"\n//> Assistant: \n{assistant_response}")

        # Update conversation history
        conversation_history.append({"role": "user", "content": user_query})
        conversation_history.append({"role": "assistant", "content": assistant_response})
        
        with open(write_file, "w") as f:
            f.write(f"\n//> {HOST}:{PORT} Assistant: \n{assistant_response}")

    else:
        print("Only supporting LlamaCpp server tests. Use --llamacpp_server.") 


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
