import sys
import os
import logging
import click
import torch
import time

celloai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, celloai_path)

from config import TEXT_EMBEDDING_MODEL_NAME, CODE_EMBEDDING_MODEL_NAME
from config import ROOT_DIRECTORY, SOURCE_DIRECTORY, PERSIST_DIRECTORY
from config import MODEL_ID, MODEL_BASENAME, MAX_NEW_TOKENS, MODELS_PATH
from chromadb.config import Settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)


import re
import glob

from config import TEXT_EMBEDDING_MODEL_NAME, CODE_EMBEDDING_MODEL_NAME
from extract import extract_cpp_functions, extract_cpp_comments
from write import rewrite_file_with_comments
from retrieval_pipeline import retrieval_code_text_qa_pipline_with_logging
from retrieval_pipeline import retrieval_qa_pipline_with_logging
from llamacpp_request import query_llamacpp_server_with_rag
from prompt_template import doxygen_prompt, chatbot_prompt 


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
    Implements the main CelloAI Assistant 

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'mps', 'cuda', etc.
    - show_sources (bool): Flag to determine whether to display the source documents used for answering.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Notes:
    - Logging information includes the device type, whether source documents are displayed, and the use of history.
    - The user can exit the interactive loop by entering "exit".
    - The source documents are displayed if the show_sources flag is set to True.
    """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    if llamacpp_server == True:
        """
        Llamacpp server pipeline.
        """
        conversation_history = []
        system_prompt = chatbot_prompt
        print(f"\nSystem Prompt: {system_prompt}")
        print("Enter your query below. Type 'exit' or 'quit' to end the conversation.")

        while True:
            try:
                user_query = input("\n> You: ")
                if user_query.lower() in ["exit", "quit"]:
                    break

                assistant_response = query_llamacpp_server_with_rag(
                    user_query=user_query,
                    history=conversation_history,
                    system_prompt=system_prompt
                )

                pattern = "<|start|>assistant<|channel|>final<|message|>"
                response_start = assistant_response.find(pattern) + len(pattern)
                print(f"\n> Thining: {assistant_response[0:response_start]}")
                print(f"\n> Assistant: {assistant_response[response_start:]}")

                # Update conversation history
                conversation_history.append({"role": "user", "content": user_query})
                conversation_history.append({"role": "assistant", "content": assistant_response})

            except (KeyboardInterrupt, EOFError):
                break
        print("\nConversation ended.")

    else:
        """
        Regular pipeline using HF transformers. Use LlamaCpp server for better performance.
        """
        qa = retrieval_code_text_qa_pipline_with_logging(device_type, use_history, promptTemplate_type=model_type, temperature=temperature, chatbot=True)
        #qa = retrieval_qa_pipline_with_logging(device_type, use_history, promptTemplate_type=model_type, temperature=temperature, chatbot=True)

        # Interactive questions and answers
        while True:
        #for i in range(0,1):
            print("\nEnter a query: ")
            query = input(" ")
            if query == "exit":
                break

            #query = "You are an expert GPU and high-performance computing software engineer. You will convert a set of CUDA __global__ or __device__ kernels from the FastCaloSim project into equivalent, performant OpenMP kernels using the above strategy. The resulting code must compile, target new GPU architectures, and integrate cleanly into the existing CMake build. 1. For every CUDA function in Table 1 rewrite the implementation in OpenMP. 2. Preserve numerical results bit-wise where feasible (double precision). 3. Insert explicit device memory management where the CUDA runtime previously handled mapping: allocate once per event batch and reuse. 4. Use features from OpenMP that are known to be more efficient. 5. Supply a two-sentence performance note per kernel (expected speed-up, occupancy bottlenecks). Table 1: ```testHello``` ```testCell``` ```testGeo``` ```testGeo_g``` ```simulate_clean``` ```simulate_ct``` ```simulate_A``` ```atomicAdd``` ```CaloGpuGeneral_cu::simulate_hits```"
            #query = "You are an expert GPU and high-performance computing software engineer. You will convert a set of CUDA __global__ or __device__ kernels from the FastCaloSim project into equivalent, performant OpenMP kernels using the above strategy. The resulting code must compile, target new GPU architectures, and integrate cleanly into the existing CMake build. 1. For every CUDA function rewrite the implementation in OpenMP. 2. Preserve numerical results bit-wise where feasible (double precision). 3. Insert explicit device memory management where the CUDA runtime previously handled mapping: allocate once per event batch and reuse. 4. Use features from OpenMP that are known to be more efficient. 5. Supply a two-sentence performance note per kernel (expected speed-up, occupancy bottlenecks)."
            # Get the answer from the chain
            res = qa(query)
            answer, docs = res["result"], res["source_documents"]

            # Print the result
            print("> Question:")
            print(query)
            print("> Answer:")
            print(answer)

            # Print the relevant sources used for the answer
            #for document in docs:
            #    print("----------------------------------SOURCE DOCUMENTS---------------------------")
            #    print("\n> " + document.metadata["source"] + ":")
            #    print(document.page_content)

            # Log the Q&A to CSV only if save_qa is True
            if save_qa:
                log_to_file(query, answer, docs)
   



if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
