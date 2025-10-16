import sys
import os
import logging
import click
import torch
import time

celloai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, celloai_path)

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,    
)

import re
import glob

from extract import extract_cpp_functions, extract_cpp_comments
from write import rewrite_file_with_comments
from retrieval_pipeline import retrieval_qa_pipline_with_logging


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
def main(device_type, show_sources, use_history, model_type, save_qa, temperature):

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    qa = retrieval_qa_pipline_with_logging(device_type, use_history, promptTemplate_type=model_type, temperature=temperature, chatbot=False)
 
    # Code documentation
    dir_home = r'/home/atif/FCS-CelloAI-Tests/FastCaloSimAnalyzer/'
    #dir_home = r'/home/atif/FCS-GPU/FastCaloSimAnalyzer/'
    #dir_home = r'/home/atif/wire-cell-2dtoy/'
 
    file_extensions = [".cxx", ".cpp", ".cc", ".tcc"]
    head_extensions = [".h", ".hpp", ".hh", ".H"]

    # Generate function level Doxygen style comments
    list_all_files = []
    # search all source files inside a specific folder
    for ext in file_extensions:
        dir_path = dir_home+fr'/**/*{ext}'
        for file in glob.glob(dir_path, recursive=True):
            list_all_files.append(file)
    print(list_all_files)
   
    prompt = "Please read the entire C++ code given below. Generate a Doxygen style comment for each function. Write only the Doxygen style comment using only alphanumeric characters. Do not explain your thinking or write the function name.\n"
    for file_path in list_all_files:

        #print("\n" + "=" *10 + file_path + "=" * 10 + "\n")
        functions = extract_cpp_functions(file_path)
        #print("BBB", functions)
        rewrite_file_with_comments(functions, file_path, prompt, qa, temperature)

    # Generate class level Doxygen style comments
    list_all_files = []
    # search all header files inside a specific folder
    for ext in head_extensions:
        dir_path = dir_home+ifr'/**/*{ext}'
        for file in glob.glob(dir_path, recursive=True):
            list_all_files.append(file)
    print(list_all_files)
 
    prompt = "Please read the entire C++ code given below. Generate a Doxygen style comment for each class. Write only the Doxygen style comment using only alphanumeric characters. Do not explain your thinking or write the class name.\n"

    for file_path in list_all_files:

        #print("\n" + "=" *10 + file_path + "=" * 10 + "\n")
        functions = extract_cpp_functions(file_path)
        #print("BBB", functions)
        rewrite_file_with_comments(functions, file_path, prompt, qa, temperature)

    # Code summarization
    # Generate file level summary
    list_all_files = []
    # search all source files inside a specific folder
    dir_path = dir_home+r'/**/*.cxx'
    for file in glob.glob(dir_path, recursive=True):
        list_all_files.append(file)
    
    # search all header files inside a specific folder
    dir_path = dir_home+r'/**/*.h'
    for file in glob.glob(dir_path, recursive=True):
        list_all_files.append(file)
    print(list_all_files)
    
    with open(dir_home+f"/celloai_summary_temperature_{temperature}.txt", 'w', encoding="utf-8") as destination_file:
        destination_file.write(f"This is generated by {MODEL_ID}:{MODEL_BASENAME} at temperature {temperature}.\n")
        for file_path in list_all_files:

            functions_comments, functions = extract_cpp_comments(file_path)
            destination_file.write(f'\n\n *-*-*-*-*-*-*-*-*-*-*-*\n ')
            destination_file.write(f'\n File {file_path} has ')
            for function in functions:
                destination_file.write("\n    ")
                destination_file.write(function)

            source_file = ""
            for function_comment in functions_comments:
                source_file += function_comment
                source_file += "\n\n"
                #print(function_comment)
                #print('\n')

            #print(source_file)
            #query = "The following source file contains functions and comments of a C++ program. Write a concise summary for the source file." + source_file
            query = "The following source file contains functions and comments of a C++ program. Write a concise summary for the source file with the following sections: ## Overview: A brief description of the file's purpose and main functionality. ## Key Components: The main classes, data structures, and interfaces defined in the file. ## Core Functionality: The primary operations and algorithms implemented in the file. ## Technical Implementation: Notable implementation details, optimizations, or design patterns used. Write a concise summary for the source file." + source_file
            res = qa(query)

            answer, docs = res["result"], res["source_documents"]
            destination_file.write(answer)
            destination_file.flush()




if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
