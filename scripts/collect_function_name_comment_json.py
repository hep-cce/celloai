import sys
import os
import logging
import click
import torch
import time

celloai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, celloai_path)

from config import TEXT_EMBEDDING_MODEL_NAME, CODE_EMBEDDING_MODEL_NAME
from config import MODEL_ID, MODEL_BASENAME, MAX_NEW_TOKENS, MODELS_PATH
from chromadb.config import Settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

import re
import glob

from extract import extract_cpp_functions, extract_cpp_comments

def main():

    dir_home = r'./athena_expert_comments/Calorimeter/CaloClusterCorrection/src/'
 
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
  
    function_name_comment = []
    for file_path in list_all_files:

        functions_comments, functions = extract_cpp_comments(file_path)
        for f, c in zip(functions, functions_comments):
            print(c)
            function_name_comment.append({
                "function": f,
                "comment": c
            })

    import json
    json_output_path = "./athena_expert_comments/comments.json"
    with open(json_output_path, "w", encoding="utf-8") as jf:
        json.dump(function_name_comment, jf, ensure_ascii=False, indent=2)
        print(f"Function-comment JSON written to {json_output_path}", flush=True)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
