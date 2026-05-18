import re

import parse_cpp
from parse_cpp import Language
from parse_cpp import Treesitter

# notes
# runTFCSShapeValidation.cxx cpp decorators
# FCS-GPU-Llama-3.3-70B-Instruct-RAGS/FastCaloSimAnalyzer/macro/runTFCSShapeValidation.cxx bad message
# dir_path = r'/home/atif//firstPCA.cxx' # need to use latin-1 encoding for diacritics
# 
# Currently reading files larger than 2 lines
# Added a watermark statement for LLM comment generation
# Skip functions with existing comments, or Rand4Hits_hip.cxx
# dir_path = r'/home/atif/TFCS1DFunctionFactory.cxx' # treesitter skips 1st functions due to file level comment block

def extract_cpp_functions(file_path):
    """
    This function returns a list of function source code.
    """

    functions = []
    print(f'Collecting functions in {file_path}', flush=True)
    with open(file_path, "r", encoding="latin-1") as file:
        # Read the entire content of the file into a string
        file_bytes = file.read().encode()

        file_extension = "cpp" 
        programming_language = Language.CPP 

        treesitter_parser = Treesitter.create_treesitter(programming_language)
        treesitterNodes: list[TreesitterMethodNode] = treesitter_parser.parse(
            file_bytes
        )

        for node in treesitterNodes:
            # Count the number of lines in the function
            num_lines = node.method_source_code.count('\n')
            if node.doc_comment is None:
                num_lines_comment = 0
            else:
                num_lines_comment =  node.doc_comment.count('\n')
            # Add uncommented functions to list
            if  num_lines_comment < 3 and num_lines > 2:
                function_name = node.name
                print(f"Will generate comment for function: {function_name}")
                functions.append(node.method_source_code)

    file.close()
    return functions


def extract_cpp_function_names_params_returns(file_path):
    """
    This function returns a list of function_name, its params, and its return.
    """

    function_name_params_returns_gt = [] # list to store ground truth 

    print(f'Collecting functions, params, return in {file_path}', flush=True)
    with open(file_path, "r", encoding="latin-1") as file:
        file_bytes = file.read().encode()
        file_extension = "cpp"
        programming_language = Language.CPP

        treesitter_parser = Treesitter.create_treesitter(programming_language)
        treesitterNodes: list[TreesitterMethodNode] = treesitter_parser.parse(file_bytes)

        for node in treesitterNodes:
            num_lines = node.method_source_code.count('\n')
            if node.doc_comment is None:
                num_lines_comment = 0
            else:
                num_lines_comment =  node.doc_comment.count('\n')
            if  num_lines_comment < 3 and num_lines > 2:
                function_name = node.name
                params = node.param_count
                ret_count = node.return_count
                list_params = node.list_params
                #var_names = [re.split(r'\s*&?\*?', p.strip())[-1] for p in list_params]
                var_names = []
                for p in list_params:
                    p = p.strip().split('=')[0]
                    m = re.search(r'([A-Za-z_]\w*)\s*$', p)
                    var_names.append(m.group(1) if m else "")
                print(f"./src/extract.py Function: {function_name}, Params: {params}, Returns: {ret_count}, Params = {var_names}")
                function_name_params_returns_gt.append({
                    "function": function_name,
                    "param_count": params,
                    "arg_var_names": var_names,
                    "return_count": ret_count
                })
 
    file.close()
    return function_name_params_returns_gt


def extract_cpp_comments(file_path):
    
    functions_comments = []
    functions = []
    
    with open(file_path, "r", encoding="latin-1") as file:
        # Read the entire content of the file into a string
        file_bytes = file.read().encode()

        file_extension = "cpp" #utils.get_file_extension(file_name)
        programming_language = Language.CPP #utils.get_programming_language(file_extension)

        treesitter_parser = Treesitter.create_treesitter(programming_language)
        treesitterNodes: list[TreesitterMethodNode] = treesitter_parser.parse(
            file_bytes
        )

        for node in treesitterNodes:
            # Count the number of lines in the function
            num_lines = node.method_source_code.count('\n')
            if num_lines > 2:
                first_newline = node.method_source_code.find('\n')
                first_line = node.method_source_code[0:first_newline]
                functions_comments.append(f'{node.doc_comment} \n {first_line}')
                functions.append(f'{first_line}')

    file.close()
    return functions_comments, functions


