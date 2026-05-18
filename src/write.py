import os
import numpy as np

from config import *
from config import TEXT_EMBEDDING_MODEL_NAME, CODE_EMBEDDING_MODEL_NAME
from llamacpp_request import openai_nonstream
#from prompt_template import doxygen_system_prompt
from retrieval_pipeline import get_text_embeddings, CHROMA_SETTINGS
from langchain_chroma.vectorstores import Chroma

doxygen_system_prompt = """You are an expert code documentation specialist for ATLAS FastCaloSim. You will use the provided context from technical literature to write high-quality documentation comments for code.
When generating comments:
- First understand how the specific function/method fits into the larger software system described in the context
- Briefly explain different parts of the code
- Connect the function's purpose to the broader architectural concepts when relevant
- Use consistent terminology from the literature/documentation
- Do not create full form of abbreviations unless it is derived from the context
- Be scientific and technical in your descriptions
- Format the comments appropriately for the language being documented
- Focus only on writing proper documentation comments - no explanations or meta-commentary is needed
If you cannot determine the function's purpose based on the provided context and code analysis, provide a basic comment based purely on the function signature and code."""



def remove_comment_block_init_end(answer):
    
    # this is needed as a check to remove any possibility
    # of incompilable hallucination or nested comment blocks
    answer = answer.replace('\n/**\n','')
    answer = answer.replace('/**\n','')
    answer = answer.replace('/**','')
    return answer.replace('*/','')


def query_llm_server(query: str):

    text_embeddings = get_text_embeddings()

    # Compute query vector (user_prompt + code) embedding
    query_vec = np.array(text_embeddings.embed_query(query))
    query_vec = query_vec / np.linalg.norm(query_vec)
    #print("QUERYVEC", query_vec)

    db_text = Chroma(persist_directory=PERSIST_DIRECTORY, 
            embedding_function=text_embeddings, 
            collection_name="text_collection",
            collection_metadata={"hnsw:space": "cosine"},
            client_settings=CHROMA_SETTINGS
            )
    dbget_text = db_text.get() 

    file_names = [meta["source"] for meta in dbget_text["metadatas"] if "source" in meta]
    print(PERSIST_DIRECTORY, "NUM TEXT DOCS in DB", len(file_names), flush=True)

    # Retrieve documents along with their similarity scores
    num_text_ret = NUM_TEXT
    retriever_text = db_text.as_retriever(search_type="similarity", search_kwargs={"k": num_text_ret})  
    retrieved_docs = retriever_text.get_relevant_documents(query)
    #print(retrieved_docs)
    #for i, doc in enumerate(retrieved_docs, 1):
    #    retrieved_string = f"{doc.metadata['source']}:{doc.page_content[0:100]}"
    #    page_content = doc.page_content
    #    page_content_vec = np.array(text_embeddings.embed_query(page_content))
    #    page_content_vec = page_content_vec/np.linalg.norm(page_content_vec)
    #    sims = np.dot(page_content_vec, query_vec)
    #    print(f'{retrieved_string} -> {sims}')

    context_str = "\n\n".join(d.page_content for d in retrieved_docs)
    prompt_text = f"""
Context:
- {context_str}

Question: {query}
"""
    messages = [{"role": "system", "content": doxygen_system_prompt}]
    messages.append({"role": "user", "content": prompt_text})
    nonstream_fn = openai_nonstream

    #host = "lambda2.csi.bnl.gov" #"localhost"
    host = "localhost"
    port = 8000
    max_tokens = 16384
    temperature = 0.1
    no_stream = True # Using non-streaming for simplicity in this example
    
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        #"stream": False,
    }
    text, latency, ttft, tokens = nonstream_fn(url, payload)
    
    response_rag = text
    response_rag_vec = np.array(text_embeddings.embed_query(response_rag))
    response_rag_vec = response_rag_vec / np.linalg.norm(response_rag_vec)

    return response_rag



def rewrite_file_with_comments(functions, file_path, prompt, qa, temperature, llm_server=False): 


    if len(functions) > 0:
        file_path_comments = file_path + ".comments.cxx"
        print(f'Generating comments for {len(functions)} functions in {file_path}', flush=True)
 
        functions_first_line = []
        for function in functions:
            #print("FF", function)
            newline_index = function.index('\n')
            line = function[:newline_index]
            functions_first_line.append(line)


        idx = 0
        try:
            with open(file_path, 'r', encoding="latin-1") as source_file:
                with open(file_path_comments, 'w', encoding="utf-8") as destination_file:
                    for line in source_file:
                        # if there are no function definitions
                        if len(functions_first_line) > 0:
                            # if all functions have been read
                            if idx == len(functions_first_line):
                                destination_file.write(line)
                            else:
                                # Strip the lines of trailing/leading whitespaces
                                if line.strip() == functions_first_line[idx].strip():
                                    query = prompt + functions[idx]
                                    if llm_server:
                                        answer = query_llm_server(query)
                                    else:
                                        res = qa(query)
                                        answer, docs = res["result"], res["source_documents"]
                                    destination_file.write("/**\n")
                                    clean_answer = remove_comment_block_init_end(answer)
                                    print(f"Going to write \n'{clean_answer}'", flush=True)
                                    destination_file.write(clean_answer)
                                    if llm_server:
                                         destination_file.write(f"*\n * [This comment was generated by CelloAI with openai/gpt-oss-120b at temperature 0.1.]\n")
                                    else:
                                         destination_file.write(f"*\n * [This comment was generated by CelloAI with {MODEL_ID}:{MODEL_BASENAME} at temperature {temperature}.]\n")
                                    destination_file.write("*/ \n")
                                    destination_file.write(line)
                                    print(f'Function {idx}: {line} Done', flush=True)
                                    idx = idx + 1
                                else:
                                    destination_file.write(line)
                        else:
                            destination_file.write(line)

        except FileNotFoundError:
            print(f"Error: The file {c_file_path} was not found.")
        except IOError as e:
            print(f"Error: An I/O error occurred. {e}")

        # Exchange file names
        #os.rename(file_path, file_path+'.orig')
        os.rename(file_path_comments, file_path)

    else:
        print(f'No functions captured.', flush=True)




