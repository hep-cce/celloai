import sys
import os
import logging
import click
import torch
import time
import re

import torch, json, tempfile, pathlib
from torch.profiler import profile, record_function, ProfilerActivity

from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain.docstore.document import Document
from typing import List
from pydantic import Field
from transformers import GenerationConfig

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

celloai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, celloai_path)

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_chroma.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)

from config import TEXT_EMBEDDING_MODEL_NAME, CODE_EMBEDDING_MODEL_NAME
from config import ROOT_DIRECTORY, SOURCE_DIRECTORY, PERSIST_DIRECTORY
from config import MODEL_ID, MODEL_BASENAME, MAX_NEW_TOKENS, MODELS_PATH
from config import NUM_CODE, NUM_TEXT
from prompt_template import langchain_prompt_template 
from chromadb.config import Settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

import re
import glob

from load_models import load_model
from config import *
from function_relationships import get_function_relationships, load_function_graph


# Custom retreiver to improve context's quality for text+code data 
class CelloRetriever(BaseRetriever):
    """
    Class to define custom retriever for code and text combinations.
    """
    docs_code: list[Document] = []
    docs_text: list[Document] = []
    k1 : int = NUM_CODE  # code
    k2 : int = NUM_TEXT  # text
    retriever_code: BaseRetriever = Field(None)
    retriever_text: BaseRetriever = Field(None)

    def __init__(self, retriever_code, retriever_text, k1, k2):
        super().__init__()
        self.k1: int = k1
        self.k2: int = k2
        self.retriever_code = retriever_code
        self.retriever_text = retriever_text

    def collect_patterns_for_matching(self, query: str) -> list[str]:
        pattern = re.compile(r'```(.*?)```', re.DOTALL)
        list_patterns = pattern.findall(query)
        return list_patterns

    def add_callgraph_lineage(self, patterns):
        callgraph_patterns = []
        callgraph_text = ""
        if DIR_PATH:
            dir_path = DIR_PATH
            doxygen_html_path = os.path.join(dir_path,"html")
            json_path = os.path.join(doxygen_html_path,"merged_graph.json")
            try:
                graph = load_function_graph(json_path)
                for pattern in patterns:
                    all_relationships = get_function_relationships(pattern, graph)
                    if all_relationships:
                        callgraph_text += f"\n```{pattern}```"
                        for relationships in all_relationships:
                            callgraph_text += f'  \n{relationships["name"]}\n'
                            callgraph_text += f"  calls\n"
                            for called_func in relationships["calls"]:
                                callgraph_text += f"  - ```{called_func}```\n"
                                callgraph_patterns.append(called_func)
                            callgraph_text += f"  and is called by\n"
                            for caller in relationships["called_by"]:
                                callgraph_text += f"  - ```{caller}```\n"
                                callgraph_patterns.append(caller)
            except:
                print(f"Merged callgraph related error. Check existence at {json_path}")

        return callgraph_patterns, callgraph_text

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """Return the first k documents from the list of documents"""

        # Collect patterns within ``` for exact matching
        patterns = self.collect_patterns_for_matching(query)
        patterns = list(set(patterns)) #deduplicate
        print("\n *** Matching for patterns: ", patterns)

        # Look in top n docs
        # this returns all documents ranked according to semantic match 
        code_and_scores: List[Tuple[Document, float]] = (
            self.retriever_code.vectorstore.similarity_search_with_score(query, k=1000)
        )
        text_and_scores: List[Tuple[Document, float]] = (
            self.retriever_text.vectorstore.similarity_search_with_score(query, k=100)
        )

        # return many docs from above; then rerank if pattern match
        front : List[Tuple[Document, float]] = []
        back  : List[Tuple[Document, float]] = []
        for i, (doc, score) in enumerate(code_and_scores, 1):
            #print(f"\n-------DOC {i}--------\n", doc, score)
            if any(pattern in doc.page_content for pattern in patterns):
                front.append((doc,score))
            else:
                back.append((doc,score))
        if patterns:
            code_and_scores = front + back

        front : List[Tuple[Document, float]] = []
        back  : List[Tuple[Document, float]] = []
        for i, (doc, score) in enumerate(text_and_scores, 1):
            #print(f"\n-------DOC {i}--------\n", doc, score)
            if any(pattern in doc.page_content for pattern in patterns):
                front.append((doc,score))
            else:
                back.append((doc,score))
        if patterns:
            text_and_scores = front + back

        with open("./retrieved_docs", 'w', encoding="utf-8") as file:
            for i, (doc,score) in enumerate(code_and_scores[:self.k1], 1):
                file.write(f"\n-------CODE DOC {i} : Score {score:.4f}--------\n")
                file.write(str(doc))
            for i, (doc,score) in enumerate(text_and_scores[:self.k2], 1):
                file.write(f"\n-------TEXT DOC {i} : Score {score:.4f}--------\n")
                file.write(str(doc))

        docs_code = [doc for doc, _ in code_and_scores] 
        docs_text = [doc for doc, _ in text_and_scores]
        # then return a given number of top ranking
        return docs_code[:self.k1] + docs_text[:self.k2]


def get_text_embeddings(device_type="cuda"):
    return HuggingFaceEmbeddings(
        model_name=TEXT_EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type, "trust_remote_code": True},
    )


def get_code_embeddings(device_type="cuda"):
    return HuggingFaceEmbeddings(
        model_name=CODE_EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type, "trust_remote_code": True},
    )


def retrieve_docs_enhance_prompt_from_cello(
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
    print(PERSIST_DIRECTORY, "NUM CODE DOCS", len(file_names))
    file_names = [meta["source"] for meta in dbget_text["metadatas"] if "source" in meta]
    print(PERSIST_DIRECTORY, "NUM TEXT DOCS", len(file_names))
 
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
        hist_patterns = retriever_comb.collect_patterns_for_matching(concatenated_string)
        patterns.extend(hist_patterns)
        print("HISTORY2=", hist_patterns) 
    
    # Add callgraph lineage of matched patterns
    patterns = list(set(patterns)) #deduplicate
    callgraph_patterns, callgraph_text = retriever_comb.add_callgraph_lineage(patterns)

    # Add callers and callees to pattern-match code reranking
    if ENHANCE_PROMPT_WITH_LINEAGE:
        query = query + callgraph_text

    return query, retrieved_docs


def retrieval_qa_pipline_with_logging(device_type, use_history, promptTemplate_type="llama", temperature=0.2, chatbot=False):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    """
    (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within ingest.py.

    (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
    their respective huggingface repository, project page or github repository.
    """
    if device_type == "hpu":
        from gaudi_utils.embeddings import load_embeddings

        embeddings = load_embeddings()
    else:
        embeddings = get_text_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")

    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    #text_embeddings = get_text_embeddings(device_type)
    #db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=text_embeddings, collection_name="text_collection", client_settings=CHROMA_SETTINGS)
    dbget = db.get() # ids embeddings metadatas documents uris data
    file_names = [meta["source"] for meta in dbget["metadatas"] if "source" in meta]
    #print("Stored file names:", file_names)
    print(PERSIST_DIRECTORY, "NUM DOCS", len(file_names))
    num_docs = NUM_TEXT
    retriever = db.as_retriever(search_kwargs={"k": num_docs}) # search_type="similarity|mmr" # maximum marginal relevance 
    #print("RETRIEVER", retriever)

    # get the prompt template and memory if set by the user.
    prompt_t = langchain_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history, chatbot=chatbot)
    prompt, memory = prompt_t.get_prompt_memory()

    # load the llm pipeline
    llm, tokenizer = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging, temperature=temperature)

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # TODO try other chains types as well. stuff, refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  
            #chain_type="map_reduce",  # ValueError: A single document was longer than the context length, we cannot handle this
            #chain_type="refine",      # 
            #chain_type="map_rerank",  
            retriever=retriever,
            return_source_documents=True,  #verbose=False,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    #return qa

    def qa_with_logging(query):
        """Wrapper function that prints the query, retrieved docs, prompt, memory, and response."""

        # Manually retrieve documents using the retriever.
        retrieved_docs = retriever.get_relevant_documents(query)
       
        #print("\n=== Prompt ===")
        #print(prompt)
       
        prompt_string = prompt.template
        encode_prompt = tokenizer.encode(prompt_string)
        len_encode_prompt = len(encode_prompt)
        print(f"\n=== Prompt Template Token Length = {len_encode_prompt} ===\n")
        len_token = len_encode_prompt

        if use_history:
            print("\n=== Memory ===")
            print(memory)
            encode_memory = tokenizer.encode(memory)
            len_encode_memory = len(encode_memory)
            print(f"\n=== Memory Token Length = {len_encode_memory} ===\n")
            len_token += len_encode_memory

        #print("\n=== Query ===")
        #print(query)
        encode_query = tokenizer.encode(query)
        len_encode_query = len(encode_query)
        print(f"\n=== Query Token Length = {len_encode_query} ===\n")
        len_token += len_encode_query

        #print("\n=== Retrieved Documents ===")
        for i, doc in enumerate(retrieved_docs, 1):
            retrieved_string = f"Document {i}: {doc.metadata['source']}:\n{doc.page_content}\n"
            #print(retrieved_string)
            encode_doc = tokenizer.encode(retrieved_string)
            len_encode_doc = len(encode_doc)
            print(f"--- Token Length = {len_encode_doc} ---\n")
            len_token += len_encode_doc

        ## Retrieve documents along with their similarity scores
        #retrieved_docs_with_scores = retriever.vectorstore.similarity_search_with_score(query, k=11)  # Retrieve top 10 docs
        ## Print retrieved documents along with similarity scores
        #print("\n=== Retrieved Documents with Similarity Scores ===")
        #for i, (doc, score) in enumerate(retrieved_docs_with_scores, 1):
        #    print(f"Document {i}:{doc.metadata['source']}:\n{doc.page_content}")
        #    print(f"  Similarity Score: {score:.4f}")  # Print similarity score
        #    print(f"--- ------------- ---\n")
        #    #print(f"  Content: {doc.page_content[:300]}...")  # Print first 300 characters for preview
        #    #print(f"  Metadata: {doc.metadata}\n")  # Print metadata (e.g., filename)

        # Run the QA pipeline.
        start_time = time.perf_counter()
        response = qa(query)
        execution_time = time.perf_counter() - start_time

        out_token = tokenizer.encode(response["result"])
        len_out_token = len(out_token)
        print(f"\n=== Total Token count: input={len_token}, output={len_out_token} in {execution_time:.2f} secs, tokens/sec = {len_token/execution_time:.2f} ===\n")
        start = response["result"].find(prompt_t.END_STRING) + len(prompt_t.END_STRING)
        #print(f"\n=== Response ===")
        #print(response["result"][start:])
        return {"result":response["result"][start:], "source_documents": response["source_documents"]}

    return qa_with_logging

# Retrieval pipeline for coupled code and text
# Used for code generation tasks
# Calls custom CelloRetriever
def retrieval_code_text_qa_pipline_with_logging(device_type, use_history, promptTemplate_type="llama", temperature=0.2, chatbot=False):
    """
    Combined retrieval of text and code
    """
    text_embeddings = get_text_embeddings(device_type)
    code_embeddings = get_code_embeddings(device_type)

    logging.info(f"Loaded text embeddings from {TEXT_EMBEDDING_MODEL_NAME}")
    logging.info(f"Loaded code embeddings from {CODE_EMBEDDING_MODEL_NAME}")

    # load the vectorstore
    db_code = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=code_embeddings, collection_name="code_collection", client_settings=CHROMA_SETTINGS)
    db_text = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=text_embeddings, collection_name="text_collection", client_settings=CHROMA_SETTINGS)
    dbget_code = db_code.get() # ids embeddings metadatas documents uris data
    dbget_text = db_text.get() # ids embeddings metadatas documents uris data

    file_names = [meta["source"] for meta in dbget_code["metadatas"] if "source" in meta]
    #print("Stored file names:", file_names)
    print(PERSIST_DIRECTORY, "NUM CODE DOCS", len(file_names))
    file_names = [meta["source"] for meta in dbget_text["metadatas"] if "source" in meta]
    print(PERSIST_DIRECTORY, "NUM TEXT DOCS", len(file_names))
 
    # Retrieve documents along with their similarity scores
    retriever_code = db_code.as_retriever() # search_type="similarity|mmr" # maximum marginal relevance 
    retriever_text = db_text.as_retriever() # search_type="similarity|mmr" # maximum marginal relevance 

    # Combine text and code retrievers
    num_code_ret = NUM_CODE
    num_text_ret = NUM_TEXT
    retriever_comb = CelloRetriever(retriever_code, retriever_text, num_code_ret, num_text_ret)

    # get the prompt template and memory if set by the user.
    prompt_t = langchain_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history, chatbot=chatbot)
    prompt, memory = prompt_t.get_prompt_memory()
 
    # load the llm pipeline
    llm, tokenizer = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging, temperature=temperature)

    if "nopenai" in MODEL_ID:
        # 
        #

        def prompt_qa_chain(query):
            docs = retriever_comb._get_relevant_documents(query)
            context = "\n\n".join(d.page_content for d in docs)
            #context = "\n\n".join(textwrap.shorten(d["text"], 4096) for d in docs)
            #for doc in docs:
            #    print(doc)
            #print(prompt)
            message = f"""### Context {context} \n ### Question {query} \n ### Answer"""

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": message},
            ]
            
            #message = message[0:8192]
            #encode_prompt = tokenizer.encode(message)
            #len_encode_prompt = len(encode_prompt)
            #print(f"\n\n Context token length {len_encode_prompt} \n\n message \n\n ")
          
            #out = llm(
            #        inputs,
            #        max_new_tokens=4096,
            #        do_sample=False,
            #        cache_implementation="offloaded",
            #)[0]["generated_text"]
            #return {"result":out[len(message):], "source_documents": docs}
 
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(llm.device)
            
            #N = 2106
            #inputs['input_ids'] = inputs['input_ids'][:,0:N]
            #inputs['attention_mask'] = inputs['attention_mask'][:,0:N]
            #print(inputs)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            print("input_ids", input_ids.size(), input_ids.nelement() * input_ids.element_size() / (1024**2), "MB")
            print("attention_mask", attention_mask.size(), attention_mask.nelement() * attention_mask.element_size() / (1024**2), "MB")

            gen_cfg = GenerationConfig(
                #max_new_tokens=512,
                #cache_implementation="offloaded", 
                #cache_implementation="quantized", 
                #cache_implementation="static",    
                #max_cache_len=500,                 
                #temperature=0.7,
                torch_dtype=torch.bfloat16,
            )
          
            #llm.generation_config.sliding_window = 2000
            ##torch.set_float32_matmul_precision('high') # highest, medium
            #with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = llm.generate(**inputs, generation_config=gen_cfg)
                out = tokenizer.decode(out[0], skip_special_tokens=True)



            #input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(llm.device)
            #n_tokens = input_ids.size(1)
            #
            ## ----  streamed pre-fill  -----
            #past = None
            #attention_mask = None          # we grow it along the way
            #CHUNK = 1024 
            #with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            #    for start in range(0, n_tokens, CHUNK):
            #        end = min(start + CHUNK, n_tokens)
            #        chunk_ids = input_ids[:, start:end]
            #
            #        out = llm(
            #            input_ids     = chunk_ids,
            #            past_key_values = past,     # None for the first chunk
            #            attention_mask = attention_mask,
            #            use_cache      = True
            #        )
            #        past = out.past_key_values      # reuse for the next chunk
            #
            #        # grow the causal mask so the model knows total seq-len so far
            #        if attention_mask is None:
            #            attention_mask = torch.ones_like(chunk_ids, dtype=torch.long)
            #        else:
            #            attention_mask = torch.cat(
            #                (attention_mask,
            #                 torch.ones_like(chunk_ids, dtype=torch.long)),
            #                dim=1
            #            )
            #
            ## at this point the whole prompt is inside `past` but never sat in VRAM all at once
            #print(f"Prefill done – sequence length {n_tokens}, peak on GPU-0:",
            #      torch.cuda.max_memory_allocated(0) / 1e9, "GB")
            #
            ## ----  now generate new tokens  -----
            #gen_cfg = GenerationConfig(
            #    max_new_tokens=1024,
            #    cache_implementation="offloaded",   # now it really kicks in
            #)
            #next_ids = llm.generate(
            #    input_ids = input_ids[:, -1:],      # *only* the last token
            #    past_key_values = past,
            #    attention_mask  = attention_mask,
            #    generation_config = gen_cfg,
            #)
            #out = tokenizer.decode(next_ids[0])
            #print(tokenizer.decode(next_ids[0], skip_special_tokens=True))

            return {"result":out[:], "source_documents": docs}


        return prompt_qa_chain

    else:
        #
        #
        if use_history:
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",  # TODO try other chains types as well. stuff, refine, map_reduce, map_rerank
                retriever=retriever_comb,
                return_source_documents=True,  # verbose=True,
                callbacks=callback_manager,
                chain_type_kwargs={"prompt": prompt, "memory": memory},
            )
        else:
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",  
                retriever=retriever_comb,
                return_source_documents=True,  #verbose=False,
                callbacks=callback_manager,
                chain_type_kwargs={
                    "prompt": prompt,
                },
            )

        #return qa

        def qa_code_text_with_logging(query):
            """Wrapper function that prints the query, retrieved docs, prompt, memory, and response."""

            print("\n=== Prompt ===")
            #print(prompt)
           
            if COMPUTE_METRICS:
                prompt_string = prompt.template
                encode_prompt = tokenizer.encode(prompt_string)
                len_encode_prompt = len(encode_prompt)
                print(f"\n=== Prompt Template Token Length = {len_encode_prompt} ===\n")
                len_token = len_encode_prompt

            # Collect patterns within ``` for exact matching
            patterns = retriever_comb.collect_patterns_for_matching(query)
            print("CURRENT", patterns)

            if use_history:
                print("\n=== Memory ===")
                # TODO
                #encode_memory = tokenizer.encode(memory)
                #len_encode_memory = len(encode_memory)
                #print(f"\n=== Memory Token Length = {len_encode_memory} ===\n")
                #len_token += len_encode_memory
                if memory.chat_memory.messages:
                    idx = memory.chat_memory.messages[-1:][0].content.rfind("User:")
                    last_hist = memory.chat_memory.messages[-1:][0].content[idx:]
                    #print(memory.chat_memory.messages[-1:][0].content[idx:])
                    hist_patterns = retriever_comb.collect_patterns_for_matching(last_hist)
                    patterns = patterns + hist_patterns
                    #print("HISTORY=", hist_patterns) 
                #print("\n=== Memory ===")
 
            # Add callgraph lineage of matched patterns
            patterns = list(set(patterns)) #deduplicate
            callgraph_patterns, callgraph_text = retriever_comb.add_callgraph_lineage(patterns)

            print("\n=== Query ===")
            # Add callers and callees to pattern-match code reranking
            if ENHANCE_PROMPT_WITH_LINEAGE:
                query = query + callgraph_text
            print(query)
            
            if COMPUTE_METRICS:
                encode_query = tokenizer.encode(query)
                len_encode_query = len(encode_query)
                print(f"\n=== Query Token Length = {len_encode_query} ===\n")
                len_token += len_encode_query
            
            # Run the QA pipeline.
            start_time = time.perf_counter()
            response = qa(query)
            execution_time = time.perf_counter() - start_time
         
            if COMPUTE_METRICS:
                print(f"\n=== Retrieved -- Code 1:{NUM_CODE}, Text:{1+NUM_CODE}:{NUM_CODE+NUM_TEXT} ===")
                for i, doc in enumerate(response["source_documents"], 0):
                    retrieved_string = f"Document {i}: {doc.metadata['source']}:\n{doc.page_content}\n"
                    #print(retrieved_string[:300])
                    #print(f"--- ------------- ---\n")
                    encode_doc = tokenizer.encode(retrieved_string)
                    len_encode_doc = len(encode_doc)
                    print(f"--- Token Length = {len_encode_doc} ---")
                    len_token += len_encode_doc
 
                out_token = tokenizer.encode(response["result"])
                len_out_token = len(out_token)
                print(f"\n=== Total Token count: input={len_token}, output={len_out_token} in {execution_time:.2f} secs, tokens/sec = {len_token/execution_time:.2f} ===\n")
            
            start = response["result"].find(prompt_t.END_STRING) + len(prompt_t.END_STRING)
            #print(f"\n=== Response ===")
            #print(response["result"][start:])
 
            return {"result":response["result"][start:], "source_documents": response["source_documents"]}


        return qa_code_text_with_logging
