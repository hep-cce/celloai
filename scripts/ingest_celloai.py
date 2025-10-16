import sys
import os
import logging
import click
import torch
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

celloai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, celloai_path)

from chromadb.config import Settings
from langchain_community.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

import parse_cpp
from parse_cpp import Treesitter
from retrieval_pipeline import retrieval_qa_pipline_with_logging

INGEST_THREADS = os.cpu_count() or 8

from config import TEXT_EMBEDDING_MODEL_NAME, CODE_EMBEDDING_MODEL_NAME
from config import ROOT_DIRECTORY, SOURCE_DIRECTORY, PERSIST_DIRECTORY
from config import MODEL_ID, MODEL_BASENAME, MAX_NEW_TOKENS, MODELS_PATH
from retrieval_pipeline import get_text_embeddings, get_code_embeddings
from chromadb.config import Settings
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
)

TEXT_DOCUMENT_MAP = {
    ".html": UnstructuredHTMLLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".py": TextLoader,
    ".pdf": UnstructuredFileLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
}

CPP_EXTENSIONS = {".cxx", ".cpp", ".h", ".hh", ".cu", ".cuh", ".C", ".c", ".cc", ".hpp"}


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    try:
        file_extension = os.path.splitext(file_path)[1]
        # Read code as text
        if file_extension in CPP_EXTENSIONS:
            loader_class = TEXT_DOCUMENT_MAP.get(".txt")
        else:
            loader_class = TEXT_DOCUMENT_MAP.get(file_extension)
        
        if loader_class:
            loader = loader_class(file_path)
        else:
            raise ValueError("Document type is undefined")
        return loader.load()[0]
    except Exception as ex:
        print("%s loading error: \n%s" % (file_path, ex))
        return None


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        if futures is None:
            print(name + " failed to submit")
            return None
        else:
            data_list = [future.result() for future in futures]
            # return data and file paths
            return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            #print("Importing: " + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in TEXT_DOCUMENT_MAP.keys() or file_extension in CPP_EXTENSIONS:
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            try:
                #future = executor.submit(load_single_document, filepaths)
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                print("executor task failed: %s" % (ex))
                future = None
            if future is not None:
                futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                print("Exception: %s" % (ex))

    return docs


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        if futures is None:
            print(name + " failed to submit")
            return None
        else:
            data_list = [future.result() for future in futures]
            # return data and file paths
            return (data_list, filepaths)

def load_text_documents(paths) -> list[Document]:

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            try:
                #future = executor.submit(load_single_document, filepaths)
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                print("executor task failed: %s" % (ex))
                future = None
            if future is not None:
                futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                print("Exception: %s" % (ex))

    return docs


def convert_pdf_to_md(paths):

    new_paths = []
    # Converts PDFs to MDs and returns new path with MDs only
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    for path in paths:
        file_extension = os.path.splitext(path)[1]
        if file_extension == ".pdf":
            print("Converting to .md: " + path)
            rendered = converter(path)
            text, _, images = text_from_rendered(rendered)
            md_file_path = os.path.splitext(path)[0] + ".md"
            with open(md_file_path, 'w', encoding="utf-8") as destination_file:
                destination_file.write(text)
                #print(text)
            new_paths.append(md_file_path)
        else:
            new_paths.append(path)

    return new_paths


def collect_cpp(documents: list[Document]):
    
    # Collect documents 
    cpp_docs = []
    
    for doc in documents:
        if doc is not None:
            file_extension = os.path.splitext(doc.metadata["source"])[1]
            if file_extension in CPP_EXTENSIONS:
                print("Importing Code Doc: " + doc.metadata["source"])
                cpp_docs.append(doc)

    return cpp_docs



def collect_paths_text_documents(source_dir: str) -> list:
    # walks through SOURCE_DOCUMENTS directory and collects all paths
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in TEXT_DOCUMENT_MAP.keys():
                print("Importing Text Doc: " + file_name)
                paths.append(source_file_path)

    return paths

def main(device_type):

    text_embeddings = get_text_embeddings(device_type)
    code_embeddings = get_code_embeddings(device_type)

    print("\n***Don't forget to add all relevant file extensions in TEXT_DOCUMENT_MAP***\n")
    
    # Collect paths of text docs
    paths = collect_paths_text_documents(SOURCE_DIRECTORY)

    new_paths = convert_pdf_to_md(paths)
    
    text_documents = load_text_documents(new_paths)
    #for doc in text_documents:
    #    print(f"\n\n{doc}\n\n")
    #    print(f"*_*_*_*_*_*_*_*\n")

    # Load documents and split in chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(text_documents)
    logging.info(f"Loaded {len(text_documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    #for text in texts:
    #    print("TXT ", text.page_content)
    
    logging.info(f"Loaded embeddings from {TEXT_EMBEDDING_MODEL_NAME}")

    db = Chroma.from_documents(
        texts,
        embedding=text_embeddings,
        collection_name="text_collection",
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )



    documents = load_documents(SOURCE_DIRECTORY)
    cpp_documents = collect_cpp(documents)
   
    #print("CODE", cpp_documents)

    for doc in cpp_documents:
        #print(doc)
        #print(doc.page_content)
        #print(doc.metadata['source'])
        functions = []
        file_bytes = doc.page_content.encode()
        file_extension = "cpp" 
        from parse_cpp import Language
        programming_language = Language.CPP 

        treesitter_parser = Treesitter.create_treesitter(programming_language)
        treesitterNodes: list[TreesitterMethodNode] = treesitter_parser.parse(
            file_bytes
        )

        for node in treesitterNodes:
            # Count the number of lines in the function
            num_lines = node.method_source_code.count('\n')
            # Add uncommented functions to list
            if node.doc_comment == None and num_lines > 2:
                functions.append(node.method_source_code)

        texts = []
        for function in functions: 
            page = Document(page_content=function, metadata = {"source": doc.metadata['source']})
            texts.append(page)

        if len(texts) == 0:
            page = Document(page_content=doc.page_content, metadata = {"source": doc.metadata['source']})
            texts.append(page)
        #texts =  Document(page_content=function, metadata={"source": doc.metadata['source']})
        db = Chroma.from_documents(
            texts,
            embedding=code_embeddings,
            collection_name="code_collection",
            persist_directory=PERSIST_DIRECTORY,
            client_settings=CHROMA_SETTINGS,
        )



if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main(device_type="cuda")
