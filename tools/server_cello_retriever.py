import sys
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Tuple

from fastmcp import FastMCP
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Add the repository's src directory so this server can import retrieval_pipeline.
SRC_DIRECTORY = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from retrieval_pipeline import (  # noqa: E402
    CHROMA_SETTINGS,
    CelloRetriever,
    get_code_embeddings,
    get_text_embeddings,
)


mcp = FastMCP(
    "CelloAI Retriever",
    instructions=(
        "Use celloai_retrieve for semantic search over an existing CelloAI "
        "Chroma index when the relevant source files, functions, classes, or "
        "documentation are not already known. Pass an absolute path to the "
        "Chroma persistence directory and use a focused query. Start with "
        "small result counts and increase them only when necessary. The "
        "embedding model names must match the models used to create the "
        "collections. Retrieved chunks may be incomplete or stale, so inspect "
        "the current source files before making final implementation claims. "
        "Use the separate CelloAI Callgraph server for callers, callees, call "
        "paths, reachability, and dependency relationships. Explicitly combine "
        "the retriever and callgraph tools when both semantic context and call "
        "relationships are required."
    ),
)


# Embedding models are expensive to initialize, so keep them for the lifetime
# of the MCP server process.
_text_embeddings_cache: Dict[Tuple[str, str], Any] = {}
_code_embeddings_cache: Dict[Tuple[str, str], Any] = {}

# Include embedding identity in the vector-store cache key. Reusing a Chroma
# object with a different query embedding model could produce dimension errors
# or incorrect behavior.
_vectorstore_cache: Dict[
    Tuple[str, str, str, str, str],
    Chroma,
] = {}

_cache_lock = RLock()


def get_cached_text_embeddings(model_name: str, device_type: str) -> Any:
    """Return a cached text embedding model.
    
    Note: The underlying get_text_embeddings() only accepts device_type;
    it uses TEXT_EMBEDDING_MODEL_NAME from config. The model_name parameter
    is kept for cache key compatibility but not passed through.
    """

    key = (model_name, device_type)

    with _cache_lock:
        if key not in _text_embeddings_cache:
            _text_embeddings_cache[key] = get_text_embeddings(device_type)

        return _text_embeddings_cache[key]


def get_cached_code_embeddings(model_name: str, device_type: str) -> Any:
    """Return a cached code embedding model.
    
    Note: The underlying get_code_embeddings() only accepts device_type;
    it uses CODE_EMBEDDING_MODEL_NAME from config. The model_name parameter
    is kept for cache key compatibility but not passed through.
    """

    key = (model_name, device_type)

    with _cache_lock:
        if key not in _code_embeddings_cache:
            _code_embeddings_cache[key] = get_code_embeddings(device_type)

        return _code_embeddings_cache[key]


def get_cached_vectorstore(
    persist_directory: str,
    embedding_function: Any,
    collection_name: str,
    embedding_kind: str,
    embedding_model_name: str,
    device_type: str,
) -> Chroma:
    """Return a cached Chroma vector store."""

    key = (
        persist_directory,
        collection_name,
        embedding_kind,
        embedding_model_name,
        device_type,
    )

    with _cache_lock:
        if key not in _vectorstore_cache:
            _vectorstore_cache[key] = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_function,
                collection_name=collection_name,
                client_settings=CHROMA_SETTINGS,
            )

        return _vectorstore_cache[key]


def validate_persist_directory(persist_directory: str) -> str:
    """Validate and normalize an absolute Chroma persistence path."""

    path = Path(persist_directory).expanduser()

    if not path.is_absolute():
        raise ValueError(
            "persist_directory must be an absolute path, "
            f"but received: {persist_directory!r}"
        )

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Chroma persistence directory does not exist: {path}"
        )

    if not path.is_dir():
        raise NotADirectoryError(
            f"Chroma persistence path is not a directory: {path}"
        )

    return str(path)


def validate_device_type(device_type: str) -> str:
    """Validate the requested embedding device."""

    normalized = device_type.strip().lower()

    if normalized not in {"cpu", "cuda"}:
        raise ValueError(
            "device_type must be either 'cpu' or 'cuda', "
            f"but received: {device_type!r}"
        )

    return normalized


def serialize_document(document: Document) -> Dict[str, Any]:
    """Convert a LangChain document into an MCP-serializable dictionary."""

    return {
        "page_content": document.page_content,
        "metadata": dict(document.metadata),
    }


@mcp.tool
def celloai_retrieve(
    query: str,
    persist_directory: str,
    num_code: int = 50,
    num_text: int = 10,
    text_embedding_model_name: str = (
        "Lajavaness/bilingual-embedding-large"
    ),
    code_embedding_model_name: str = (
        "Lajavaness/bilingual-embedding-large"
    ),
    device_type: str = "cpu",
) -> Dict[str, Any]:
    """
    Search an existing CelloAI Chroma index for relevant code and text.

    Use this tool when semantic retrieval is needed to discover relevant source
    code, documentation, functions, classes, or implementation details. Do not
    use it when the relevant local files are already known and can be inspected
    directly.

    For callers, callees, reachability, call paths, and dependency
    relationships, use the separate CelloAI Callgraph MCP tool.

    Args:
        query:
            Focused semantic search query. Include known function, class,
            component, or subsystem names when possible.

        persist_directory:
            Absolute path to the existing Chroma persistence directory.

        num_code:
            Maximum number of code chunks to return. Start with a small number
            and increase it only when the initial results are insufficient.

        num_text:
            Maximum number of documentation or other text chunks to return.

        text_embedding_model_name:
            Embedding model used to query the text collection. This must match
            the model used when the text collection was created.

        code_embedding_model_name:
            Embedding model used to query the code collection. This must match
            the model used when the code collection was created.

        device_type:
            Device used to calculate query embeddings. Use "cpu" by default.
            Use "cuda" only when PyTorch, the CUDA runtime, and the NVIDIA
            driver have been verified as compatible.

    Returns:
        A dictionary containing:

        - query: The submitted semantic query.
        - persist_directory: The normalized Chroma directory.
        - retrieved_documents: Retrieved chunks containing page_content and
          metadata.
        - document_count: Number of returned documents.
        - code_limit: Requested maximum number of code chunks.
        - text_limit: Requested maximum number of text chunks.

        If retrieval fails, the dictionary contains an error field and an empty
        retrieved_documents list.
    """
    # THIS IS A DEBUG PRINT TO SEE IF THIS CODE IS BEING EXECUTED
    import sys
    print("DEBUG: celloai_retrieve function called!", file=sys.stderr)
    print(f"DEBUG: query='{query}'", file=sys.stderr)
    normalized_query = query.strip()

    if not normalized_query:
        return {
            "error": "query must not be empty",
            "query": query,
            "retrieved_documents": [],
            "document_count": 0,
        }

    if num_code < 0:
        return {
            "error": "num_code must be greater than or equal to zero",
            "query": normalized_query,
            "retrieved_documents": [],
            "document_count": 0,
        }

    if num_text < 0:
        return {
            "error": "num_text must be greater than or equal to zero",
            "query": normalized_query,
            "retrieved_documents": [],
            "document_count": 0,
        }

    if num_code == 0 and num_text == 0:
        return {
            "error": "At least one of num_code or num_text must be positive",
            "query": normalized_query,
            "retrieved_documents": [],
            "document_count": 0,
        }

    try:
        normalized_directory = validate_persist_directory(
            persist_directory
        )
        normalized_device = validate_device_type(device_type)

        text_embeddings = get_cached_text_embeddings(
            text_embedding_model_name,
            normalized_device,
        )
        code_embeddings = get_cached_code_embeddings(
            code_embedding_model_name,
            normalized_device,
        )

        text_vectorstore = get_cached_vectorstore(
            persist_directory=normalized_directory,
            embedding_function=text_embeddings,
            collection_name="text_collection",
            embedding_kind="text",
            embedding_model_name=text_embedding_model_name,
            device_type=normalized_device,
        )

        code_vectorstore = get_cached_vectorstore(
            persist_directory=normalized_directory,
            embedding_function=code_embeddings,
            collection_name="code_collection",
            embedding_kind="code",
            embedding_model_name=code_embedding_model_name,
            device_type=normalized_device,
        )

        text_retriever = text_vectorstore.as_retriever(
            search_kwargs={"k": num_text}
        )
        code_retriever = code_vectorstore.as_retriever(
            search_kwargs={"k": num_code}
        )

        combined_retriever = CelloRetriever(
            code_retriever,
            text_retriever,
            num_code,
            num_text,
        )

        # BaseRetriever.invoke() is LangChain's public retrieval interface.
        retrieved_documents: List[Document] = combined_retriever.invoke(
            normalized_query
        )

        serialized_documents = [
            serialize_document(document)
            for document in retrieved_documents
        ]

        return {
            "query": normalized_query,
            "persist_directory": normalized_directory,
            "retrieved_documents": serialized_documents,
            "document_count": len(serialized_documents),
            "code_limit": num_code,
            "text_limit": num_text,
        }

    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "query": normalized_query,
            "persist_directory": persist_directory,
            "retrieved_documents": [],
            "document_count": 0,
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")
