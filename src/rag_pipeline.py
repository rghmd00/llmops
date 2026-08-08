import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import requests
from docling.datamodel.base_models import InputFormat  # type: ignore
from docling.document_converter import DocumentConverter  # type: ignore
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker  # type: ignore
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_core.documents import Document  # type: ignore
from langchain_core.output_parsers import StrOutputParser  # type: ignore
from langchain_core.prompts import PromptTemplate  # type: ignore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_openai import ChatOpenAI  # type: ignore
from transformers import AutoTokenizer  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants & Configurations
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss_docling_index"
DEFAULT_FILE_PATH = Path("./data/paper.pdf")

DEFAULT_VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
DEFAULT_LLM_MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
OUTPUT_DIR = "outputs"

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


# --- 1. Document Ingestion & VectorStore Functions ---

def process_and_chunk_document(file_path: Path, model_name: str = MODEL_NAME) -> List[Document]:
    """Converts a document using Docling and chunks it into LangChain Documents."""
    logger.info("Initializing converter and tokenizer...")
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(model_name)
    )

    doc_converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.PPTX,
            InputFormat.ASCIIDOC,
            InputFormat.MD,
            InputFormat.XLSX,
        ],
    )

    logger.info("Processing document: %s", file_path)
    conv_result = doc_converter.convert(file_path)
    source_name = conv_result.input.file.name
    docling_document = conv_result.document

    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=256, overlap=50)
    documents: List[Document] = []

    for doc_id, chunk in enumerate(chunker.chunk(docling_document), start=1):
        refs = " ".join(item.get_ref().cref for item in chunk.meta.doc_items)
        
        documents.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "doc_id": doc_id,
                    "source": source_name,
                    "ref": refs,
                },
            )
        )

    logger.info("Created %d chunks.", len(documents))
    return documents


def build_and_save_vectorstore(
    documents: List[Document], 
    embedding_model: HuggingFaceEmbeddings, 
    save_path: str = INDEX_PATH
) -> FAISS:
    """Creates a FAISS vectorstore from documents and saves it locally."""
    logger.info("Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(save_path)
    logger.info("Vectorstore saved to '%s'", save_path)
    return vectorstore


def load_retriever(
    embedding_model: HuggingFaceEmbeddings, 
    load_path: str = INDEX_PATH, 
    top_k: int = 5
):
    """Loads a saved FAISS index or creates one if missing, then returns a retriever instance."""
    if not os.path.exists(load_path):
        logger.info("Index directory '%s' not found. Indexing default document first...", load_path)
        documents = process_and_chunk_document(DEFAULT_FILE_PATH)
        build_and_save_vectorstore(documents, embedding_model, save_path=load_path)

    logger.info("Loading vectorstore from '%s'...", load_path)
    vectorstore = FAISS.load_local(
        load_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )


# --- 2. vLLM Server & LLM Client Setup ---

def ensure_output_directory(path: str = OUTPUT_DIR) -> None:
    """Ensures local outputs directory exists."""
    os.makedirs(path, exist_ok=True)


def wait_for_vllm_server(
    vllm_url: str = DEFAULT_VLLM_URL,
    max_retries: int = 60,
    retry_interval: int = 5,
) -> str:
    """Polls the vLLM server until reachable and returns loaded model ID."""
    logger.info("Waiting for vLLM server at %s...", vllm_url)
    models_endpoint = f"{vllm_url}/v1/models"

    for attempt in range(max_retries):
        try:
            response = requests.get(models_endpoint, timeout=5)
            if response.status_code == 200:
                model_id = response.json()["data"][0]["id"]
                logger.info("Connected to %s — model: %s", vllm_url, model_id)
                return model_id
        except requests.RequestException:
            pass

        time.sleep(retry_interval)
        if attempt % 6 == 5:
            logger.info("Still waiting... (%ds elapsed)", (attempt + 1) * retry_interval)

    raise RuntimeError(f"vLLM server at {vllm_url} not reachable after {max_retries * retry_interval} seconds.")


def create_llm_client(
    model_name: str = DEFAULT_LLM_MODEL_NAME,
    vllm_url: str = DEFAULT_VLLM_URL,
    temperature: float = 0.0,
    disable_thinking: bool = True,
) -> ChatOpenAI:
    """Constructs and returns a ChatOpenAI client pointed at the vLLM instance."""
    extra_body = {}
    if disable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    logger.info("Initializing ChatOpenAI client for model: %s", model_name)
    
    return ChatOpenAI(
        model=model_name,
        base_url=f"{vllm_url}/v1",
        api_key="unused",
        temperature=temperature,
        extra_body=extra_body if extra_body else None,  # Passed directly here
    )

def initialize_vllm_pipeline(
    vllm_url: str = DEFAULT_VLLM_URL,
    model_name: Optional[str] = None,
    disable_thinking: bool = True,
) -> ChatOpenAI:
    """Orchestrates output directory check, server healthcheck, and client initialization."""
    ensure_output_directory()
    server_model_id = wait_for_vllm_server(vllm_url=vllm_url)
    target_model = model_name or server_model_id
    
    return create_llm_client(
        model_name=target_model,
        vllm_url=vllm_url,
        disable_thinking=disable_thinking,
    )


# --- 3. Async LCEL Execution Pipeline ---

def format_docs(docs: List[Document]) -> str:
    """Helper to join retrieved documents into context string."""
    return "\n\n".join(doc.page_content for doc in docs)
