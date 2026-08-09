import os
from pathlib import Path
from docling.datamodel.base_models import InputFormat #type: ignore

# --- Environment Flags ---
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

# --- Models & Server URLs ---
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
LLM_MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")

# --- Paths & Directories ---
INDEX_PATH = "faiss_docling_index"
DEFAULT_FILE_PATH = Path("./data/paper.pdf")
OUTPUT_DIR = "outputs"
METRICS_SNAPSHOT_PATH = f"{OUTPUT_DIR}/metrics_snapshot.json"
METRICS_TIMESERIES_PATH = f"{OUTPUT_DIR}/metrics_timeseries.json"

# --- Document Chunking & Vector Store ---
CHUNK_MAX_TOKENS = 256
CHUNK_OVERLAP = 50
ALLOWED_FORMATS = [
    InputFormat.PDF,
    InputFormat.DOCX,
    InputFormat.PPTX,
    InputFormat.ASCIIDOC,
    InputFormat.MD,
    InputFormat.XLSX,
]
TOP_K = 3

# --- LLM Sampling Parameters ---
TEMPERATURE = 0.2
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0.3
REPETITION_PENALTY = 1.1

# --- Server Connection & Monitoring ---
SERVER_MAX_RETRIES = 60
SERVER_RETRY_INTERVAL = 5
POLL_INTERVAL = 0.05

METRIC_KEYS = [
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "vllm:gpu_cache_usage_perc",
    "vllm:cpu_cache_usage_perc",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
]

# --- RAG Prompt Template ---
RAG_PROMPT_TEMPLATE = """You are a highly reliable technical assistant.
Answer ONLY using the information provided in the context.
If the answer is not explicitly stated in the context, respond with:
"I don't have enough information from the provided documents to answer this."

Do NOT use prior knowledge, guess, or add details beyond what the context supports.
If pieces of context conflict, point out the conflict instead of guessing.

CONTEXT:
{context}

QUESTION:
{question}

FINAL ANSWER (based ONLY on context):"""