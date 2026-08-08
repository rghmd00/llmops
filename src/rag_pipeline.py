# vllm serve ./models/Qwen3-0.6B-W4A16 \
#   --quantization compressed-tensors \
#   --gpu-memory-utilization 0.85 \
#   --max-model-len 4096 \
#   --enable-prefix-caching


import asyncio
import logging
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional

import requests
from docling.datamodel.base_models import InputFormat   #type: ignore
from docling.document_converter import DocumentConverter #type: ignore
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker  #type: ignore
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer  #type: ignore
from langchain_community.vectorstores import FAISS  #type: ignore
from langchain_core.documents import Document   #type: ignore
from langchain_core.output_parsers import StrOutputParser   #type: ignore
from langchain_core.prompts import PromptTemplate   #type: ignore
from langchain_core.runnables import RunnableParallel, RunnablePassthrough  #type: ignore
from langchain_huggingface import HuggingFaceEmbeddings     #type: ignore
from langchain_openai import ChatOpenAI #type: ignore
from transformers import AutoTokenizer

 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
 
# --- Constants ---
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "faiss_docling_index"
DEFAULT_FILE_PATH = Path("./data/paper.pdf")
 
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
LLM_MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
OUTPUT_DIR = "outputs"
TOP_K = 3  # keep small for a small model — less context to lose coherence in
 
# Sampling params. temperature=0 (pure greedy) + no penalty is what causes
# small models to fall into "edge edge edge ..." repetition loops.
TEMPERATURE = 0.2
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0.3
REPETITION_PENALTY = 1.1  # vLLM-native, passed via extra_body
 
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
 
 
# --- 1. Document Ingestion & VectorStore ---
 
def process_and_chunk_document(file_path: Path, model_name: str = EMBED_MODEL_NAME) -> List[Document]:
    """Converts a document using Docling and chunks it into LangChain Documents."""
    tokenizer = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained(model_name))
    doc_converter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX,
            InputFormat.ASCIIDOC, InputFormat.MD, InputFormat.XLSX,
        ],
    )
 
    logger.info("Processing document: %s", file_path)
    conv_result = doc_converter.convert(file_path)
    docling_document = conv_result.document
 
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=256, overlap=50)
    documents = [
        Document(
            page_content=chunk.text,
            metadata={
                "doc_id": doc_id,
                "source": conv_result.input.file.name,
                "ref": " ".join(item.get_ref().cref for item in chunk.meta.doc_items),
            },
        )
        for doc_id, chunk in enumerate(chunker.chunk(docling_document), start=1)
    ]
 
    logger.info("Created %d chunks.", len(documents))
    return documents
 
 
def load_retriever(
    embedding_model: HuggingFaceEmbeddings,
    load_path: str = INDEX_PATH,
    top_k: int = TOP_K,
):
    """Loads a saved FAISS index, building it from the default file if missing."""
    if not os.path.exists(load_path):
        logger.info("Index '%s' not found. Building from %s", load_path, DEFAULT_FILE_PATH)
        documents = process_and_chunk_document(DEFAULT_FILE_PATH)
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(load_path)
    else:
        logger.info("Loading vectorstore from '%s'", load_path)
        vectorstore = FAISS.load_local(load_path, embedding_model, allow_dangerous_deserialization=True)
 
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
 
 
# --- 2. vLLM Server & LLM Client ---
 
def wait_for_vllm_server(vllm_url: str = VLLM_URL, max_retries: int = 60, retry_interval: int = 5) -> str:
    """Polls the vLLM server until reachable and returns the loaded model ID."""
    logger.info("Waiting for vLLM server at %s...", vllm_url)
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{vllm_url}/v1/models", timeout=5)
            if response.status_code == 200:
                model_id = response.json()["data"][0]["id"]
                logger.info("Connected to %s — model: %s", vllm_url, model_id)
                return model_id
        except requests.RequestException:
            pass
        if attempt % 6 == 5:
            logger.info("Still waiting... (%ds elapsed)", (attempt + 1) * retry_interval)
        time.sleep(retry_interval)
 
    raise RuntimeError(f"vLLM server at {vllm_url} not reachable after {max_retries * retry_interval} seconds.")
 
 
def create_llm_client(
    model_name: str = LLM_MODEL_NAME,
    vllm_url: str = VLLM_URL,
    disable_thinking: bool = True,
) -> ChatOpenAI:
    """Constructs a ChatOpenAI client pointed at vLLM, with sampling params
    that avoid greedy-decoding repetition loops (see constants above)."""
    extra_body = {"repetition_penalty": REPETITION_PENALTY}
    if disable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False} #type: ignore
 
    logger.info("Initializing ChatOpenAI client for model: %s", model_name)
    return ChatOpenAI(
        model=model_name,
        base_url=f"{vllm_url}/v1",
        api_key="unused",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        frequency_penalty=FREQUENCY_PENALTY,
        extra_body=extra_body,
    )
 
 
def initialize_vllm_pipeline(
    vllm_url: str = VLLM_URL,
    model_name: Optional[str] = None,
    disable_thinking: bool = True,
) -> ChatOpenAI:
    """Ensures output dir exists, waits for the server, and builds the LLM client."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    server_model_id = wait_for_vllm_server(vllm_url=vllm_url)
    return create_llm_client(
        model_name=model_name or server_model_id,
        vllm_url=vllm_url,
        disable_thinking=disable_thinking,
    )
 
 
def get_vllm_metrics(base_url: str = VLLM_URL) -> dict:
    """Scrape vLLM Prometheus /metrics and return {name: value}."""
    r = requests.get(f"{base_url}/metrics")
    metrics = {}
    for line in r.text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split()[0]
        try:
            metrics[name] = float(line.split()[-1])
        except (ValueError, IndexError):
            continue
    return metrics
 
 
METRIC_KEYS = [
    "vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:num_requests_swapped",
    "vllm:gpu_cache_usage_perc", "vllm:cpu_cache_usage_perc",
    "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
]
 
 
def print_metrics(metrics: dict, save_path: str = f"{OUTPUT_DIR}/metrics_snapshot.json") -> None:
    """Prints key vLLM metrics and saves the full snapshot to disk."""
    print("Current vLLM Metrics:")
    for key in METRIC_KEYS:
        if key in metrics:
            print(f"  {key.replace('vllm:', '')}: {metrics[key]:g}")
 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  (full snapshot saved to {save_path})")
 
 
# --- 3. RAG Chain ---
 
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a highly reliable technical assistant.
Answer ONLY using the information provided in the context.
If the answer is not explicitly stated in the context, respond with:
"I don't have enough information from the provided documents to answer this."
 
Do NOT use prior knowledge, guess, or add details beyond what the context supports.
If pieces of context conflict, point out the conflict instead of guessing.
 
CONTEXT:
{context}
 
QUESTION:
{question}
 
FINAL ANSWER (based ONLY on context):""",
)
 
 
def format_docs(docs: List[Document]) -> str:
    """Helper to join retrieved documents into context string."""
    return "\n\n".join(doc.page_content for doc in docs)
 
 
def build_rag_chain(retriever, llm: ChatOpenAI):
    """LCEL chain: retrieve -> format -> prompt -> llm -> parse string."""
    return (
        RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
 


async def run_with_metrics(chain, queries: List[str], poll_interval: float = 0.05) -> List[str]:
    """Runs chain.abatch(queries) while continuously polling vLLM metrics
    until the batch completes, so you can see continuous batching
    (num_requests_running/waiting), KV-cache pressure, and prefix-cache
    hit rate as they actually evolve — not just a single snapshot."""
    before = get_vllm_metrics()
    task = asyncio.create_task(chain.abatch(queries))
 
    samples = []
    while not task.done():
        m = get_vllm_metrics()
        samples.append({
            "t": round(time.time(), 3),
            "running": m.get("vllm:num_requests_running", 0.0),
            "waiting": m.get("vllm:num_requests_waiting", 0.0),
            "gpu_cache_usage_perc": m.get("vllm:gpu_cache_usage_perc", 0.0),
        })
        await asyncio.sleep(poll_interval)
 
    results = await task
    after = get_vllm_metrics()
 
    print_batch_summary(samples, before, after)
    return results
 
 
def print_batch_summary(samples: List[dict], before: dict, after: dict) -> None:
    """Prints continuous-batching and prefix-caching behavior for one abatch() call."""
    print()
    if samples:
        running_vals = [s["running"] for s in samples]
        waiting_vals = [s["waiting"] for s in samples]
        cache_vals = [s["gpu_cache_usage_perc"] for s in samples]
        print(f"  Polled {len(samples)} times during the batch:")
        print(f"    peak requests running:  {max(running_vals):g}")
        print(f"    peak requests waiting:  {max(waiting_vals):g}")
        print(f"    peak GPU KV-cache use:  {max(cache_vals) * 100:.1f}%")
    else:
        print("  Batch finished faster than one poll interval — nothing sampled mid-flight.")
        print("  (Lower poll_interval, or use more/longer queries, to catch it in the act.)")
 
    # Prefix caching — metric names vary across vLLM versions (e.g.
    # "prefix_cache" vs "prefix_caching"), so match on the "prefix_cach"
    # stem rather than the full word.
    prefix_keys = sorted(k for k in after if "prefix_cach" in k)
    if prefix_keys:
        print("\n  Prefix caching:")
        for k in prefix_keys:
            delta = after[k] - before.get(k, 0)
            label = k.replace("vllm:", "")
            # rate/perc-style metrics are already ratios — a raw delta is meaningless for those
            if "rate" in k or "perc" in k:
                print(f"    {label}: {after[k]:.1%}" if after[k] <= 1 else f"    {label}: {after[k]:g}")
            else:
                print(f"    {label}: {after[k]:g}  (+{delta:g} this batch)")
 
        hits_key = next((k for k in prefix_keys if "hits_total" in k), None)
        queries_key = next((k for k in prefix_keys if "queries_total" in k), None)
        if hits_key and queries_key:
            d_hits = after[hits_key] - before.get(hits_key, 0)
            d_queries = after[queries_key] - before.get(queries_key, 0)
            if d_queries > 0:
                print(f"    batch hit rate: {d_hits / d_queries:.1%}")
    else:
        # Nothing matched the expected stem — dump anything cache-related
        # so you can see the real key names instead of guessing.
        cache_keys = sorted(k for k in after if "cache" in k)
        if cache_keys:
            print("\n  No metric matched 'prefix_cach*' — here's everything cache-related found instead:")
            for k in cache_keys:
                print(f"    {k}: {after[k]:g}")
        else:
            print("\n  No prefix-cache metrics found on this server.")
            print("  Prefix caching must be enabled when vLLM is launched: --enable-prefix-caching")
 
    tokens = after.get("vllm:generation_tokens_total", 0) - before.get("vllm:generation_tokens_total", 0)
    if tokens:
        print(f"\n  Tokens generated this batch: {tokens:g}")
 
    print_metrics(after)
 
    history_path = f"{OUTPUT_DIR}/metrics_timeseries.json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  (time series saved to {history_path})")