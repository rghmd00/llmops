# High-Performance RAG & vLLM Benchmarking Pipeline

An end-to-end Retrieval-Augmented Generation (RAG) pipeline built with **LangChain**, **Docling**, **FAISS**, and **vLLM**. 

This project features built-in real-time telemetry and profiling tools designed to measure **client-side retrieval concurrency** and **server-side vLLM performance** (including KV-cache usage, throughput, and prefix-caching hit rates).

---

## ✨ Features

- **Document Processing & Chunking:** Uses [Docling](https://github.com/DS4SD/docling)'s `DocumentConverter` and `HybridChunker` with HuggingFace tokenization for structure-aware document parsing.
- **Vector Retrieval:** Powered by **FAISS** and `sentence-transformers/all-MiniLM-L6-v2` embeddings for fast similarity search.
- **High-Throughput LLM Serving:** Powered by **vLLM** running a 4-bit quantized model (`Qwen3-0.6B-W4A16`) with **prefix-caching** enabled.
- **Retrieval Concurrency Profiler:** Custom thread-safe `RetrievalProfiler` using a **sweep-line algorithm** to measure peak overlapping retrieval requests and identify async/IO bottlenecks.
- **vLLM Live Telemetry:** Asynchronous polling of vLLM's `/metrics` endpoint during batch execution to capture real-time GPU KV-cache usage, request queues (running/waiting), and prefix-cache hit rates.

---

## 🛠️ Tech Stack

* **LLM Engine:** [vLLM](https://github.com/vllm-project/vllm)
* **Framework:** LangChain (LCEL)
* **Document Parsing:** Docling & Docling Core
* **Embeddings & Vector Database:** HuggingFace Transformers, FAISS
* **Async Runtime:** Python `asyncio`
