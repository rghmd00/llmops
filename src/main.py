
import time
import asyncio

from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from rag_pipeline import load_retriever,initialize_vllm_pipeline,build_rag_chain,run_with_metrics
from utils import queries



async def main():
    

    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    llm = initialize_vllm_pipeline()
    retriever = load_retriever(embedding_model)
    chain = build_rag_chain(retriever, llm)


    print(f"Sending {len(queries)} concurrent RAG queries...\n")

    print("--- Running Batch 1 (Cache Population) ---")
    start = time.time()
    results = await run_with_metrics(chain, queries)
    elapsed = time.time() - start # All 33 completed in 5.42s


    print(f"\nAll {len(queries)} completed in {elapsed:.2f}s")
    for q, a in zip(queries, results):
        print(f"\nQ: {q}\nA: {a}")

if __name__ == "__main__":
    asyncio.run(main())