
import asyncio

from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from rag_pipeline import load_retriever,initialize_vllm_pipeline





async def main():
    # Setup shared embedding model
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    embedding_model = HuggingFaceEmbeddings(
    model_name=MODEL_NAME,
    model_kwargs={"device": "cpu"}
)

    # 1. Initialize retriever
    retriever = load_retriever(embedding_model)

    # 2. Initialize LLM via vLLM pipeline check
    llm = initialize_vllm_pipeline()



if __name__ == "__main__":
    asyncio.run(main())