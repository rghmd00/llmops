
import asyncio

from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from rag_pipeline import load_retriever,initialize_vllm_pipeline
from retriever import retrieval_function




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


    queries = [
    "What are the three distinct RAG paradigms defined in the survey paper, and how do their workflows differ?",
    "What is the difference between naive RAG and advanced RAG?",
    "How does retrieval augmentation help reduce hallucination?",
    "What evaluation metrics are used for RAG systems in the survey?",
    "What are the main components of a modular RAG architecture?",
    ]


    retrieval_function(llm=llm,retriever=retriever,queries=queries)



if __name__ == "__main__":
    asyncio.run(main())