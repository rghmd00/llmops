import time
import concurrent.futures
from langchain_openai import ChatOpenAI # type: ignore
from langchain_core.prompts import PromptTemplate   # type: ignore
from langchain_classic.chains import RetrievalQA    # type: ignore
from llm_model import get_vllm_metrics

model_name = 'Qwen/Qwen3-0.6B'


def retrieval_function(llm,retriever,queries):

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    You are a highly reliable technical assistant.
    Answer ONLY using the information provided in the context.
    If the answer is not explicitly stated in the context, respond with:
    "I don't have enough information from the provided documents to answer this."

    STRICT RULES:
    - Do NOT use prior knowledge.
    - Do NOT guess or hallucinate.
    - Do NOT add missing technical details.
    - Do NOT assume beyond what the context supports.
    - Only answer if the context contains clear and direct evidence.

    --------------------
    CONTEXT:
    {context}
    --------------------

    QUESTION:
    {question}

    FINAL ANSWER (based ONLY on context):
    """
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )


    def _ask(query):
        return qa_chain.invoke({"query": query})

    before = get_vllm_metrics()
    print(f"Sending {len(queries)} concurrent RAG queries...\n")
    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = {pool.submit(_ask, q): q for q in queries}

        time.sleep(0.5)
        during = get_vllm_metrics()
        running = during.get("vllm:num_requests_running", "--")
        waiting = during.get("vllm:num_requests_waiting", "--")
        print(f"  [mid-flight]  running: {running}  |  waiting: {waiting}")

        results = {}
        for f in concurrent.futures.as_completed(futures):
            q = futures[f]
            resp = f.result()
            results[q] = resp["result"]
            print(f"  done: \"{q[:50]}...\"")

    elapsed = time.time() - start
    after = get_vllm_metrics()
    tokens = after.get("vllm:generation_tokens_total", 0) - before.get(
        "vllm:generation_tokens_total", 0)

    print(f"\nAll {len(queries)} completed in {elapsed:.2f}s")
    if tokens > 0:
        print(f"Tokens generated: {tokens:g}  |  ~{tokens / elapsed:.1f} tokens/s")

    for q, answer in results.items():
        print(f"\nQ: {q}\nA: {answer}")