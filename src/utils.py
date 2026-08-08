import requests



# queries to test if GPU is well utilizied
queries = [
    # Core paradigms
    "What are the three RAG paradigms and how do they differ?",
    "What defines the Naive RAG 'Retrieve-Read' framework?",
    "What pre-retrieval and post-retrieval strategies does Advanced RAG use?",
    "What new modules does Modular RAG introduce beyond Naive and Advanced RAG?",
    "What are some new patterns enabled by Modular RAG's flexible architecture?",

    # Retrieval
    "What are the main types of retrieval sources used in RAG?",
    "What challenges arise when retrieving from semi-structured data like PDFs?",
    "What retrieval granularities are discussed, from fine to coarse?",
    "What chunking strategies are used during the indexing phase?",
    "What is the Small2Big chunking approach?",
    "How can metadata attachments improve retrieval filtering?",
    "What is a hierarchical index structure and how does it help retrieval?",
    "How does a Knowledge Graph index reduce hallucination in retrieval?",
    "What is query expansion and how do Multi-Query and Sub-Query differ?",
    "What is HyDE and how does it transform queries?",
    "What is Step-back Prompting and how is it used in query transformation?",
    "What is query routing, and what's the difference between metadata and semantic routers?",
    "What embedding models are mentioned, and what is the MTEB leaderboard used for?",
    "Why would you fine-tune an embedding model instead of using a pretrained one?",

    # Generation
    "Why is it problematic to feed all retrieved documents directly into an LLM?",
    "What is reranking and what methods are used to perform it?",
    "How does LLMLingua compress prompts for context selection?",
    "What is the 'Filter-Reranker' paradigm proposed by Ma et al.?",

    # Augmentation process
    "What is the difference between iterative, recursive, and adaptive retrieval?",
    "How does Self-RAG use reflection tokens to control retrieval?",
    "How does FLARE decide when to trigger retrieval during generation?",

    # Evaluation
    "What are the three primary quality scores used to evaluate RAG systems?",
    "What are the four required abilities evaluated in RAG robustness testing?",
    "What evaluation frameworks and benchmarks are compared in the paper, and what do they measure?",

    # Challenges / future directions
    "Why is RAG still useful even as LLM context windows grow very large?",
    "What does the paper say about scaling laws and their applicability to RAG?",

    # Edge cases — should trigger "not enough information" per your strict prompt
    "What did the paper say about the stock price of companies using RAG?",
    "Does the survey recommend a specific programming language for implementing RAG?",
]





def get_vllm_metrics(base_url):
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

