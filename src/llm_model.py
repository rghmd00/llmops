import logging
import os
import sys
import time
from typing import Optional

import requests
from langchain_openai import ChatOpenAI  #type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants / Defaults
DEFAULT_VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
OUTPUT_DIR = "outputs"

# Environment configuration
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"


def ensure_output_directory(path: str = OUTPUT_DIR) -> None:
    """Ensures local outputs directory exists."""
    os.makedirs(path, exist_ok=True)


def wait_for_vllm_server(
    vllm_url: str = DEFAULT_VLLM_URL,
    max_retries: int = 60,
    retry_interval: int = 5,
) -> str:
    """
    Polls the vLLM server until it is reachable and returns the loaded model ID.
    Raises RuntimeError if server does not respond within the maximum retry limit.
    """
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
    model_name: str = DEFAULT_MODEL_NAME,
    vllm_url: str = DEFAULT_VLLM_URL,
    temperature: float = 0.0,
    disable_thinking: bool = True,
) -> ChatOpenAI:
    """
    Constructs and returns a ChatOpenAI client pointed at the local vLLM instance.
    Optionally disables Qwen3 thinking/reasoning mode via chat template parameters.
    """
    extra_body = {}
    if disable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}

    logger.info("Initializing ChatOpenAI client for model: %s", model_name)
    
    return ChatOpenAI(
        model=model_name,
        base_url=f"{vllm_url}/v1",
        api_key="unused",
        temperature=temperature,
        model_kwargs={"extra_body": extra_body} if extra_body else {},
    )


def initialize_vllm_pipeline(
    vllm_url: str = DEFAULT_VLLM_URL,
    model_name: Optional[str] = None,
    disable_thinking: bool = True,
) -> ChatOpenAI:
    """Orchestrates directory creation, health check polling, and LLM client creation."""
    ensure_output_directory()
    server_model_id = wait_for_vllm_server(vllm_url=vllm_url)
    
    # Use served model ID if explicit model name was not provided
    target_model = model_name or server_model_id
    
    return create_llm_client(
        model_name=target_model,
        vllm_url=vllm_url,
        disable_thinking=disable_thinking,
    )


if __name__ == "__main__":
    # Example execution when run directly as a script
    llm = initialize_vllm_pipeline()
    logger.info("LLM Client ready for invocations.")