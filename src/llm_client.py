"""llm_client.py -- Resilient client for chat completion with retry and backoff."""
import os
import sys

# Ensure root directory is on sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from llm_client import LLMClientError, call_llm, URL, BASE_DELAY, MAX_RETRIES
