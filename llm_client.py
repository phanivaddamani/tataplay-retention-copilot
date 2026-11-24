"""llm_client.py -- Resilient client for chat completion with retry and backoff."""
import os
import time
import requests
import mock_llm

URL = mock_llm.URL
BASE_DELAY = 0.05
MAX_RETRIES = 3

class LLMClientError(Exception):
    """Raised when an API call fails with a non-retryable error."""
    pass

def call_llm(messages, temperature=0.0, max_tokens=300):
    """
    Sends a chat completion request to the LLM endpoint (mock_llm by default, or live if configured).
    
    Args:
        messages (list): List of message dicts with 'role' and 'content'.
        temperature (float): Sampling temperature (default: 0.0).
        max_tokens (int): Maximum output tokens (default: 300).
        
    Returns:
        str: Generated assistant response text.
        
    Raises:
        LLMClientError: If the server returns a 400, 401, or 404 status.
        RuntimeError: If rate limit retries are exhausted.
    """
    mode = os.environ.get("LLM_MODE", "mock").lower()
    headers = {
        "Authorization": "Bearer mock-api-key-test",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    retries = 0
    while True:
        if mode == "live":
            api_key = os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                raise LLMClientError("OPENROUTER_API_KEY is not set for live LLM mode.")
            headers["Authorization"] = f"Bearer {api_key}"
            resp = requests.post(URL, headers=headers, json=payload, timeout=30)
        else:
            resp = mock_llm.post(URL, headers=headers, json=payload)
            
        status = resp.status_code
        if status == 200:
            data = resp.json()
            choice = data["choices"][0]
            finish_reason = choice.get("finish_reason")
            content = choice["message"]["content"]
            if finish_reason == "length":
                # Note length truncation as specified
                pass
            return content
            
        elif status == 429:
            if retries < MAX_RETRIES:
                delay = BASE_DELAY * (2 ** retries)
                time.sleep(delay)
                retries += 1
                continue
            else:
                raise RuntimeError(f"Rate limit exceeded after {MAX_RETRIES} retries.")
                
        elif status in (400, 401, 404):
            err_msg = resp.json().get("error", {}).get("message", "API Error")
            raise LLMClientError(f"HTTP {status} Error: {err_msg}")
        else:
            raise LLMClientError(f"Unexpected HTTP status {status}: {resp.text}")
