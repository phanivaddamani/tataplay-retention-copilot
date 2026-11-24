"""test_llm_client.py -- Verify LLM client demonstrations and error handling."""
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import mock_llm
from llm_client import call_llm, LLMClientError

def run_demonstrations():
    print("=== TASK 1: LLM CLIENT DEMONSTRATIONS ===")
    
    # (a) Auth and URL errors
    mock_llm.reset()
    print("\n--- Demo (a): 401 and 404 Status ---")
    resp_no_auth = mock_llm.post(mock_llm.URL, headers={}, json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    print(f"Request with no Authorization header -> status {resp_no_auth.status_code}")
    assert resp_no_auth.status_code == 401
    
    resp_wrong_url = mock_llm.post("https://wrong.url/api", headers={"Authorization": "Bearer key"}, json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    print(f"Request with wrong URL -> status {resp_wrong_url.status_code}")
    assert resp_wrong_url.status_code == 404
    
    # (b) max_tokens truncation
    mock_llm.reset()
    print("\n--- Demo (b): Finish Reason 'length' ---")
    resp_len = mock_llm.post(mock_llm.URL, headers={"Authorization": "Bearer key"}, json={"model": "m", "messages": [{"role": "user", "content": "Tell me a long story"}], "max_tokens": 3})
    finish_reason = resp_len.json()["choices"][0]["finish_reason"]
    print(f"Request with max_tokens=3 -> finish_reason='{finish_reason}'")
    assert finish_reason == "length"
    
    # (c) Statelessness demonstration
    mock_llm.reset()
    print("\n--- Demo (c): Statelessness ---")
    reply1 = call_llm([{"role": "user", "content": "What is my name?"}])
    print(f"Asking without history: '{reply1}'")
    assert "I do not know your name" in reply1
    
    reply2 = call_llm([
        {"role": "user", "content": "My name is Asha."},
        {"role": "assistant", "content": "Hello Asha! How can I help you?"},
        {"role": "user", "content": "What is my name?"}
    ])
    print(f"Asking with history: '{reply2}'")
    assert "Your name is Asha." in reply2
    
    # (d) Rate limit recovery loop (30 calls -> 3 rate limits, 33 post calls)
    mock_llm.reset()
    print("\n--- Demo (d): Rate Limit Recovery (30 calls) ---")
    rate_limits = 0
    # Monkey-patch mock_llm.post temporarily or count via mock_llm._state['calls']
    for i in range(30):
        ans = call_llm([{"role": "user", "content": f"Ping {i}"}])
    
    total_calls = mock_llm._state["calls"]
    recovered = total_calls - 30
    print(f"Completed 30 calls successfully.")
    print(f"Total post() calls invoked: {total_calls}")
    print(f"Rate-limited (429) retries recovered: {recovered}")
    assert total_calls == 33
    assert recovered == 3
    print("Demo (d) verification PASSED!")

if __name__ == "__main__":
    run_demonstrations()
