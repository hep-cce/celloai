# launch LlamaCpp server with
# ./llama.cpp/build-08152025/bin/llama-server -m /data/llms/model--unsloth--gpt-oss-120b-GGUF/gpt-oss-120b-F16.gguf --port 8000 -c 131072 --n-gpu-layers 99 -ot ".ffn_(up|down)_exps.=CPU" &

from __future__ import annotations

import os, sys
import json
import time
from typing import List, Optional, Tuple
import requests  # type: ignore

celloai_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, celloai_path)
from retrieval_pipeline import retrieve_docs_enhance_prompt_from_cello
from prompt_template import doxygen_prompt, chatbot_prompt 

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# OpenAI‑compatible streaming server llama.cpp

def openai_stream(url: str, payload: dict) -> Tuple[str, float, Optional[float], int]:
    with requests.post(url, json=payload, stream=True, timeout=None) as resp:
        resp.raise_for_status()
        text_parts: List[str] = []
        usage = None
        start = time.time()
        ttft: Optional[float] = None
        for line in resp.iter_lines():
            if not line:
                continue
            if not line.startswith(b"data: "):
                continue
            raw = line[6:]
            if raw == b"[DONE]":
                break
            chunk = json.loads(raw)
            if ttft is None and chunk["choices"][0]["delta"].get("content"):
                ttft = time.time() - start
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                text_parts.append(delta["content"])
            if "usage" in chunk:
                usage = chunk["usage"]
        latency = time.time() - start
        text_parts = [tp for tp in text_parts if tp is not None]
        if len(text_parts) > 0:
            text = "".join(text_parts).strip()
        else:
            text = ""
        completion_tokens = usage.get("completion_tokens") if usage else len(text.split())
        return text, latency, ttft, completion_tokens


def openai_nonstream(url: str, payload: dict) -> Tuple[str, float, Optional[float], int]:
    start = time.time()
    resp = requests.post(url, json=payload, timeout=None)
    latency = time.time() - start
    #print("Status:", resp.status_code, resp.headers.get("content-type"))
    #print("Body:", resp.text)  # read server's explanation
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", len(text.split()))
    return text, latency, None, completion_tokens


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def query_llamacpp_server_with_rag(
    user_query: str,
    history: List[dict],
    system_prompt: str
) -> str:
    """
    Queries the LLM with a prompt augmented by context and conversation history.
    """
    user_query, retrieved_docs = retrieve_docs_enhance_prompt_from_cello(user_query, history)
    
    context_str = "\n\n".join(d.page_content for d in retrieved_docs)
    prompt_text = f"""Based on the following context, please answer the question.

Context:
- {context_str}

Question: {user_query}
"""
    #print(f"\n--- Augmented Prompt ---\n{prompt_text}")
    print(f"\n--- Augmented Prompt ---\n{user_query}")

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt_text})

    #print("MESSAGES", messages)
    # --- LLM Inference ---
    host = "localhost"
    port = 8000
    max_tokens = 32768
    temperature = 0.1
    no_stream = True # Using non-streaming for simplicity in this example
    
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    stream_fn = openai_stream
    nonstream_fn = openai_nonstream

    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))
    if no_stream:
        text, latency, ttft, tokens = nonstream_fn(url, payload)
    else:
        text, latency, ttft, tokens = stream_fn(url, payload)
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time()))

    tps_wall = tokens / latency if latency else 0.0
    tps_post = tokens / (latency - ttft) if ttft and latency - ttft > 0 else None

    # ---- Output ----
    print("\n=== Completion Report ===\n")
    print(f"Start = {start_time} to End = {end_time}")
    print("\n=== Stats ===")
    print(f"Latency (wall): {latency:.3f} s")
    if ttft is not None:
        print(f"Time‑to‑first‑token: {ttft:.3f} s")
    else:
        print("Time‑to‑first‑token: N/A (non‑stream)")
    print(f"Completion tokens (approx): {tokens}")
    print(f"Throughput (tokens/s): {tps_wall:.2f}")

    return text


if __name__ == "__main__":
    
    conversation_history = []
    system_prompt = chatbot_prompt
    print(f"\nSystem Prompt: {system_prompt}")
    print("Enter your query below. Type 'exit' or 'quit' to end the conversation.")

    while True:
        try:
            user_query = input("\n> You: ")
            if user_query.lower() in ["exit", "quit"]:
                break

            assistant_response = query_llamacpp_server_with_rag(
                user_query=user_query,
                history=conversation_history,
                system_prompt=system_prompt
            )

            print(f"\n> Assistant: {assistant_response}")

            # Update conversation history
            conversation_history.append({"role": "user", "content": user_query})
            conversation_history.append({"role": "assistant", "content": assistant_response})

        except (KeyboardInterrupt, EOFError):
            break
    print("\nConversation ended.")
