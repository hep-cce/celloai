import os
from openai import OpenAI

access_token = os.getenv("ALCF_ACCESS_TOKEN")

def query_alcf(
    payload: dict
) -> str:
    """
    Queries ALCF
    """
    # Sophia cluster
    client = OpenAI(
        api_key=access_token,
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
    )

    response = client.chat.completions.create(
        #model="openai/gpt-oss-20b",
        #model="openai/gpt-oss-120b",
        #model="meta-llama/Llama-3.3-70B-Instruct", 
        #model="google/gemma-4-31B-it"
        model="mistralai/Mistral-Large-Instruct-2407", 
        #model="meta-llama/Llama-4-Maverick-17B-128E-Instruct", #openai.InternalServerError: Error code: 503 - Error: Endpoint sophia-vllm-meta-llamallama-4-maverick-17b-128e-instruct online but not ready to receive tasks. Please try again later. 
        messages=payload["messages"],
        temperature=payload["temperature"],
    )
    print("A", response)
    print("R", response.choices[0].message.reasoning_content)
    print("D", response.choices[0].message.content)
    
    return response.choices[0].message.content


