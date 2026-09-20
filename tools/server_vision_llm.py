import base64
import mimetypes
import os
from pathlib import Path

import httpx
from fastmcp import FastMCP

mcp = FastMCP(
    "BRAIN Vision",
    instructions=(
        "This server provides visual analysis for text-only primary models. "
        "Use analyze_image for screenshots, plots, charts, photographs, "
        "diagrams, and other images. Pass an absolute local path and a "
        "focused question. The tool returns text only."
    ),
)

BNL_URL = "https://inference0-api.sdcc.bnl.gov/v1/chat/completions"

@mcp.tool
async def analyze_image(image_path: str, question: str) -> str:
    """
    Analyze a local PNG, JPEG, WebP, or GIF using vision LLM.
    Call this tool whenever visual understanding is required for an image,
    screenshot, plot, chart, photograph, or diagram.

    Args:
        image_path: Path to the image.
        question: Focused question about the visual content.

    """
    path = Path(image_path).expanduser().resolve()

    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
        raise ValueError("Supported formats: PNG, JPEG, WebP, GIF")

    image_data = base64.b64encode(path.read_bytes()).decode("ascii")
    data_url = f"data:{mime_type};base64,{image_data}"

    payload = {
        "model": "muse-glimmer-30b",
        "max_tokens": 65536,
        "temperature": 1.0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    }

    if not os.getenv("BRAIN_API_KEY"):
        raise RuntimeError("OpenCode did not pass BNL_API_KEY to the MCP server")

    headers = {
        "Authorization": f"Bearer {os.environ['BRAIN_API_KEY']}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(BNL_URL, json=payload, headers=headers)
        response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]

if __name__ == "__main__":
    mcp.run(transport="stdio")
