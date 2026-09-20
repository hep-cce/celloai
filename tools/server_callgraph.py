import json
import sys
from pathlib import Path
from fastmcp import FastMCP

# Add the parent directory's src to the path to import retrieval_pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval_pipeline import load_function_graph, get_function_relationships

mcp = FastMCP(
    "CelloAI Callgraph",
    instructions=(
        "Use this server to answer questions about callers, callees, call "
        "paths, symbol reachability, and code dependencies."
    ),
)

@mcp.tool
def celloai_callgraph(function_name: str, merged_graph_json_path: str, depth: int = 1) -> dict:
    """
    Retrieve callgraph information for a given function.

    Args:
        function_name: The name of the function to look up.
        merged_graph_json_path: Path to the merged_graph.json file.
        depth: Depth of callers/callees to retrieve (currently only depth=1 is supported).

    Returns:
        A dictionary containing:
          - calls: list of functions called by the input function
          - called_by: list of functions that call the input function
          - formatted_text: a formatted string suitable for inclusion in a prompt
    """
    # Load the graph
    graph = load_function_graph(merged_graph_json_path)
    if graph is None:
        return {"error": f"Failed to load graph from {merged_graph_json_path}"}

    # Get relationships for the function
    relationships = get_function_relationships(function_name, graph)
    if not relationships:
        return {"error": f"No relationships found for function {function_name}"}

    # We take the first relationship (there might be multiple matches?)
    rel = relationships[0]

    calls = rel.get("calls", [])
    called_by = rel.get("called_by", [])

    # Format the text similarly to the retrieval_pipeline
    formatted_text = f"\n```{function_name}```"
    for called_func in calls:
        cf = called_func.split("::")[-1] if "::" in called_func else called_func
        if len(cf) < 3:
            cf = called_func
        if len(cf) > 3:
            formatted_text += f'\n  - ```{cf}```'
            # Note: we are not collecting patterns for simplicity
    formatted_text += '\n  and is called by\n'
    for caller in called_by:
        cf = caller.split("::")[-1] if "::" in caller else caller
        if len(cf) < 3:
            cf = caller
        if len(cf) > 3:
            formatted_text += f'\n  - ```{cf}```'

    return {
        "calls": calls,
        "called_by": called_by,
        "formatted_text": formatted_text.strip()
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
