#!/usr/bin/env python3
"""
Merge DOT files script - merges callgraphs from multiple DOT files 
with proper handling of node labels, edge directions, and JSON output.
"""

import networkx as nx
import json
import os
import subprocess
import re
import argparse
import glob
from typing import List, Dict, Tuple, Optional, Any

def get_dot_files_from_directory(directory: str) -> List[str]:
    """
    Get all DOT files from a directory
    
    Args:
        directory: Path to directory containing DOT files
        
    Returns:
        List of paths to DOT files
    """
    # Find all .dot files in the directory
    #dot_files = glob.glob(os.path.join(directory, "*.dot"), recursive=True)
    dot_files = []
    dir_path = directory + r'/**/*.dot'
    for file in glob.glob(dir_path, recursive=True):
        if "struct" not in file and "class" not in file:
            dot_files.append(file)

    if not dot_files:
        print(f"No DOT files found in directory: {directory}")
        return []
    
    print(f"Found {len(dot_files)} DOT files in {directory}")
    return dot_files


def merge_dot_files_enhanced(output_dir: str = '.') -> Tuple[nx.DiGraph, Dict[str, Any], Optional[str]]:
    """
    Merge DOT files with proper handling of node labels, edge directions, and JSON output

    Args:
        dot_files: List of paths to DOT files
        output_dir: Directory where output files will be saved

    Returns:
        Tuple containing:
        - merged_graph: The NetworkX DiGraph object
        - stats: Dictionary with graph statistics
        - png_path: Path to the generated PNG file (or None if generation failed)
    """
    dot_files = get_dot_files_from_directory(output_dir)
    #print(dot_files)
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file paths
    output_dot = os.path.join(output_dir, 'merged_graph.dot')
    output_png = os.path.join(output_dir, 'merged_graph.png')
    output_json = os.path.join(output_dir, 'merged_graph.json')
    
    # Create a new directed graph
    merged_graph = nx.DiGraph()
    
    # Track function name sources and metadata
    function_sources = {}
    file_info = {}
    
    # First pass: collect all function names from labels
    for file_index, dot_file in enumerate(dot_files):
        source_name = os.path.basename(dot_file)
        is_inverse = "icgraph" in source_name
        
        # Read raw content to extract graph directions
        try:
            with open(dot_file, 'r') as f:
                dot_content = f.read()
                
            # Extract graph direction
            rankdir_match = re.search(r'rankdir\s*=\s*"([^"]+)"', dot_content)
            rankdir = rankdir_match.group(1) if rankdir_match else 'LR'
            
            # Store file metadata
            file_info[source_name] = {
                "path": dot_file,
                "is_inverse": is_inverse,
                "rankdir": rankdir,
                "needs_inversion": is_inverse or rankdir == 'RL'
            }
            
            # Load the graph
            graph = nx.drawing.nx_pydot.read_dot(dot_file)
            
            # Extract function names from labels
            for node, attrs in graph.nodes(data=True):
                if 'label' in attrs:
                    # Get the actual function name from the label
                    # Need to replace :: with __ due to Pydot error
                    func_name = attrs['label']#.strip('"').replace("::","__")
            
                    # Track source information
                    if func_name not in function_sources:
                        function_sources[func_name] = [source_name]
                    elif source_name not in function_sources[func_name]:
                        function_sources[func_name].append(source_name)
        except Exception as e:
            print(f"Error in first pass for {dot_file}: {str(e)}")
    
    # Second pass: build the merged graph with proper edge directions
    node_id_counter = 1
    func_to_node_id = {}  # Map function names to unique node IDs
    
    for file_index, dot_file in enumerate(dot_files):
        source_name = os.path.basename(dot_file)
        file_metadata = file_info[source_name]
        needs_inversion = file_metadata["needs_inversion"]
        
        try:
            # Load the graph
            graph = nx.drawing.nx_pydot.read_dot(dot_file)
            
            # Create mapping from node IDs to function names in this graph
            node_to_func = {}
            for node, attrs in graph.nodes(data=True):
                if 'label' in attrs:
                    func_name = attrs['label']#.strip('"').replace("::","__")
                    node_to_func[node] = func_name
                    
                    # Generate a unique node ID for this function if needed
                    if func_name not in func_to_node_id:
                        func_to_node_id[func_name] = f"node_{node_id_counter}"
                        node_id_counter += 1
            
            # Add nodes to the merged graph
            for func_name in node_to_func.values():
                node_id = func_to_node_id[func_name]
                
                if node_id not in merged_graph:
                    # Create node with function name as label
                    merged_graph.add_node(node_id, label=func_name)
            
            # Process edges with correct directionality
            for u, v, attrs in graph.edges(data=True):
                if u not in node_to_func or v not in node_to_func:
                    continue  # Skip if either node doesn't have a function name
                
                # Get function names
                u_func = node_to_func[u]
                v_func = node_to_func[v]
                
                # Get node IDs for functions
                u_id = func_to_node_id[u_func]
                v_id = func_to_node_id[v_func]
                
                # Determine edge direction
                if needs_inversion or ('dir' in attrs and attrs['dir'] == '"back"'):
                    # Inverse graph or backward edge - reverse direction
                    from_id, to_id = v_id, u_id
                else:
                    # Regular graph with normal edge direction
                    from_id, to_id = u_id, v_id
                
                # Add edge to merged graph
                merged_graph.add_edge(from_id, to_id, source=source_name)
        
        except Exception as e:
            print(f"Error in second pass for {dot_file}: {str(e)}")
    
    # Add color attributes to nodes based on sources
    colors = ['#ffcc00', '#00cc99', '#ff6666', '#6699ff', '#cc99ff', '#f08080', '#90ee90']
    file_to_color = {}
    
    for i, file in enumerate(dot_files):
        file_name = os.path.basename(file)
        file_to_color[file_name] = colors[i % len(colors)]
    
    # Add color to the graph
    for node in merged_graph.nodes():
        label = merged_graph.nodes[node].get('label')
        if label and label in function_sources:
            sources = function_sources[label]
            
            if len(sources) > 1:
                # Overlapping node - highlight in yellow
                merged_graph.nodes[node]['color'] = 'yellow'
                merged_graph.nodes[node]['style'] = 'filled'
                merged_graph.nodes[node]['penwidth'] = '2.0'
            else:
                # Node from a single source - color by source
                color = file_to_color[sources[0]]
                merged_graph.nodes[node]['color'] = color
                merged_graph.nodes[node]['style'] = 'filled'
    
    # Set graph attributes for better visualization
    merged_graph.graph['rankdir'] = 'LR'
    merged_graph.graph['concentrate'] = 'true'
    merged_graph.graph['splines'] = 'polyline'
    merged_graph.graph['overlap'] = 'false'
    merged_graph.graph['nodesep'] = '0.4'
    merged_graph.graph['ranksep'] = '0.8'
    
    # Write DOT file
    #nx.drawing.nx_pydot.write_dot(merged_graph, output_dot)
    #print(f"Created colored DOT file: {output_dot}")
    
    # Create JSON output
    save_to_json(merged_graph, func_to_node_id, function_sources, file_info, output_json)
    
    # Generate PNG file
    png_path = None
    try:
        # Generate colored PNG
        subprocess.run([
            'dot', 
            '-Tpng', 
            '-Gsize=5000,5000', 
            '-Gdpi=300', 
            output_dot, 
            '-o', 
            output_png
        ], check=True)
        print(f"Created PNG file: {output_png}")
        png_path = output_png
        
        # Calculate statistics
        stats = {
            "Number of functions": len(func_to_node_id),
            "Number of edges": merged_graph.number_of_edges(),
            "Overlapping functions": sum(1 for sources in function_sources.values() if len(sources) > 1),
            "Source files": [os.path.basename(file) for file in dot_files],
            "File colors": {file: color for file, color in file_to_color.items()}
        }
        
        return merged_graph, stats, png_path
    
    except Exception as e:
        print(f"Error generating PNG file: {str(e)}")
        return merged_graph, None, None
    return merged_graph, None, None


def save_to_json(merged_graph: nx.DiGraph, func_to_node_id: Dict[str, str], 
               function_sources: Dict[str, List[str]], file_info: Dict[str, Dict[str, Any]], 
               output_path: str) -> str:
    """
    Save the merged graph to a JSON file with proper function relationships

    Args:
        merged_graph: The NetworkX DiGraph object
        func_to_node_id: Mapping from function names to node IDs
        function_sources: Mapping from function names to source file names
        file_info: Metadata about source files
        output_path: Path where JSON output will be saved

    Returns:
        Path to the created JSON file
    """
    # Create a mapping from node IDs back to function names
    node_id_to_func = {node_id: func_name for func_name, node_id in func_to_node_id.items()}
    
    # Initialize the JSON structure
    json_data = {
        "metadata": {
            "total_functions": len(func_to_node_id),
            "total_edges": merged_graph.number_of_edges(),
            "source_files": [
                {
                    "name": name,
                    "path": info["path"],
                    "is_inverse": info["is_inverse"],
                    "rankdir": info["rankdir"]
                } 
                for name, info in file_info.items()
            ]
        },
        "functions": {}
    }
    
    # Populate function data
    for node_id in merged_graph.nodes():
        func_name = node_id_to_func.get(node_id) or merged_graph.nodes[node_id].get('label')
        if not func_name:
            continue
            
        # Get callers and callees
        callees = []
        for _, to_node in merged_graph.out_edges(node_id):
            callee = node_id_to_func.get(to_node) or merged_graph.nodes[to_node].get('label')
            if callee:
                callees.append(callee)
        
        callers = []
        for from_node, _ in merged_graph.in_edges(node_id):
            caller = node_id_to_func.get(from_node) or merged_graph.nodes[from_node].get('label')
            if caller:
                callers.append(caller)
        
        # Add function entry to JSON
        json_data["functions"][func_name] = {
            "calls": callees,
            "calledBy": callers,
            "sources": function_sources.get(func_name, []),
            "overlapping": len(function_sources.get(func_name, [])) > 1
        }
    
    # Write to file
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Created JSON file: {output_path}")
    return output_path


def display_graph_statistics(stats: Dict[str, Any]) -> None:
    """
    Display statistics about the merged graph
    
    Args:
        stats: Dictionary with graph statistics
    """
    if not stats:
        print("No statistics available")
        return
    
    print("\nMerged Graph Statistics:")
    for key, value in stats.items():
        if key == "Source files":
            print(f"{key}:")
            for i, file in enumerate(value):
                color = stats.get("File colors", {}).get(file, "unspecified")
                print(f"  {i+1}. {file} (Color: {color})")
        elif key == "File colors":
            continue  # Already displayed with source files
        else:
            print(f"{key}: {value}")


def main():
    """Main function to parse arguments and run the script"""
    parser = argparse.ArgumentParser(description='Merge multiple DOT callgraph files into a single graph')
    
    # Create a mutually exclusive group for either file list or directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-f', '--files', nargs='+', help='List of DOT files to merge')
    input_group.add_argument('-d', '--directory', help='Directory containing DOT files to merge')
    
    parser.add_argument('-o', '--output-dir', default='output', 
                        help='Directory where output files will be saved (default: "output")')
    
    args = parser.parse_args()

    # Get the list of DOT files
    if args.files:
        dot_files = args.files
    else:
        dot_files = get_dot_files_from_directory(args.directory)
    
    if not dot_files:
        print("No DOT files to process. Exiting.")
        return
    else:
        print(f"Found {len(dot_files)}: {dot_files}")
    
    # Process the files
    merged_graph, stats, png_path = merge_dot_files_enhanced(dot_files, output_dir=args.output_dir)
    
    # Display statistics
    display_graph_statistics(stats)
    
    # Report completion
    if png_path:
        print(f"\nMerged graph visualization available at: {png_path}")
    print(f"All output files saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
