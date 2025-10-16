"""
function_relationships.py - Helper module for extracting function relationships from call graphs.

This module provides utilities to load a call graph from a JSON file and get relationship
information (callers/callees) for functions to enhance documentation prompts.

Example usage:
    
    # Load the call graph once
    graph = load_function_graph("merged_graph.json")
    
    # For each function to document
    function_name = "runTFCSShapeValidation"
    relationship_text = get_relationship_text(function_name, graph)
    
    # Add to prompt
    enhanced_prompt = f"{base_prompt}\n\n{relationship_text}\n\n{function_code}"
"""

import json
import re

def load_function_graph(json_file_path):
    """
    Load the function call graph from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing the function call graph
        
    Returns:
        Dictionary with function information from the graph
    """
    try:
        with open(json_file_path, 'r') as file:
            graph_data = json.load(file)
            
        # Return the functions dictionary directly
        return graph_data.get('functions', {})
    except Exception as e:
        print(f"Error loading function graph: {e}")
        return {}

def clean_function_name(name):
    """
    Clean up function name by removing DOT formatting markers and other non-descriptive elements.
    
    Args:
        name: Function name to clean
        
    Returns:
        Cleaned function name
    """
    # Remove DOT formatting markers
    cleaned = name.replace('\\l', '')
    
    # Remove any trailing/leading whitespace
    cleaned = cleaned.strip()
    
    # Remove any line breaks
    cleaned = cleaned.replace('\n', ' ')
    cleaned = cleaned.replace('"', '')
    
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned

def match_function_name(function_name, function_graph):
    """
    Match a function name to an entry in the function graph.
    
    Args:
        function_name: Name of the function to match
        function_graph: Function graph dictionary
        
    Returns:
        Matched name in the graph or None if no match found
    """
    graph_names = []

    # Clean the function name
    clean_name = clean_function_name(function_name)
    
    # Try direct match first
    if clean_name in function_graph:
        graph_names.append(clean_name)
    
    # Try cleaning all graph names and matching
    for graph_name in function_graph:
        if clean_name == clean_function_name(graph_name):
            graph_names.append(graph_name)
    
    # Try matching by parts of the name
    for graph_name in function_graph:
        graph_clean = clean_function_name(graph_name)
        # If function name is part of graph name or vice versa
        if clean_name in graph_clean: # or graph_clean in clean_name:
            graph_names.append(graph_name)
    
    if graph_names:
        return graph_names
    else:
        return None

def get_function_relationships(function_name, function_graph):
    """
    Get relationship information for a function.
    
    Args:
        function_name: Name of the function to look up
        function_graph: Function graph dictionary
        
    Returns:
        Dictionary with calls and called_by information, or None if not found
    """
    # Match the function name to the graph
    matched_name = match_function_name(function_name, function_graph)
    
    if not matched_name:
        return None
    
    # Get the function info
    all_matches = []

    for m_n in matched_name:
        func_info = function_graph[m_n]

        callers, callees = [], []
        for func in func_info.get("calls", []):
            callers.append(func)

        for func in func_info.get("calledBy", []):
            callees.append(func)

        all_matches.append({
            "name": clean_function_name(m_n),
            "calls": [clean_function_name(func) for func in callers],
            "called_by": [clean_function_name(func) for func in callees]
        })
        
    return all_matches

def get_relationship_text(function_name, function_graph):
    """
    Generate formatted text about function relationships for inclusion in prompts.
    
    Args:
        function_name: Name of the function to look up
        function_graph: Function graph dictionary
        
    Returns:
        Formatted text describing function relationships, or empty string if not found
    """
    # Get the relationships
    all_relationships = get_function_relationships(function_name, function_graph)
    
    if not all_relationships:
        return ""

    # Format the text
    text = "Function Relationship Information:\n\n"
    
    # Format what this function calls
    for relationships in all_relationships:
        text += f'\n\n{relationships["name"]}\n' 
        text += "This function calls:\n"
        if relationships["calls"]:
            for called_func in relationships["calls"]:
                text += f"- {called_func}\n"
        else:
            text += "- This function does not call any other functions in the tracked codebase.\n"
        
        # Format what calls this function
        text += "\nThis function is called by:\n"
        if relationships["called_by"]:
            for caller in relationships["called_by"]:
                text += f"- {caller}\n"
        else:
            text += "- This function is not called by any other functions in the tracked codebase.\n"
    
    return text

def enhance_prompt_with_relationships(base_prompt, function_name, function_code, function_graph):
    """
    Enhance a documentation prompt with function relationship information.
    
    Args:
        base_prompt: The base documentation prompt
        function_name: Name of the function to document
        function_code: Code of the function to document
        function_graph: Function graph dictionary
        
    Returns:
        Enhanced prompt with relationship information
    """
    # Get relationship text
    relationship_text = get_relationship_text(function_name, function_graph)
    
    # If no relationships found, just return the original prompt
    if not relationship_text:
        return f"{base_prompt}\n\n{function_code}"
    
    # Enhance the prompt with relationship information
    enhanced_prompt = f"{base_prompt}\n\n{relationship_text}\n\n{function_code}"
    
    return enhanced_prompt
