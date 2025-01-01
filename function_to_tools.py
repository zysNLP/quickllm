import inspect
from typing import Callable, Dict, List
from quickllm.openaillms.function_calling import function_calls


def get_type_info(param_type) -> str:
    """Convert Python type to JSON schema type"""
    type_mapping = {
        str: "string",
        int: "number",
        float: "number",
        bool: "boolean"
    }
    return type_mapping.get(param_type, "string")

def function_to_tool_format(func: Callable) -> Dict:
    """
    Convert a Python function to the OpenAI tool format
    
    Args:
        func: The function to convert
    
    Returns:
        Dict: The function in tool format
    """
    sig = inspect.signature(func)
    doc = inspect.getdoc(func)
    
    # Get parameters info
    properties = {}
    required = []
    
    for name, param in sig.parameters.items():
        param_type = param.annotation if param.annotation != inspect._empty else str
        properties[name] = {
            "type": get_type_info(param_type),
            "description": f"{name} parameter." # In a real system, you'd want better descriptions
        }
        if param.default == inspect._empty:
            required.append(name)
    
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": doc.split('\n')[0] if doc else "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False
            }
        }
    }

def generate_tools() -> List[Dict]:
    """
    Generate tools list from all functions in airline_functions module
    
    Returns:
        List[Dict]: List of tools in OpenAI format
    """
    tools = []
    
    # Get all functions from the module
    functions = inspect.getmembers(function_calls, inspect.isfunction)
    
    for name, func in functions:
        tool = function_to_tool_format(func)
        tools.append(tool)
    
    return tools

def save_tools_to_json(tools: List[Dict], filename: str) -> None:
    """
    Save tools list to a JSON file
    
    Args:
        tools: List of tools in OpenAI format
        filename: Name of the output JSON file
    """
    import json
    with open(filename, 'w') as f:
        json.dump(tools, f, indent=4)

if __name__ == "__main__":
    # Generate tools and save to JSON
    tools = generate_tools()
    print(tools)
    # save_tools_to_json(tools, "generated_tools.json")