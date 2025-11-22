"""
Module 6: Tool Calling
Learn: Tool use, JSON schemas, building agents
"""

import json
import re
from datetime import datetime
from llama_cpp import Llama
from config import MODEL_PATH, check_model_exists

# Tool definitions
def calculator(operation: str, a: float, b: float) -> float:
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b if b != 0 else "Error"}
    return ops.get(operation, "Invalid")

def get_time(timezone: str = "UTC") -> str:
    return f"Current time ({timezone}): {datetime.now().strftime('%H:%M:%S')}"

# Tool schemas
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform arithmetic: add, subtract, multiply, divide",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["operation", "a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time",
            "parameters": {"type": "object", "properties": {"timezone": {"type": "string"}}}
        }
    }
]

FUNCTIONS = {"calculator": calculator, "get_time": get_time}

# Utility functions
def clean_malformed_json(content: str) -> str:
    """Clean up malformed JSON from model output (double braces, unquoted keys)"""
    content = content.strip()
    # Remove double braces
    content = content.replace('{{', '{').replace('}}', '}')
    # Fix unquoted property names and values
    content = re.sub(r'{\s*"?name"?\s*:', '{"name":', content)
    content = re.sub(r',\s*"?arguments"?\s*:', ',"arguments":', content)
    # Fix unquoted function names (calculator, get_time, etc.)
    content = re.sub(r':\s*([a-z_]+)\s*,', r': "\1",', content)
    return content

def extract_function_call(message: dict) -> tuple:
    """
    Extract tool name and arguments from model message.
    Returns: (tool_name, tool_args) or (None, None) if no valid call found
    """
    # Check for proper tool_calls format (native tool calling)
    if 'tool_calls' in message:
        tool_call = message['tool_calls'][0]
        func_name = tool_call['function']['name']
        func_args = json.loads(tool_call['function']['arguments'])
        return func_name, func_args

    # Check for text-based tool calls (workaround for models without proper tool calling)
    if 'content' in message and message['content'] and '{' in message['content']:
        try:
            content = clean_malformed_json(message['content'])
            content_json = json.loads(content)

            if 'name' in content_json and 'arguments' in content_json:
                return content_json['name'], content_json['arguments']
        except (json.JSONDecodeError, KeyError):
            pass

    return None, None

def execute_function_call(llm, messages: list, func_name: str, func_args: dict,
                         use_native_format: bool = False, tool_call_id: str = None) -> dict:
    """
    Execute a tool and get final response from model.

    Args:
        llm: The language model instance
        messages: Conversation message history
        func_name: Name of tool to call
        func_args: Tool arguments as dict
        use_native_format: Whether to use native tool_calls format
        tool_call_id: Tool call ID for native format

    Returns:
        Dict with 'result', 'response', and 'tokens' keys
    """
    print(f"→ Calling: {func_name}({func_args})")
    result = FUNCTIONS[func_name](**func_args)
    print(f"→ Result: {result}")

    if use_native_format and tool_call_id:
        # Use native tool calling format
        messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": str(result)})
    else:
        # Use workaround format (text-based)
        messages.append({"role": "user", "content": f"The function returned: {result}"})

    final = llm.create_chat_completion(messages=messages)
    response_text = final['choices'][0]['message']['content']

    return {
        'result': result,
        'response': response_text,
        'tokens': final['usage']['total_tokens']
    }

def main():
    print("=== Module 6: Tool Calling ===\n")

    check_model_exists()
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, verbose=False)

    # Example 1: Simple tool call
    print("1. Calculator tool\n")

    messages = [
        {"role": "system", "content": "You have access to tools. Use them when needed."},
        {"role": "user", "content": "What is 25 × 4?"}
    ]

    print("User: What is 25 × 4?")

    response = llm.create_chat_completion(messages=messages, tools=TOOLS, tool_choice="auto")
    message = response['choices'][0]['message']

    # Extract and execute tool call
    func_name, func_args = extract_function_call(message)

    if func_name:
        messages.append({"role": "assistant", "content": message.get('content', '')})

        # Determine if using native tool calling format
        use_native = 'tool_calls' in message
        tool_call_id = message['tool_calls'][0]['id'] if use_native else None

        result_data = execute_function_call(llm, messages, func_name, func_args,
                                           use_native, tool_call_id)

        print(f"Assistant: {result_data['response']}")
        print(f"[Stats] Total tokens: {result_data['tokens']}")
    else:
        print("Assistant: I don't have the tools to answer that question.")

    # Example 2: Multiple tools
    print("\n\n2. Multi-tool conversation\n")

    messages = [{"role": "system", "content": "You have access to tools."}]

    queries = ["What time is it?", "Calculate 100 ÷ 5"]

    for query in queries:
        print(f"User: {query}")
        messages.append({"role": "user", "content": query})

        response = llm.create_chat_completion(messages=messages, tools=TOOLS, tool_choice="auto")
        msg = response['choices'][0]['message']

        # Extract and execute tool call
        func_name, func_args = extract_function_call(msg)

        if func_name:
            messages.append({"role": "assistant", "content": msg.get('content', '')})

            # Determine if using native tool calling format
            use_native = 'tool_calls' in msg
            tool_call_id = msg['tool_calls'][0]['id'] if use_native else None

            result_data = execute_function_call(llm, messages, func_name, func_args,
                                               use_native, tool_call_id)

            print(f"Assistant: {result_data['response']}")
            print(f"[Stats] Tokens: {result_data['tokens']}\n")
            messages.append({"role": "assistant", "content": result_data['response']})
        else:
            if 'content' in msg:
                print(f"Assistant: {msg['content']}\n")
                messages.append(msg)

    print("✓ Module 6 complete")

if __name__ == "__main__":
    main()
