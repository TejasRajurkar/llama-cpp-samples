"""
Module 6: Function Calling
Learn: Tool use, JSON schemas, building agents
"""

import json
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

def main():
    print("=== Module 6: Function Calling ===\n")

    check_model_exists()
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, verbose=False)

    # Example 1: Simple function call
    print("1. Calculator function\n")

    messages = [
        {"role": "system", "content": "You have access to tools. Use them when needed."},
        {"role": "user", "content": "What is 25 × 4?"}
    ]

    print("User: What is 25 × 4?")

    response = llm.create_chat_completion(messages=messages, tools=TOOLS, tool_choice="auto")
    message = response['choices'][0]['message']

    if 'tool_calls' in message:
        tool_call = message['tool_calls'][0]
        func_name = tool_call['function']['name']
        func_args = json.loads(tool_call['function']['arguments'])

        print(f"→ Calling: {func_name}({func_args})")

        result = FUNCTIONS[func_name](**func_args)
        print(f"→ Result: {result}")

        # Send result back to model
        messages.append(message)
        messages.append({"role": "tool", "tool_call_id": tool_call['id'], "content": str(result)})

        final = llm.create_chat_completion(messages=messages)
        print(f"Assistant: {final['choices'][0]['message']['content']}")
        print(f"[Stats] Total tokens: {final['usage']['total_tokens']}")

    # Example 2: Multiple tools
    print("\n\n2. Multi-tool conversation\n")

    messages = [{"role": "system", "content": "You have access to tools."}]

    queries = ["What time is it?", "Calculate 100 ÷ 5"]

    for query in queries:
        print(f"User: {query}")
        messages.append({"role": "user", "content": query})

        response = llm.create_chat_completion(messages=messages, tools=TOOLS, tool_choice="auto")
        msg = response['choices'][0]['message']

        if 'tool_calls' in msg:
            for tc in msg['tool_calls']:
                func_name = tc['function']['name']
                func_args = json.loads(tc['function']['arguments'])

                print(f"→ Calling: {func_name}({func_args})")
                result = FUNCTIONS[func_name](**func_args)
                print(f"→ Result: {result}")

                messages.append(msg)
                messages.append({"role": "tool", "tool_call_id": tc['id'], "content": str(result)})

            final = llm.create_chat_completion(messages=messages)
            reply = final['choices'][0]['message']['content']
            print(f"Assistant: {reply}")
            print(f"[Stats] Tokens: {final['usage']['total_tokens']}\n")
            messages.append({"role": "assistant", "content": reply})

    print("✓ Module 6 complete")

if __name__ == "__main__":
    main()
