"""
Module 4: Chat Completion
Learn: Multi-turn conversations, message roles, history management
"""

from llama_cpp import Llama
from config import MODEL_PATH, check_model_exists

def print_chat_stats(response):
    """Print chat completion statistics"""
    usage = response['usage']
    print(f"[Stats] Prompt: {usage['prompt_tokens']} | "
          f"Completion: {usage['completion_tokens']} | "
          f"Total: {usage['total_tokens']}")

def main():
    print("=== Module 4: Chat Completion ===\n")

    check_model_exists()
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, verbose=False)

    # Example 1: System message effect
    print("1. Without system message")
    messages = [
        {"role": "user", "content": "What is a variable?"}
    ]
    response = llm.create_chat_completion(messages=messages, max_tokens=50)
    print(f"Response: {response['choices'][0]['message']['content']}")
    print_chat_stats(response)

    print("\n2. With system message (Python tutor)")
    messages = [
        {"role": "system", "content": "You are a concise Python tutor."},
        {"role": "user", "content": "What is a variable?"}
    ]
    response = llm.create_chat_completion(messages=messages, max_tokens=500)
    print(f"Response: {response['choices'][0]['message']['content']}")
    print_chat_stats(response)

    # Example 2: Multi-turn conversation
    print("\n3. Multi-turn conversation")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    # Turn 1
    messages.append({"role": "user", "content": "What are list comprehensions?"})
    print("\nUser: What are list comprehensions?")

    response = llm.create_chat_completion(messages=messages, max_tokens=60)
    assistant_reply = response['choices'][0]['message']['content']
    print(f"Assistant: {assistant_reply}")
    print_chat_stats(response)

    messages.append({"role": "assistant", "content": assistant_reply})

    # Turn 2
    messages.append({"role": "user", "content": "Show me an example."})
    print("\nUser: Show me an example.")

    response = llm.create_chat_completion(messages=messages, max_tokens=600)
    assistant_reply = response['choices'][0]['message']['content']
    print(f"Assistant: {assistant_reply}")
    print_chat_stats(response)

    # Example 3: Streaming chat
    print("\n4. Streaming chat completion")
    messages = [
        {"role": "user", "content": "Explain functions in one sentence."}
    ]

    print("Response: ", end="", flush=True)
    stream = llm.create_chat_completion(messages=messages, max_tokens=40, stream=True)

    full_response = ""
    for chunk in stream:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            token = delta['content']
            print(token, end="", flush=True)
            full_response += token

    print(f"\n[Stats] Response length: {len(full_response)} chars")

    print("\n✓ Module 4 complete")

if __name__ == "__main__":
    main()
