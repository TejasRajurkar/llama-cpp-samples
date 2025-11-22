"""
Module 3: Streaming Response
Learn: Real-time token streaming, generators, better UX
"""

from llama_cpp import Llama
from config import MODEL_PATH, check_model_exists

def main():
    print("=== Module 3: Streaming Response ===\n")

    check_model_exists()
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, verbose=False)

    # Non-streaming (blocking)
    print("1. Non-streaming (wait for full response)")
    print("Response: ", end="", flush=True)
    output = llm("Explain streaming in one sentence.", max_tokens=50, stream=False)
    print(output['choices'][0]['text'])
    print(f"\n[Stats] Tokens: {output['usage']['total_tokens']}")

    # Streaming (real-time)
    print("\n2. Streaming (tokens appear as generated)")
    print("Response: ", end="", flush=True)

    stream = llm("Explain streaming in one sentence.", max_tokens=50, stream=True)

    token_count = 0
    for chunk in stream:
        token = chunk['choices'][0]['text']
        print(token, end="", flush=True)
        token_count += 1

    print(f"\n\n[Stats] Tokens streamed: {token_count}")

    # Streaming with processing
    print("\n3. Streaming with real-time stats")
    print("Response: ", end="", flush=True)

    stream = llm("List 3 programming languages.", max_tokens=50, stream=True)

    full_text = ""
    char_count = 0

    for chunk in stream:
        token = chunk['choices'][0]['text']
        print(token, end="", flush=True)
        full_text += token
        char_count += len(token)

    print(f"\n\n[Stats] Characters: {char_count} | Words: ~{len(full_text.split())}")

    print("\n✓ Module 3 complete")

if __name__ == "__main__":
    main()
