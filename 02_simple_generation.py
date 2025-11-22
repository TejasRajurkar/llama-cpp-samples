"""
Module 2: Simple Generation
Learn: Prompt → completion, temperature, max_tokens
"""

from llama_cpp import Llama
from config import MODEL_PATH, check_model_exists

def print_stats(output):
    """Print model usage statistics"""
    usage = output['usage']
    print(f"\n[Stats] Prompt: {usage['prompt_tokens']} tokens | "
          f"Completion: {usage['completion_tokens']} tokens | "
          f"Total: {usage['total_tokens']} tokens")

def main():
    print("=== Module 2: Simple Generation ===\n")

    check_model_exists()
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, verbose=False)

    # Example 1: Basic generation
    print("1. Basic generation")
    output = llm("What is Python?", max_tokens=50, temperature=0.7)
    print(f"Response: {output['choices'][0]['text']}")
    print_stats(output)

    # Example 2: Temperature effect
    print("\n2. Temperature: 0.0 (deterministic)")
    output = llm("The best thing about coding is", max_tokens=30, temperature=0.0)
    print(f"Response: {output['choices'][0]['text']}")
    print_stats(output)

    print("\n3. Temperature: 1.5 (creative)")
    output = llm("The best thing about coding is", max_tokens=30, temperature=1.5)
    print(f"Response: {output['choices'][0]['text']}")
    print_stats(output)

    # Example 3: Max tokens
    print("\n4. Max tokens: 20 (short)")
    output = llm("Explain recursion.", max_tokens=20, temperature=0.7)
    print(f"Response: {output['choices'][0]['text']}")
    print(f"Finish reason: {output['choices'][0]['finish_reason']}")
    print_stats(output)

    print("\n✓ Module 2 complete")

if __name__ == "__main__":
    main()
