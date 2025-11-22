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

    # Basic generation
    print("\n1. Basic generation\n")
    output = llm("What are large language models?", max_tokens=50)
    print(f"Response: {output['choices'][0]['text']}")
    print_stats(output)

    print("\n✓ Module 2 complete")

if __name__ == "__main__":
    main()
