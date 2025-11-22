"""
Module 5: Parameter Tuning
Learn: temperature, top_p, top_k, repeat_penalty - control output quality
"""

from llama_cpp import Llama
from config import MODEL_PATH, check_model_exists

def generate(llm, prompt, **kwargs):
    """Generate and print with stats"""
    output = llm(prompt, **kwargs)
    response = output['choices'][0]['text']
    print(f"Response: {response}")
    print(f"[Stats] Tokens: {output['usage']['total_tokens']}")
    return response

def main():
    print("=== Module 5: Parameter Tuning ===\n")

    check_model_exists()
    llm = Llama(model_path=str(MODEL_PATH), n_ctx=2048, verbose=False)

    prompt = "The future of AI is"

    # Temperature
    print("1. Temperature (randomness)\n")

    print("temp=0.0 (deterministic)")
    generate(llm, prompt, max_tokens=30, temperature=0.0)

    print("\ntemp=0.7 (balanced)")
    generate(llm, prompt, max_tokens=30, temperature=0.7)

    print("\ntemp=1.5 (creative)")
    generate(llm, prompt, max_tokens=30, temperature=1.5)

    # Top_p
    print("\n\n2. Top_p (nucleus sampling)\n")

    print("top_p=0.1 (focused)")
    generate(llm, prompt, max_tokens=30, temperature=0.8, top_p=0.1)

    print("\ntop_p=0.9 (balanced)")
    generate(llm, prompt, max_tokens=30, temperature=0.8, top_p=0.9)

    # Top_k
    print("\n\n3. Top_k (token choices)\n")

    print("top_k=10 (limited)")
    generate(llm, prompt, max_tokens=30, temperature=0.8, top_k=10)

    print("\ntop_k=80 (diverse)")
    generate(llm, prompt, max_tokens=30, temperature=0.8, top_k=80)

    # Repeat penalty
    print("\n\n4. Repeat penalty\n")

    repeat_prompt = "List languages: Python, Java,"

    print("repeat_penalty=1.0 (no penalty)")
    generate(llm, repeat_prompt, max_tokens=40, temperature=0.8, repeat_penalty=1.0)

    print("\nrepeat_penalty=1.3 (moderate)")
    generate(llm, repeat_prompt, max_tokens=40, temperature=0.8, repeat_penalty=1.3)

    # Parameter recipes
    print("\n\n5. Parameter Recipes\n")

    print("Code generation (low temp, no penalty)")
    generate(llm, "def factorial(n):", max_tokens=60,
             temperature=0.2, top_p=0.9, repeat_penalty=1.0)

    print("\nCreative writing (high temp, diverse)")
    generate(llm, "Once upon a time,", max_tokens=60,
             temperature=1.1, top_p=0.95, top_k=80, repeat_penalty=1.2)

    print("\n✓ Module 5 complete")

if __name__ == "__main__":
    main()
