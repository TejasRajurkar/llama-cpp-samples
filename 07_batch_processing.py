"""
Module 7: Batch Processing
Learn: Process multiple prompts efficiently, measure throughput
"""

import time
from llama_cpp import Llama
from config import MODEL_PATH, check_model_exists

def process_batch(llm, prompts, max_tokens=50):
    """Process multiple prompts and return results with timing"""
    results = []
    start = time.time()

    for prompt in prompts:
        output = llm(prompt, max_tokens=max_tokens, temperature=0.7)
        results.append({
            'prompt': prompt,
            'response': output['choices'][0]['text'],
            'tokens': output['usage']['total_tokens']
        })

    elapsed = time.time() - start
    return results, elapsed

def main():
    print("=== Module 7: Batch Processing ===\n")

    check_model_exists()

    # Load with batch optimization
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,
        n_batch=512,  # Batch size for processing
        verbose=False
    )

    # Example 1: Batch classification
    print("1. Batch sentiment classification\n")

    reviews = [
        "This product is amazing!",
        "Terrible quality, very disappointed.",
        "It's okay, nothing special.",
    ]

    prompts = [f"Classify sentiment (positive/negative/neutral): '{r}'\nSentiment:" for r in reviews]

    results, elapsed = process_batch(llm, prompts, max_tokens=5)

    for review, result in zip(reviews, results):
        print(f"'{review}' → {result['response'].strip()}")

    print(f"\n[Stats] Processed {len(reviews)} reviews in {elapsed:.2f}s")
    print(f"[Stats] Throughput: {len(reviews)/elapsed:.2f} reviews/sec")

    # Example 2: Batch summarization
    print("\n\n2. Batch summarization\n")

    texts = [
        "Python is a programming language known for simplicity and readability.",
        "Machine learning enables systems to learn from data without explicit programming.",
    ]

    prompts = [f"Summarize in 5 words: {t}\nSummary:" for t in texts]

    results, elapsed = process_batch(llm, prompts, max_tokens=10)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['response'].strip()}")

    print(f"\n[Stats] Time: {elapsed:.2f}s | Avg: {elapsed/len(texts):.2f}s per summary")

    # Example 3: Throughput measurement
    print("\n\n3. Throughput measurement\n")

    test_prompts = [f"What is {i} + {i+1}?" for i in range(5)]

    results, elapsed = process_batch(llm, test_prompts, max_tokens=10)

    total_tokens = sum(r['tokens'] for r in results)
    throughput = len(test_prompts) / elapsed

    print(f"Processed {len(test_prompts)} prompts")
    print(f"[Stats] Time: {elapsed:.2f}s")
    print(f"[Stats] Throughput: {throughput:.2f} prompts/sec")
    print(f"[Stats] Total tokens: {total_tokens}")
    print(f"[Stats] Tokens/sec: {total_tokens/elapsed:.1f}")

    # Example 4: Shared prefix optimization
    print("\n\n4. Shared prefix optimization\n")

    prefix = "Classify this as code/text/data:\n\n"
    items = ["def foo():", "Hello world", "1,2,3,4"]

    prompts = [f"{prefix}{item}\n\nType:" for item in items]

    results, elapsed = process_batch(llm, prompts, max_tokens=5)

    for item, result in zip(items, results):
        print(f"'{item}' → {result['response'].strip()}")

    print(f"\n[Stats] Time: {elapsed:.2f}s (prefix cached across prompts)")

    print("\n✓ Module 7 complete")
    print("\n🎉 Workshop complete! You've learned:")
    print("  ✓ Model loading and configuration")
    print("  ✓ Text generation and parameters")
    print("  ✓ Streaming responses")
    print("  ✓ Chat completion and conversations")
    print("  ✓ Parameter tuning for quality")
    print("  ✓ Function calling and tool use")
    print("  ✓ Batch processing for efficiency")

if __name__ == "__main__":
    main()
