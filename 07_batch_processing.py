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
        output = llm(prompt, max_tokens=max_tokens, temperature=0.5)
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
    print("\n1. Batch sentiment classification\n")

    reviews = [
        "This product is amazing!",
        "Terrible quality, very disappointed.",
        "It's okay, nothing special.",
    ]

    prompts = [f"Classify sentiment (positive/negative/neutral): '{r}'\nSentiment:" for r in reviews]

    results, elapsed = process_batch(llm, prompts, max_tokens=1)

    for review, result in zip(reviews, results):
        print(f"'{review}' → {result['response'].strip()}")

    print(f"\n[Stats] Processed {len(reviews)} reviews in {elapsed:.2f}s")
    print(f"[Stats] Throughput: {len(reviews)/elapsed:.2f} reviews/sec")


    print("\n✓ Module 7 complete")
if __name__ == "__main__":
    main()
