"""
Module 1: Model Loading
Learn: Load GGUF models, understand quantization, configure context size
"""

from llama_cpp import Llama
from config import MODEL_PATH, check_model_exists

def main():
    print("=== Module 1: Model Loading ===\n")

    check_model_exists()

    print("Loading model...")
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,           # Context window size
        n_gpu_layers=0,       # 0=CPU, -1=all GPU
        verbose=False
    )
    print("\n✓ Model loaded\n")

    # Show model info
    print(f"Context size: {llm.n_ctx()} tokens (~{int(llm.n_ctx() * 0.75)} words)")
    print(f"Vocabulary: {llm.n_vocab()} tokens\n")

    print("\n✓ Module 1 complete")

if __name__ == "__main__":
    main()
