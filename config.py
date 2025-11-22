"""
Shared configuration for llama.cpp workshop
This file contains common settings used across all workshop modules
"""
from pathlib import Path

# ==================== MODEL CONFIGURATION ====================

# Path to your GGUF model file
# Download a model and place it in the models/ directory
# Recommended: qwen2.5-3b-instruct-q2_k.gguf
# Download from: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q2_k.gguf?download=true
MODEL_PATH = Path("models/qwen2.5-3b-instruct-q2_k.gguf")

# Context window size (number of tokens the model can process)
# Larger = more context but more memory usage
DEFAULT_CONTEXT_SIZE = 2048

# GPU acceleration (set to -1 to use GPU, 0 for CPU only)
# If you have a compatible GPU, setting this to -1 will speed up inference
DEFAULT_GPU_LAYERS = 0  # Change to -1 for GPU acceleration


# ==================== GENERATION PARAMETERS ====================

# Default parameters for text generation
# These can be overridden in individual modules
DEFAULT_PARAMS = {
    "max_tokens": 256,        # Maximum tokens to generate
    "temperature": 0.7,       # Randomness (0.0 = deterministic, 2.0 = very random)
    "top_p": 0.9,            # Nucleus sampling (0.0-1.0)
    "top_k": 40,             # Top-k sampling (limits token choices)
    "repeat_penalty": 1.1,   # Penalty for repeating tokens (1.0 = no penalty)
}


# ==================== HELPER FUNCTIONS ====================

def check_model_exists():
    """
    Validate that the model file exists before attempting to load it

    Raises:
        FileNotFoundError: If the model file is not found
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"\n❌ Model not found at: {MODEL_PATH}\n\n"
            f"📥 Please download a GGUF model and place it in the models/ directory\n"
            f"   Recommended: Llama-2-7B-Chat Q4_K_M\n"
            f"   Download from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF\n"
        )
    print(f"✅ Model found at: {MODEL_PATH}")


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)
