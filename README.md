# Local LLMs with llama.cpp - Workshop

A comprehensive, beginner-friendly workshop for learning to work with local Large Language Models using llama.cpp and Python.

## 🎯 Workshop Overview

This workshop teaches you how to run and interact with LLMs locally on your machine. By the end, you'll understand how to:

- Load and configure GGUF models
- Generate text responses
- Stream responses in real-time
- Build conversational AI
- Fine-tune generation parameters
- Implement function calling / tool use
- Process batches efficiently

**Tech Stack**: Python 3.8+, llama-cpp-python

---

## 📋 Prerequisites

### Required

- **Python 3.8 or higher** installed
- **4GB+ RAM** (8GB+ recommended)
- **10GB free disk space** (for model storage)
- **Prior experience working with LLMs**
- 

### Optional

- CUDA-compatible GPU (for faster inference)
- C++ compiler (for GPU acceleration)

---

## 🚀 Quick Start

### 1. Clone/Download Workshop

```bash
cd llama-cpp-samples
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**For GPU acceleration** (NVIDIA GPUs only):
```bash
# Install with CUDA support
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### 3. Download a Model

Download a GGUF model and place it in the `models/` directory.

**Recommended for beginners**: Llama-2-7B-Chat Q4_K_M (~4GB)

**Download from Hugging Face**:
```bash

# visit in browser:
# https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q2_k.gguf?download=true
```

### 4. Update Configuration

Edit `config.py` and ensure `MODEL_PATH` points to your downloaded model:

```python
MODEL_PATH = Path("models/qwen2.5-3b-instruct-q2_k.gguf")
```
---

## 📚 Workshop Modules

### Module 1: Model Loading
**File**: [`01_model_loading.py`](01_model_loading.py)
**Run it**:
```bash
python 01_model_loading.py
```

---

### Module 2: Simple Generation
**File**: [`02_simple_generation.py`](02_simple_generation.py)
**Run it**:
```bash
python 02_simple_generation.py
```

**Key Takeaway**: Learn how to generate text from prompts and control basic parameters.

---

### Module 3: Streaming Response
**File**: [`03_streaming.py`](03_streaming.py)
**Run it**:
```bash
python 03_streaming.py
```

**Key Takeaway**: Create responsive, real-time AI interfaces like ChatGPT.

---

### Module 4: Chat Completion
**File**: [`04_chat_completion.py`](04_chat_completion.py)
**Run it**:
```bash
python 04_chat_completion.py
```

**Key Takeaway**: Build conversational AI that maintains context across multiple turns.

---

### Module 5: Parameter Tuning
**File**: [`05_parameter_tuning.py`](05_parameter_tuning.py)
**Run it**:
```bash
python 05_parameter_tuning.py
```

**Key Takeaway**: Master parameter tuning to control output quality and style.

---

### Module 6: Function Calling
**File**: [`06_function_calling.py`](06_function_calling.py)
**Run it**:
```bash
python 06_function_calling.py
```

**Key Takeaway**: Enable LLMs to use tools and interact with external systems.

---

### Module 7: Batch Processing
**File**: [`07_batch_processing.py`](07_batch_processing.py)
**Run it**:
```bash
python 07_batch_processing.py
```

**Key Takeaway**: Efficiently process large datasets with batch operations.

---


## 🔧 Troubleshooting

### "Model not found" Error

**Problem**: `FileNotFoundError: Model not found`

**Solution**:
1. Ensure you downloaded a GGUF model
2. Place it in the `models/` directory
3. Update `MODEL_PATH` in `config.py` to match the filename

---

### Out of Memory Error

**Problem**: System runs out of RAM

**Solutions**:
- Use a smaller model (Q2_K quantization)
- Reduce `n_ctx` (context size) in config
- Close other applications
- Use a smaller model (7B instead of 13B)

---

### Slow Generation

**Problem**: Text generation is very slow

**Solutions**:
- Enable GPU acceleration (if you have NVIDIA GPU)
- Use a smaller quantization (Q4_K_M instead of Q8_0)
- Reduce `max_tokens` to generate less text
- Increase `n_batch` parameter for better throughput

---

### Import Error for llama-cpp-python

**Problem**: `ImportError: No module named 'llama_cpp'`

**Solution**:
```bash
pip install llama-cpp-python
```

For GPU support:
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

---

### Generation Quality Issues

**Problem**: Outputs are repetitive, incoherent, or low quality

**Solutions**:
- Check model quantization (Q2_K is lower quality)
- Adjust temperature (try 0.7 for balanced output)
- Increase `repeat_penalty` to reduce repetition
- Ensure prompts are clear and well-formatted
- Try a larger model if possible

---

## 📖 Additional Resources

### Official Documentation
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp) - Original C++ implementation
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - Python bindings
- [GGUF Format Spec](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - Model format details

### Model Sources
- [Hugging Face Models](https://huggingface.co/models?library=gguf) - All GGUF models

### Learning Resources
- [Prompt Engineering Guide](https://www.promptingguide.ai/) - Advanced prompting techniques
- [LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) - Compare model performance

---

## 🤝 Contributing

Found an issue or want to improve the workshop?

1. Fork the repository
2. Make your changes
3. Submit a pull request

Or open an issue for bugs and suggestions!

---

## 📄 License

This workshop is provided for educational purposes.

Models have their own licenses - check individual model licenses before use.

---

## 🙏 Acknowledgments

- **llama.cpp** by Georgi Gerganov - Amazing C++ implementation
- **llama-cpp-python** by Andrei Betlen - Excellent Python bindings
- **Qwen AI** - For the Llama model family

---

## 📞 Support

**Questions?**
- Check the [Troubleshooting](#-troubleshooting) section
- Open an issue in the repository
- Join the llama.cpp community discussions

---

**Happy Learning! 🚀**

Built with ❤️ for the AI community
