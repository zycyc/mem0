# Local Model Setup for Evaluation

This guide explains how to run the evaluation using local models instead of OpenAI API.

## Overview

The evaluation now supports running with:
- **Local LLM**: Qwen2.5-7B-Instruct via vLLM (OpenAI-compatible API)
- **Local Embeddings**: Qwen3-Embedding-0.6B via transformers with CUDA acceleration

## Prerequisites

1. **GPU**: CUDA-capable GPU (e.g., H100s)
2. **Models**: Pre-downloaded from HuggingFace:
   - `Qwen/Qwen2.5-7B-Instruct`
   - `Qwen/Qwen3-Embedding-0.6B`

## Installation

Install required packages using `uv`:

```bash
# Install PyTorch with CUDA support
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install transformers and related packages
uv pip install transformers accelerate

# Install vLLM
uv pip install vllm

# Add to pyproject.toml (if needed)
# uv add torch transformers accelerate vllm
```

## Configuration

### 1. Environment Variables

Create or update your `.env` file in the `evaluation/` directory:

```bash
# vLLM Server Configuration
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=token-abc123

# Model Configuration
MODEL=Qwen/Qwen2.5-7B-Instruct

# Optional: Override for other scripts that might use OpenAI directly
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=token-abc123
```

### 2. Start vLLM Server

Before running experiments, start the vLLM server:

```bash
cd evaluation
./start_vllm_server.sh
```

Or manually:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key token-abc123 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192 \
    --dtype auto \
    --trust-remote-code
```

Wait for the server to fully start (you'll see "Application startup complete" in the logs).

### 3. Verify Server is Running

Test the vLLM server:

```bash
curl http://localhost:8000/v1/models \
    -H "Authorization: Bearer token-abc123"
```

## Running Experiments

Once the vLLM server is running, you can run experiments as usual:

```bash
# Run LangMem experiments with local models
make run-langmem

# Or manually
python run_experiments.py --technique_type langmem --output_folder results/
```

## Architecture Changes

### Modified Files

1. **`src/local_embeddings.py`** (NEW)
   - Wrapper for Qwen3-Embedding-0.6B model
   - Handles tokenization, embedding generation, and normalization
   - Uses CUDA for acceleration
   - Singleton pattern for model reuse

2. **`src/langmem.py`** (MODIFIED)
   - OpenAI client now points to vLLM server (`base_url` parameter)
   - Custom embedding function for LangGraph's InMemoryStore
   - Automatic embedding dimension detection (896 for Qwen3-Embedding-0.6B)
   - Environment variable management for LangGraph agent

### How It Works

```
┌─────────────────┐
│  langmem.py     │
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌──────────────────┐
│ OpenAI Client   │  │ LocalEmbeddings  │
│ (→ vLLM)        │  │ (Qwen3-Emb)      │
└────────┬────────┘  └────────┬─────────┘
         │                    │
         ▼                    ▼
┌─────────────────┐  ┌──────────────────┐
│ vLLM Server     │  │ Transformers +   │
│ (Qwen2.5-7B)    │  │ CUDA             │
│ :8000           │  │                  │
└─────────────────┘  └──────────────────┘
```

## Performance Notes

- **Optimization Strategy**: Maximum throughput via concurrent request batching
- **GPU Usage**: Embedding model (~2GB) shares GPU 0 with vLLM, all 8 GPUs available for LLM
- **Embedding Model**: Runs on GPU 0 (1024-dim embeddings, ~2GB memory)
- **LLM Model**: Served via vLLM with TP=1 (single GPU per request, ~24GB per instance)
- **Total per GPU**: ~26GB used out of 80GB available (plenty of headroom)
- **Automatic Routing**: vLLM automatically batches and routes concurrent requests across all 8 GPUs
- **KV Caching**: Enabled via `--enable-prefix-caching` for faster repeated queries
- **Chunked Prefill**: Enabled for better throughput with long contexts
- **Concurrency**: Can handle ~8 concurrent requests efficiently (one per GPU)
- **Batch Processing**: vLLM's continuous batching automatically distributes work

## Troubleshooting

### vLLM Server Won't Start

- Check GPU availability: `nvidia-smi`
- Ensure models are downloaded: `ls ~/.cache/huggingface/hub/`
- Check port availability: `lsof -i :8000`

### Out of Memory Errors

- Reduce `--gpu-memory-utilization` (e.g., 0.8)
- Reduce `--max-model-len` (e.g., 4096)
- Use smaller batch sizes in experiments

### Connection Errors

- Verify vLLM server is running: `curl http://localhost:8000/health`
- Check `.env` has correct `VLLM_BASE_URL`
- Ensure firewall allows localhost connections

## Extending to Other Scripts

To use local models in other evaluation scripts (e.g., `rag.py`, `memzero/search.py`):

1. Import the local embeddings:
   ```python
   from local_embeddings import get_embedding_model
   embedding_model = get_embedding_model(device="cuda")
   ```

2. Replace OpenAI client:
   ```python
   from openai import OpenAI
   client = OpenAI(
       base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
       api_key=os.getenv("VLLM_API_KEY", "token-abc123")
   )
   ```

3. Replace embedding calls:
   ```python
   # Old: response = client.embeddings.create(model="...", input=texts)
   # New: embeddings = embedding_model.encode(texts)
   ```

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3-Embedding Model Card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [Qwen2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
