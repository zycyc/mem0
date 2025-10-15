#!/bin/bash
# Start vLLM server with Qwen3-Embedding-0.6B model
# This script starts an OpenAI-compatible embeddings API server using vLLM

set -e

# Configuration
MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
PORT=8001
HOST="0.0.0.0"
API_KEY="token-abc123"

# Additional vLLM parameters for optimal throughput on 8x H100 GPUs
# Using Data Parallelism to replicate model across all 8 GPUs
GPU_MEMORY_UTILIZATION=0.2  # Small embedding model (~2GB), minimal memory needed
MAX_MODEL_LEN=8192           # Max sequence length for Qwen3-Embedding
TENSOR_PARALLEL_SIZE=1       # No tensor parallelism (tiny model fits on 1 GPU)
DATA_PARALLEL_SIZE=8         # Replicate model across 8 GPUs for maximum throughput
# Total workers = TP * DP = 1 * 8 = 8 independent workers

echo "Starting vLLM embedding server for maximum throughput..."
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "API Key: $API_KEY"
echo "Configuration: TP=$TENSOR_PARALLEL_SIZE, DP=$DATA_PARALLEL_SIZE (8 independent workers)"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION per GPU (~4GB per GPU)"
echo "Optimization: Data Parallel - model replicated across all 8 GPUs"
echo "Task: Embedding (--task embed)"
echo ""

# Start vLLM embedding server
# The model will be loaded from the HuggingFace cache
vllm serve "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --task embed \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --data-parallel-size "$DATA_PARALLEL_SIZE" \
    --dtype auto \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-chunked-prefill

# Alternative: If you want to run in background
# nohup vllm serve "$MODEL_NAME" \
#     --host "$HOST" \
#     --port "$PORT" \
#     --api-key "$API_KEY" \
#     --task embed \
#     --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
#     --max-model-len "$MAX_MODEL_LEN" \
#     --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
#     --data-parallel-size "$DATA_PARALLEL_SIZE" \
#     --dtype auto \
#     --trust-remote-code \
#     --enable-prefix-caching \
#     --enable-chunked-prefill \
#     > vllm_embedding_server.log 2>&1 &
#
# echo "vLLM embedding server started in background. Check vllm_embedding_server.log for logs."
# echo "PID: $!"
