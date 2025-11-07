#!/bin/bash
# Start vLLM server with Qwen2.5-7B-Instruct model
# This script starts an OpenAI-compatible API server using vLLM

# export CUDA_VISIBLE_DEVICES=4,5,6,7

set -e

# Configuration
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
HOST="0.0.0.0"
API_KEY="token-abc123"

# Additional vLLM parameters for optimal throughput on 8x H100 GPUs
# Using Data Parallelism to replicate model across all 8 GPUs
GPU_MEMORY_UTILIZATION=0.6
MAX_MODEL_LEN=32768      # Actual config.json max_position_embeddings for this model
TENSOR_PARALLEL_SIZE=1  # No tensor parallelism (model fits on 1 GPU)
DATA_PARALLEL_SIZE=8     # Replicate model across 8 GPUs for maximum throughput
# Total workers = TP * DP = 1 * 8 = 8 independent workers

echo "Starting vLLM server for maximum throughput..."
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "API Key: $API_KEY"
echo "Configuration: TP=$TENSOR_PARALLEL_SIZE, DP=$DATA_PARALLEL_SIZE ($DATA_PARALLEL_SIZE independent workers)"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION per GPU"
echo "Optimization: Data Parallel - model replicated across $DATA_PARALLEL_SIZE GPUs"
echo ""

# Start vLLM server
# The model will be loaded from the HuggingFace cache
vllm serve "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --api-key "$API_KEY" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --data-parallel-size "$DATA_PARALLEL_SIZE" \
    --dtype auto \
    --trust-remote-code \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# Alternative: If you want to run in background
# nohup vllm serve "$MODEL_NAME" \
#     --host "$HOST" \
#     --port "$PORT" \
#     --api-key "$API_KEY" \
#     --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
#     --max-model-len "$MAX_MODEL_LEN" \
#     --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
#     --data-parallel-size "$DATA_PARALLEL_SIZE" \
#     --dtype auto \
#     --trust-remote-code \
#     --enable-prefix-caching \
#     --enable-chunked-prefill \
#     --enable-auto-tool-choice \
#     --tool-call-parser hermes \
#     > vllm_server.log 2>&1 &
#
# echo "vLLM server started in background. Check vllm_server.log for logs."
# echo "PID: $!"
