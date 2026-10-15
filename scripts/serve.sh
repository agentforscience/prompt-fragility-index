#!/bin/bash
# serve.sh <model.gguf> <port> <threads> <parallel>  -- start a llama.cpp OpenAI-compatible server
cd "$(dirname "$0")/.."
exec tools/llamacpp/llama-b11624/llama-server -m "$1" --port $2 -t $3 -tb $3 -np $4 -c $((1536*$4)) --poll 0 --no-webui
