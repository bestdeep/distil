#!/bin/bash
cd /home/openclaw/distil-paper
source ~/.secrets/distil.env
export HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || echo '')}"
export HF_HOME="/home/openclaw/.cache/validator-hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HUB_CACHE"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
exec python3 scripts/remote_validator.py \
  --lium-api-key "$LIUM_API_KEY" \
  --lium-pod-name "distil-eval" \
  --tempo 600 \
  --use-vllm
