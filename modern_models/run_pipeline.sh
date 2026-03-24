#!/bin/bash
set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-3B}"
CORPUS="${2:-clres}"

echo "=== Modern Model Pipeline ==="
echo "Model: ${MODEL}"
echo "Corpus: ${CORPUS}"
echo ""

# Step 1: Extract embeddings (needs GPU)
echo "[1/2] Extracting embeddings..."
python modern_models/extract_embeddings.py \
    --model "$MODEL" \
    --corpus "$CORPUS" \
    --layer -1 \
    --output-dir cache/modern_models

# Step 2: Run retrieval + metrics
echo "[2/2] Running retrieval and computing metrics..."
python modern_models/run_trial.py \
    --model "$MODEL" \
    --corpus "$CORPUS" \
    --top-n 50 \
    --input-dir cache/modern_models

echo "=== Done ==="
