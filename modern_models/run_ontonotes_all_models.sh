#!/bin/bash
set -euo pipefail

# Run all 7 original BERT-family models on OntoNotes using modern_models pipeline.
# Uses --layer -1 (last layer) for all models, matching the original paper protocol
# (layer 11 for base models, layer 5 for distilled — -1 always picks the last).

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
OUTDIR="cache/modern_models"
CORPUS="ontonotes"

models=(
  "bert-base-cased"
  "distilbert-base-cased"
  "roberta-base"
  "distilroberta-base"
  "albert-base-v2"
  "xlnet-base-cased"
  "openai-community/gpt2"
)

for model in "${models[@]}"; do
  slug="${model//\//_}"
  echo ""
  echo "=========================================="
  echo "Model: ${model}"
  echo "=========================================="

  # Step 1: Extract embeddings (skip if already done)
  train_pkl="${OUTDIR}/${CORPUS}_${slug}_train.pkl"
  if [ -f "$train_pkl" ]; then
    echo "[skip] Embeddings already extracted: ${train_pkl}"
  else
    echo "[1/2] Extracting embeddings..."
    python3 modern_models/extract_embeddings.py \
      --model "$model" \
      --corpus "$CORPUS" \
      --layer -1 \
      --output-dir "$OUTDIR"
  fi

  # Step 2: Run retrieval + metrics
  echo "[2/2] Running retrieval + metrics..."
  python3 modern_models/run_trial.py \
    --model "$model" \
    --corpus "$CORPUS" \
    --top-n 50 \
    --input-dir "$OUTDIR"

  echo "[done] ${model}"
done

echo ""
echo "=========================================="
echo "ALL MODELS COMPLETE"
echo "=========================================="
