#!/bin/bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate bhus

CORPUS="${1:-clres}"
QUERY_N="${QUERY_N:-1}"
TOP_N="${TOP_N:-50}"

models=(
  "bert-base-cased"
  "distilbert-base-cased"
  "roberta-base"
  "distilroberta-base"
  "albert-base-v2"
  "xlnet-base-cased"
  "gpt2"
)

counts=(0 100 250 500 1000 2500)

layer_for_model() {
  case "$1" in
    distilbert-base-cased|distilroberta-base)
      echo 5
      ;;
    *)
      echo 11
      ;;
  esac
}

echo "[metrics] corpus=${CORPUS} query_n=${QUERY_N} top_n=${TOP_N}"

# Optional baseline run, mirroring the original experiment script.
echo "[baseline] bert-base-cased"
python main.py trial --embedding-model "bert-base-cased" --metric "baseline" --query-n "${QUERY_N}" --top-n "${TOP_N}" --bert-layer 7 "${CORPUS}"
python main.py summarize --embedding-model "bert-base-cased" --metric "baseline" --query-n "${QUERY_N}" --top-n "${TOP_N}" --bert-layer 7 "${CORPUS}"
python -m MVA.extended_metrics --embedding-model "bert-base-cased" --metric "baseline" --query-n "${QUERY_N}" --top-n "${TOP_N}" --bert-layer 7 "${CORPUS}"

for model in "${models[@]}"; do
  layer="$(layer_for_model "$model")"
  echo "[model] ${model} layer=${layer}"

  for count in "${counts[@]}"; do
    if [ "$count" -eq 0 ]; then
      echo "[run] ${model} count=0 metric=cosine"
      python main.py trial --embedding-model "$model" --metric "cosine" --query-n "${QUERY_N}" --top-n "${TOP_N}" --bert-layer "$layer" "$CORPUS"
      python main.py summarize --embedding-model "$model" --metric "cosine" --query-n "${QUERY_N}" --top-n "${TOP_N}" --bert-layer "$layer" "$CORPUS"
      python -m MVA.extended_metrics --embedding-model "$model" --metric "cosine" --query-n "${QUERY_N}" --top-n "${TOP_N}" --bert-layer "$layer" "$CORPUS"
      continue
    fi

    weights="models/${model}_${count}.pt"
    if [ ! -f "$weights" ]; then
      echo "[skip] missing checkpoint $weights"
      continue
    fi

    echo "[run] ${model} count=${count} metric=cosine weights=${weights}"
    python main.py trial --embedding-model "$model" --metric "cosine" --query-n "${QUERY_N}" --top-n "${TOP_N}" --bert-layer "$layer" --override-weights "$weights" "$CORPUS"
    python main.py summarize --embedding-model "$model" --metric "cosine" --query-n "${QUERY_N}" --top-n "${TOP_N}" --bert-layer "$layer" --override-weights "$weights" "$CORPUS"
    python -m MVA.extended_metrics --embedding-model "$model" --metric "cosine" --query-n "${QUERY_N}" --top-n "${TOP_N}" --bert-layer "$layer" --override-weights "$weights" "$CORPUS"
  done
done
