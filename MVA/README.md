# Probing Word Sense Structure in Contextualized Embeddings

Reproduction and extension of [BERT Has Uncommon Sense (Gessler & Schneider, 2021)](https://aclanthology.org/2021.blackboxnlp-1.43/) for the MVA SNLP course.

## What This Project Does

Evaluates how well contextualized word embedding models cluster same-sense word occurrences using a query-by-example nearest-neighbor retrieval framework. Given a word in context, the system finds the 50 most similar uses of the same lemma in a training corpus and measures whether same-sense instances rank higher.

**Three contributions beyond the original paper:**

1. **Extended geometric analysis** — recall@k, hit@k, MRR, intra/inter-sense distances, cluster margins, centroid margin
2. **Modern LLM evaluation** — Qwen2.5-3B (3B param decoder-only model) on both corpora
3. **Sentiment/connotation analysis** — testing whether BERT embeddings separate positive vs negative sentence polarity for opinion-bearing words (SST2 + SentiWordNet)

## Results Overview

### OntoNotes — MAP@50 Precision (no fine-tuning, last layer)

| Model | Rare senses (l<500, r<0.25) | Common (l<500, r>=0.25) | Common (l>=500, r>=0.25) |
|---|---|---|---|
| bert-base-cased | **47.6%** | 86.8% | 95.0% |
| roberta-base | 47.5% | **87.1%** | **95.8%** |
| albert-base-v2 | 46.5% | 86.1% | 95.8% |
| distilroberta-base | 46.0% | 86.4% | 95.6% |
| distilbert-base-cased | 45.4% | 86.0% | 95.1% |
| xlnet-base-cased | 33.3% | 80.5% | 89.5% |
| **Qwen2.5-3B** | 23.1% | 82.4% | 87.0% |
| gpt2 | 21.5% | 76.8% | 86.5% |

### CL-Res/PDEP — MAP@50 Precision (no fine-tuning, last layer)

| Model | Rare senses (l<500, r<0.25) | Common (l<500, r>=0.25) | Common (l>=500, r>=0.25) |
|---|---|---|---|
| bert-base-cased | **59.6%** | **83.5%** | **89.4%** |
| distilbert-base-cased | 58.1% | 83.2% | 88.1% |
| albert-base-v2 | 56.5% | 82.2% | 88.4% |
| **Qwen2.5-3B** | 42.0% | 75.8% | 79.4% |
| roberta-base | 39.8% | 76.8% | 80.0% |
| xlnet-base-cased | 35.8% | 74.1% | 75.2% |
| distilroberta-base | 32.4% | 72.2% | 70.8% |
| gpt2 | 21.8% | 63.4% | 61.0% |

Full results with all fine-tuning levels (0, 100, 250, 500, 1000, 2500 STREUSLE instances) are in the root-level `*_results.tsv` and `*_extended_results.tsv` files.

### Sentiment Analysis

Using SST-2 sentences and SentiWordNet to select opinion-bearing words, BERT embeddings show **near-zero separation margin** between positive and negative polarity clusters for most words. Only 1 of 42 tested words ("sweet") showed positive margin. Conclusion: BERT's contextual embeddings primarily encode syntactic/semantic context, not sentence-level sentiment polarity.

## Repository Structure

```
main.py                          # Original CLI: finetune / trial / summarize (AllenNLP pipeline)
clres_main.py                    # Legacy CL-Res-only entry point
bssp/                            # Core library
  common/
    analysis.py                  # Metrics: precision, recall, F1, hit, MRR, geometry
    config.py                    # Config object threading through all functions
    paths.py                     # Cache/output path generation
    reading.py                   # Dataset caching, BERT layer selection, embedder construction
    nearest_neighbor_models.py   # NearestNeighborRetriever, RandomRetriever
  clres/dataset_reader.py        # PDEP/CL-Res CoNLL-U reader
  ontonotes/dataset_reader.py    # OntoNotes reader (AllenNLP-based)
  semcor/dataset_reader.py       # SemCor reader
  fews/dataset_reader.py         # FEWS reader
  fine_tuning/                   # STREUSLE supersense fine-tuning
modern_models/                   # Standalone pipeline for large HF models (no AllenNLP)
  extract_embeddings.py          # Extract word-level embeddings from any HF model
  run_trial.py                   # Nearest-neighbor retrieval + metrics
  run_pipeline.sh                # End-to-end: extract → retrieve → metrics
  run_ontonotes_all_models.sh    # Batch all 7 models on OntoNotes
MVA/
  extended_metrics.py            # Extended metrics from existing trial output
  run_all_metrics.sh             # Sweep all models on a corpus
  build_bucket_table.py          # Generate LaTeX/CSV/MD tables from results
  sentiment_analysis.ipynb       # Sentiment/connotation analysis notebook
  tables/                        # Pre-generated result tables
scripts/
  all_experiments.sh             # Original paper experiment sweep
notebooks/                       # Analysis notebooks
```

## Setup

### Environment

```bash
git submodule init && git submodule update
conda create --name bhus python=3.8
conda activate bhus
pip install -r requirements.txt
mkdir -p models cache
```

For `modern_models/` pipeline (Qwen, etc.), use Python 3.10+ with modern PyTorch:
```bash
pip install torch transformers accelerate click conllu pandas tqdm
```

### Data

#### PDEP/CL-Res (included in repo)
Already at `data/pdep/pdep_{train,test}.conllu`.

#### STREUSLE (git submodule)
```bash
git submodule update --init
```
Used for fine-tuning. Data at `data/streusle/train/streusle.ud_train.json`.

#### OntoNotes 5.0
Not included (LDC license). Two ways to obtain:

1. **Official**: [LDC2013T19](https://catalog.ldc.upenn.edu/LDC2013T19), then clone [skeleton repo](https://github.com/ontonotes/conll-formatted-ontonotes-5.0) and run `skeleton2conll.sh`
2. **Mendeley mirror**: [https://data.mendeley.com/datasets/zmycy7t9h9/2](https://data.mendeley.com/datasets/zmycy7t9h9/2) — download `conll-2012.zip`, extract, and symlink:

```bash
unzip conll-2012.zip
ln -s path/to/conll-2012/v12 data/conll-formatted-ontonotes-5.0
```

Expected structure:
```
data/conll-formatted-ontonotes-5.0/data/{train,development,test}/data/english/annotations/**/*.gold_conll
```

#### SST-2 (sentiment analysis)
Auto-downloaded via HuggingFace Datasets: `load_dataset("sst2")`.

#### SentiWordNet 3.0
Download from [https://github.com/aesuli/SentiWordNet](https://github.com/aesuli/SentiWordNet) and place `SentiWordNet_3.0.0.txt` in `MVA/`.

### Pre-extracted Embeddings

To skip the GPU-intensive embedding extraction step, download pre-extracted embeddings from HuggingFace:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Bichrai/bhus-embeddings', repo_type='dataset', local_dir='cache/modern_models')
"
```

This provides pickle files for all 8 models (7 original + Qwen2.5-3B) on both CL-Res and OntoNotes (~11GB).

### Fine-tuned Model Weights

STREUSLE-fine-tuned weights are available at [Lemhemez/SNLP](https://huggingface.co/Lemhemez/SNLP) on HuggingFace.

## How to Reproduce

### 1. CL-Res reproduction (original AllenNLP pipeline)

```bash
# Baseline
python main.py trial --embedding-model bert-base-cased --metric baseline --query-n 1 --bert-layer 7 clres
python main.py summarize --embedding-model bert-base-cased --metric baseline --query-n 1 --bert-layer 7 clres

# Model evaluation (e.g. bert-base-cased, last layer)
python main.py trial --embedding-model bert-base-cased --metric cosine --query-n 1 --bert-layer 11 clres
python main.py summarize --embedding-model bert-base-cased --metric cosine --query-n 1 --bert-layer 11 clres

# Extended metrics
python -m MVA.extended_metrics --embedding-model bert-base-cased --metric cosine --bert-layer 11 clres

# Full sweep (all models, all fine-tuning levels)
bash MVA/run_all_metrics.sh clres
```

### 2. OntoNotes reproduction (modern_models pipeline)

The `modern_models/` pipeline bypasses AllenNLP and works with any HuggingFace model. Requires GPU for embedding extraction.

```bash
export PYTHONPATH=$(pwd):$PYTHONPATH

# Single model
python modern_models/extract_embeddings.py --model bert-base-cased --corpus ontonotes --layer -1
python modern_models/run_trial.py --model bert-base-cased --corpus ontonotes --top-n 50

# All 7 original models
bash modern_models/run_ontonotes_all_models.sh

# Qwen2.5-3B
bash modern_models/run_pipeline.sh Qwen/Qwen2.5-3B ontonotes
bash modern_models/run_pipeline.sh Qwen/Qwen2.5-3B clres
```

If you downloaded the pre-extracted embeddings, skip the extraction step and run `run_trial.py` directly.

### 3. Sentiment analysis

Open `MVA/sentiment_analysis.ipynb` in Jupyter. Requires `SentiWordNet_3.0.0.txt` in `MVA/` and the `datasets` package (`pip install datasets`).

### 4. Generate tables

```bash
python MVA/build_bucket_table.py
```

Outputs LaTeX, CSV, and Markdown tables to `MVA/tables/`.

## Key Findings

1. **BERT-base-cased achieves the best rare-sense clustering** across both corpora, consistent with the original paper.
2. **Qwen2.5-3B (3B params, decoder-only) matches BERT on common senses but significantly underperforms on rare senses** despite being 27x larger. This shows architecture matters more than scale for lexical sense structure.
3. **RoBERTa underperforms BERT on CL-Res rare senses** despite stronger GLUE scores, but performs comparably on OntoNotes. The gap varies by corpus.
4. **GPT-2 consistently performs worst**, confirming autoregressive decoders are less effective at encoding word sense distinctions.
5. **BERT embeddings do not separate sentiment polarity** — intra-polarity distances are similar to inter-polarity distances for most opinion-bearing words.

## Metrics

| Metric | Description |
|---|---|
| precision@k | Fraction of top-k results with correct sense |
| recall@k | Fraction of all correct instances retrieved in top-k |
| F1@k | Harmonic mean of precision and recall |
| hit@k | Whether any correct instance appears in top-k |
| MRR | Mean reciprocal rank of first correct result |
| intra-sense distance | Mean distance from instances to their sense centroid |
| inter-sense nearest distance | Mean distance between competing sense centroids (same lemma) |
| inter/intra ratio | How well separated senses are relative to their spread |
| centroid margin | inter - intra distance (positive = well separated) |

## Citation

```bibtex
@inproceedings{gessler-schneider-2021-bert,
    title = "{BERT} Has Uncommon Sense: Similarity Ranking for Word Sense {BERT}ology",
    author = "Gessler, Luke and Schneider, Nathan",
    booktitle = "Proceedings of the Fourth BlackboxNLP Workshop",
    year = "2021",
    url = "https://aclanthology.org/2021.blackboxnlp-1.43",
    pages = "539--547"
}
```
