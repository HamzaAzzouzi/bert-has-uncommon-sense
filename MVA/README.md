# MVA

This folder contains material added for the MVA analysis work.

## Files in this folder

- `extended_metrics.py`
  Builds additional evaluation outputs from an existing `trial` run.
  It reuses the prediction TSV and cached train embeddings already produced by the main pipeline.

## What `extended_metrics.py` computes

The script aggregates the usual retrieval metrics and adds extra geometric analyses:

- `precision@k`
- `recall@k`
- `F1@k` derived from precision and recall
- `hit@k`
- `MRR`
- mean distance to the nearest correct neighbor in the top-k list
- mean distance to the farthest correct neighbor in the top-k list
- intra-sense mean distance
- nearest inter-sense mean distance
- inter/intra distance ratio
- centroid margin (`inter - intra`)

It writes a tabular summary file named `extended_results.tsv` inside the same cache directory as the corresponding run.

## Where the original metrics are stored

The existing pipeline already writes metrics outside `MVA/`.

### Prediction TSV

For a run with a given corpus / model / layer / checkpoint, the raw nearest-neighbor predictions are stored in:

- `cache/<corpus>_<metric>_q<query_n>_predictions/*.tsv`

Example:

- `cache/clres_cosine_q1_predictions/distilbert-base-cased_models__distilbert-base-cased_100.pt_5.tsv`

### Bucketed metric pickles

For each frequency / rarity bucket, the pipeline stores pickle files in the same cache directory:

- `.prec` for precision@k
- `.rec` for recall@k
- `.orec` for oracle recall
- `.oprec` for oracle precision
- `.lemmas` for the lemmas present in the bucket
- `.count` for the number of evaluated instances

The extended metric support added in this work also uses / writes:

- `.hit` for hit@k
- `.mrr` for mean reciprocal rank
- `.nsd` for nearest same-sense distance
- `.fsd` for farthest same-sense distance
- `.geom` for geometry summaries per bucket

### High-level summary TSVs

The original repo also writes coarse summaries at the project root:

- `low_freq_low_rarity_results.tsv`
- `low_freq_all_rarity_results.tsv`
- `high_freq_low_rarity_results.tsv`
- `high_freq_all_rarity_results.tsv`

Those files contain the standard aggregated outputs produced by `summarize`.

## Code paths used by the extended analysis

The script relies on the existing pipeline code in:

- `bssp/common/analysis.py`
- `main.py`
- `bssp/common/paths.py`

## Typical workflow

1. Run `python main.py trial ...` to generate prediction TSVs.
2. Run `python main.py summarize ...` if you want the repo's standard summaries.
3. Run `python MVA/extended_metrics.py ...` to generate the extended MVA-oriented analysis file.
