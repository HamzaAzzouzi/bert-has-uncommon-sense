"""
Run nearest-neighbor retrieval on pre-extracted embeddings from a modern model
and compute metrics using the existing bssp analysis pipeline.

Usage:
    python modern_models/run_trial.py \
        --model Qwen/Qwen2.5-3B --corpus clres [--top-n 50]
"""

import csv
import os
import pickle
from collections import Counter, defaultdict
import click
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

from bssp.common import paths
from bssp.common.analysis import metrics_at_k, geometry_by_bucket
from bssp.common.config import Config
from bssp.common.pickle import pickle_read
from modern_models.extract_embeddings import (
    SimpleInstance, SimpleTextField, SimpleToken,
    SimpleSpanField, SimpleLabelField, SimpleArrayField,
)


def lemma_from_label(label):
    return label[: label.rfind("_")]


def format_sentence(words, i, j):
    tokens = [t.text if hasattr(t, "text") else t for t in words]
    return " ".join(tokens[:i] + [">>" + tokens[i] + "<<"] + tokens[j + 1:])


def build_embedding_matrix(dataset):
    """Stack all span_embeddings into a (N, hidden_dim) tensor."""
    embeddings = []
    for inst in dataset:
        emb = inst["span_embeddings"].array
        if emb.ndim == 2:
            emb = emb.mean(axis=0)
        embeddings.append(emb)
    return torch.tensor(np.stack(embeddings, axis=0))


def build_lemma_index(dataset):
    """Map each lemma string to the list of instance indices that have it."""
    index = defaultdict(list)
    for i, inst in enumerate(dataset):
        lemma = lemma_from_label(inst["label"].label)
        index[lemma].append(i)
    return index


def retrieve_top_n(query_embedding, train_embeddings, lemma_indices, top_n, device):
    """
    Find the top_n most similar train instances (by cosine similarity)
    restricted to instances with the same lemma.
    """
    target = train_embeddings[lemma_indices].to(device)
    query = query_embedding.unsqueeze(0).to(device)
    distances = 1 - cosine_similarity(target, query)
    ranked = torch.argsort(distances, descending=False)

    results = []
    for idx in ranked[:top_n]:
        original_idx = lemma_indices[idx.item()]
        dist = distances[idx].item()
        results.append((original_idx, dist))
    return results


def write_predictions(cfg, train_dataset, test_dataset, train_embeddings, output_path, device):
    """Run retrieval for all test instances and write prediction TSV."""
    lemma_index = build_lemma_index(train_dataset)

    # Count label frequencies in train
    label_counts = Counter()
    for inst in train_dataset:
        label_counts[inst["label"].label] += 1

    # Filter: only test instances whose label appeared >= 5 times in train
    instances = [inst for inst in test_dataset if label_counts.get(inst["label"].label, 0) >= 5]
    print(f"Evaluating {len(instances)} test instances (filtered from {len(test_dataset)})")

    with open(output_path, "wt") as f:
        writer = csv.writer(f, delimiter="\t")
        header = ["sentence", "label", "lemma", "label_freq_in_train"]
        header += [f"label_{i+1}" for i in range(cfg.top_n)]
        header += [f"lemma_{i+1}" for i in range(cfg.top_n)]
        header += [f"sentence_{i+1}" for i in range(cfg.top_n)]
        header += [f"distance_{i+1}" for i in range(cfg.top_n)]
        writer.writerow(header)

        for inst in tqdm(instances, desc="Retrieving"):
            label = inst["label"].label
            lemma = lemma_from_label(label)
            lemma_indices = lemma_index.get(lemma, [])
            if not lemma_indices:
                continue

            # Get query embedding
            query_emb = inst["span_embeddings"].array
            if query_emb.ndim == 2:
                query_emb = query_emb.mean(axis=0)
            query_tensor = torch.tensor(query_emb, dtype=torch.float32)

            results = retrieve_top_n(query_tensor, train_embeddings, lemma_indices, cfg.top_n, device)
            results += [None] * (cfg.top_n - len(results))

            span = inst["label_span"]
            sentence_str = format_sentence(inst["text"].tokens, span.span_start, span.span_end)
            row = [sentence_str, label, lemma, label_counts[label]]

            labels, lemmas, sentences, distances = [], [], [], []
            for result in results:
                if result is None:
                    distances.append(88888888)
                    labels.append("")
                    lemmas.append("")
                    sentences.append("")
                else:
                    idx, dist = result
                    train_inst = train_dataset[idx]
                    labels.append(train_inst["label"].label)
                    lemmas.append(lemma_from_label(labels[-1]))
                    train_span = train_inst["label_span"]
                    sentences.append(format_sentence(
                        train_inst["text"].tokens,
                        train_span.span_start,
                        train_span.span_end
                    ))
                    distances.append(dist)

            row += labels + lemmas + sentences + distances
            if len(row) != 4 + 4 * cfg.top_n:
                print(f"Warning: row length {len(row)} != expected {4 + 4 * cfg.top_n}")
                continue
            writer.writerow(row)

    print(f"Wrote predictions to {output_path}")


def compute_stats(dataset, corpus_name, split):
    """Write label/lemma frequency TSVs, return counters."""
    label_counts = Counter()
    lemma_counts = Counter()
    for inst in dataset:
        label = inst["label"].label
        lemma = lemma_from_label(label)
        label_counts[label] += 1
        lemma_counts[lemma] += 1

    stats_dir = f"{corpus_name}_stats"
    from bssp.common import paths as p
    label_path = p.freq_tsv_path(stats_dir, split, "label")
    lemma_path = p.freq_tsv_path(stats_dir, split, "lemma")

    with open(label_path, "w") as f:
        for item, freq in sorted(label_counts.items(), key=lambda x: -x[1]):
            f.write(f"{item}\t{freq}\n")
    with open(lemma_path, "w") as f:
        for item, freq in sorted(lemma_counts.items(), key=lambda x: -x[1]):
            f.write(f"{item}\t{freq}\n")

    return label_counts, lemma_counts


@click.command()
@click.option("--model", "model_name", required=True, help="HuggingFace model name")
@click.option("--corpus", required=True, type=click.Choice(["clres", "ontonotes"]))
@click.option("--top-n", type=int, default=50)
@click.option("--input-dir", default="cache/modern_models", help="Where pickled datasets live")
@click.option("--device", default="auto")
def main(model_name, corpus, top_n, input_dir, device):
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_slug = model_name.replace("/", "_")

    # Load pre-extracted datasets
    train_path = os.path.join(input_dir, f"{corpus}_{model_slug}_train.pkl")
    test_path = os.path.join(input_dir, f"{corpus}_{model_slug}_test.pkl")
    print(f"Loading datasets from {input_dir}...")
    with open(train_path, "rb") as f:
        train_dataset = pickle.load(f)
    with open(test_path, "rb") as f:
        test_dataset = pickle.load(f)
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Build train embedding matrix
    print("Building embedding matrix...")
    train_embeddings = build_embedding_matrix(train_dataset).float()

    # Config for path generation and metrics
    cfg = Config(
        corpus,
        embedding_model=model_name,
        metric="cosine",
        top_n=top_n,
        query_n=1,
        bert_layers=None,  # not applicable for modern models
    )

    # Write stats
    compute_stats(train_dataset, corpus, "train")
    compute_stats(test_dataset, corpus, "test")

    # Read back stats for metrics
    readf = lambda f_obj: {k: int(v) for k, v in map(lambda l: l.strip().split("\t"), f_obj)}
    with open(paths.freq_tsv_path(f"{corpus}_stats", "train", "label"), "r") as f:
        label_freqs = readf(f)
    with open(paths.freq_tsv_path(f"{corpus}_stats", "train", "lemma"), "r") as f:
        lemma_freqs = readf(f)

    # Predictions
    predictions_path = paths.predictions_tsv_path(cfg)
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

    if not os.path.isfile(predictions_path):
        write_predictions(cfg, train_dataset, test_dataset, train_embeddings, predictions_path, device)
    else:
        print(f"Predictions already exist at {predictions_path}")

    # Compute metrics
    df = pd.read_csv(predictions_path, sep="\t", on_bad_lines="skip")

    low_freq, high_freq = (5, 500), (500, 1e9)
    low_rarity, high_rarity = (0.0, 0.25), (0.25, 1.0)

    for min_train_freq, max_train_freq in [low_freq, high_freq]:
        for min_rarity, max_rarity in [low_rarity, high_rarity]:
            print(f"Bucket: freq=[{min_train_freq},{max_train_freq}), rarity=[{min_rarity},{max_rarity})")
            metrics_at_k(
                cfg, df, label_freqs, lemma_freqs, lemma_from_label,
                min_train_freq=min_train_freq, max_train_freq=max_train_freq,
                min_rarity=min_rarity, max_rarity=max_rarity,
            )
            geometry = geometry_by_bucket(
                cfg, train_dataset, label_freqs, lemma_freqs, lemma_from_label,
                min_train_freq=min_train_freq, max_train_freq=max_train_freq,
                min_rarity=min_rarity, max_rarity=max_rarity,
            )

            def pathf(ext):
                return paths.bucketed_metric_at_k_path(
                    cfg, min_train_freq, max_train_freq, min_rarity, max_rarity, ext
                )

            def mean_average(d):
                return sum(v["label"] for v in d.values()) / len(d)

            def get_f1d(precd, recd):
                d = {}
                for i in range(1, cfg.top_n + 1):
                    p, r = precd[i]["label"], recd[i]["label"]
                    d[i] = {"label": 0.0 if p == 0 or r == 0 else 2 / (1 / r + 1 / p)}
                return d

            def scalar_or_blank(value):
                return "" if value is None else value

            prec = pickle_read(pathf("prec"))
            rec = pickle_read(pathf("rec"))
            hit = pickle_read(pathf("hit"))
            mrr = pickle_read(pathf("mrr"))
            nsd = pickle_read(pathf("nsd"))
            fsd = pickle_read(pathf("fsd"))

            if prec is None or rec is None:
                continue

            f1d = get_f1d(prec, rec)

            fname_prefix = ""
            fname_prefix += "low_freq" if min_train_freq == low_freq[0] else "high_freq"
            fname_prefix += "_"
            fname_prefix += "low_rarity" if min_rarity == low_rarity[0] else "all_rarity"
            fname_prefix += "_"

            # Append to standard results
            with open(fname_prefix + "results.tsv", "a") as f:
                vals = [corpus, model_name, "", 0,
                        mean_average(prec), mean_average(rec), mean_average(f1d)]
                f.write("\t".join(str(x) for x in vals) + "\n")

            # Append to extended results
            extended_path = fname_prefix + "extended_results.tsv"
            geom_intra = geometry.get("intra_mean") if geometry else None
            geom_inter = geometry.get("inter_nearest_mean") if geometry else None
            geom_ratio = geometry.get("inter_intra_ratio") if geometry else None
            geom_margin = geometry.get("centroid_margin") if geometry else None

            if not os.path.isfile(extended_path):
                with open(extended_path, "w") as f:
                    f.write("\t".join([
                        "corpus", "model", "bert_layers", "finetuning_count",
                        "mean_precision", "mean_recall", "mean_f1",
                        "mean_hit", "mean_mrr",
                        "mean_nearest_same_distance", "mean_farthest_same_distance",
                        "geom_intra_mean", "geom_inter_nearest_mean",
                        "geom_inter_intra_ratio", "geom_centroid_margin",
                    ]) + "\n")

            with open(extended_path, "a") as f:
                vals = [
                    corpus, model_name, "", 0,
                    mean_average(prec), mean_average(rec), mean_average(f1d),
                    scalar_or_blank(mean_average(hit) if hit else None),
                    scalar_or_blank(mean_average(mrr) if mrr else None),
                    scalar_or_blank(mean_average(nsd) if nsd else None),
                    scalar_or_blank(mean_average(fsd) if fsd else None),
                    scalar_or_blank(geom_intra), scalar_or_blank(geom_inter),
                    scalar_or_blank(geom_ratio), scalar_or_blank(geom_margin),
                ]
                f.write("\t".join(str(x) for x in vals) + "\n")

    print("Done. Results appended to *_results.tsv and *_extended_results.tsv")


if __name__ == "__main__":
    main()
