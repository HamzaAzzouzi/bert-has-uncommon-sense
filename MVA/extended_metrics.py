"""Generate extended retrieval and geometry metrics from an existing trial output."""

import os

import click
import pandas as pd

from bssp.common import paths
from bssp.common.analysis import geometry_by_bucket, metrics_at_k
from bssp.common.config import Config
from bssp.common.pickle import pickle_read
from main import get_lemma_f, read_datasets, read_stats


def mean_average(metric_dict):
    total = 0.0
    for _, values in metric_dict.items():
        total += values["label"]
    return total / len(metric_dict)


def get_f1d(precd, recd):
    scores = {}
    for i in range(1, len(precd) + 1):
        precision = precd[i]["label"]
        recall = recd[i]["label"]
        scores[i] = {"label": 0.0 if precision == 0.0 or recall == 0.0 else 2 / (1 / recall + 1 / precision)}
    return scores


def scalar_or_blank(value):
    return "" if value is None else value


def finetuning_count_from_path(override_weights):
    if override_weights is None:
        return 0
    return override_weights[override_weights.rfind("_") + 1 : override_weights.rfind(".")]


def write_extended_row(output_path, cfg, prec, rec, hit, mrr, nsd, fsd, geometry, bucket_name):
    if not os.path.isfile(output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(
                "\t".join(
                    [
                        "bucket",
                        "corpus",
                        "model",
                        "bert_layers",
                        "finetuning_count",
                        "mean_precision",
                        "mean_recall",
                        "mean_f1",
                        "mean_hit",
                        "mean_mrr",
                        "mean_nearest_same_distance",
                        "mean_farthest_same_distance",
                        "geom_intra_mean",
                        "geom_inter_nearest_mean",
                        "geom_inter_intra_ratio",
                        "geom_centroid_margin",
                    ]
                )
                + "\n"
            )

    f1d = get_f1d(prec, rec)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(
            "\t".join(
                str(x)
                for x in [
                    bucket_name,
                    cfg.corpus_name,
                    cfg.embedding_model,
                    ",".join(str(x) for x in cfg.bert_layers) if cfg.bert_layers is not None else "",
                    finetuning_count_from_path(cfg.override_weights_path),
                    mean_average(prec),
                    mean_average(rec),
                    mean_average(f1d),
                    scalar_or_blank(mean_average(hit) if hit else None),
                    scalar_or_blank(mean_average(mrr) if mrr else None),
                    scalar_or_blank(mean_average(nsd) if nsd else None),
                    scalar_or_blank(mean_average(fsd) if fsd else None),
                    scalar_or_blank(geometry.get("intra_mean") if geometry else None),
                    scalar_or_blank(geometry.get("inter_nearest_mean") if geometry else None),
                    scalar_or_blank(geometry.get("inter_intra_ratio") if geometry else None),
                    scalar_or_blank(geometry.get("centroid_margin") if geometry else None),
                ]
            )
            + "\n"
        )


@click.command(help="Write extended metrics from an existing trial TSV and cached train embeddings.")
@click.argument("corpus_slug")
@click.option("--embedding-model", default="bert-base-cased", help="transformers model slug to use")
@click.option("--metric", default="cosine", type=click.Choice(["euclidean", "cosine", "baseline"], case_sensitive=False))
@click.option("--override-weights", help="Path to override weights from fine-tuning to use with the model")
@click.option("--top-n", type=int, default=50)
@click.option("--query-n", type=int, default=1)
@click.option("--bert-layer", type=int, default=7)
def cli(corpus_slug, embedding_model, metric, override_weights, top_n, query_n, bert_layer):
    cfg = Config(
        corpus_slug,
        embedding_model=embedding_model,
        override_weights_path=override_weights,
        metric=metric,
        top_n=top_n,
        query_n=query_n,
        bert_layers=[bert_layer],
    )

    label_freqs, lemma_freqs = read_stats(cfg)
    lemma_f = get_lemma_f(cfg)
    train_dataset, _ = read_datasets(cfg)
    df = pd.read_csv(paths.predictions_tsv_path(cfg), sep="\t", on_bad_lines="skip")

    output_path = os.path.join(paths.model_dir(cfg), "extended_results.tsv")
    buckets = [
        ("low_freq_low_rarity", 5, 500, 0.0, 0.25),
        ("low_freq_all_rarity", 5, 500, 0.25, 1e9),
        ("high_freq_low_rarity", 500, 1e9, 0.0, 0.25),
        ("high_freq_all_rarity", 500, 1e9, 0.25, 1e9),
    ]

    for bucket_name, min_train_freq, max_train_freq, min_rarity, max_rarity in buckets:
        metrics_at_k(
            cfg,
            df,
            label_freqs,
            lemma_freqs,
            lemma_f,
            min_train_freq=min_train_freq,
            max_train_freq=max_train_freq,
            min_rarity=min_rarity,
            max_rarity=max_rarity,
        )
        geometry = geometry_by_bucket(
            cfg,
            train_dataset,
            label_freqs,
            lemma_freqs,
            lemma_f,
            min_train_freq=min_train_freq,
            max_train_freq=max_train_freq,
            min_rarity=min_rarity,
            max_rarity=max_rarity,
        )

        def pathf(ext):
            return paths.bucketed_metric_at_k_path(cfg, min_train_freq, max_train_freq, min_rarity, max_rarity, ext)

        prec = pickle_read(pathf("prec"))
        rec = pickle_read(pathf("rec"))
        hit = pickle_read(pathf("hit"))
        mrr = pickle_read(pathf("mrr"))
        nsd = pickle_read(pathf("nsd"))
        fsd = pickle_read(pathf("fsd"))
        write_extended_row(output_path, cfg, prec, rec, hit, mrr, nsd, fsd, geometry, bucket_name)


if __name__ == "__main__":
    cli()
