from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

from bssp.common import paths
from bssp.common.const import NOTA_SENSES
from bssp.common.pickle import pickle_write


def _iter_bucket_rows(cfg, df, label_freqs, lemma_freqs, lemma_f, min_train_freq, max_train_freq, min_rarity, max_rarity):
    for _, row in tqdm(df.iterrows()):
        label = row.label
        lemma = lemma_f(label)

        if cfg.corpus_name == "ontonotes" and label in NOTA_SENSES:
            continue
        if cfg.corpus_name == "semcor" and (label == "NE" or "_" not in label):
            continue

        rarity = label_freqs[label] / lemma_freqs[lemma]
        if not (min_rarity <= rarity < max_rarity):
            continue
        if not (min_train_freq <= row.label_freq_in_train < max_train_freq):
            continue

        yield row, label, lemma


def metrics_at_k(
    cfg,
    df,
    label_freqs,
    lemma_freqs,
    lemma_f,
    min_train_freq,
    max_train_freq,
    min_rarity,
    max_rarity,
    query_category=None,
    pos=None,
):
    def zero_dict():
        return defaultdict(float)

    score_dict = defaultdict(zero_dict)

    count = 0
    lemmas = set()
    for row, label, lemma in _iter_bucket_rows(
        cfg, df, label_freqs, lemma_freqs, lemma_f, min_train_freq, max_train_freq, min_rarity, max_rarity
    ):
        count += 1
        lemmas.add(lemma)

        num_labels_correct = 0
        num_lemmas_correct = 0
        reciprocal_rank = 0.0
        same_sense_distances = []

        for k in range(1, cfg.top_n + 1):
            label_is_correct = getattr(row, f"label_{k}") == label
            lemma_is_correct = getattr(row, f"lemma_{k}") == lemma

            if label_is_correct:
                distance = float(getattr(row, f"distance_{k}"))
                same_sense_distances.append(distance)
                if reciprocal_rank == 0.0:
                    reciprocal_rank = 1.0 / k

            num_labels_correct += label_is_correct
            num_lemmas_correct += lemma_is_correct

            score_dict[k]["label"] += num_labels_correct
            score_dict[k]["lemma"] += num_lemmas_correct

            score_dict[k]["total"] += k
            score_dict[k]["label_total"] += label_freqs[label]
            score_dict[k]["lemma_total"] += lemma_freqs[lemma]

            score_dict[k]["oracle_recall"] += min(k, label_freqs[label])
            score_dict[k]["oracle_precision"] += min(k, label_freqs[label])

            score_dict[k]["hit"] += float(num_labels_correct > 0)
            score_dict[k]["mrr"] += reciprocal_rank

            if same_sense_distances:
                score_dict[k]["nearest_same_distance_sum"] += min(same_sense_distances)
                score_dict[k]["nearest_same_distance_count"] += 1
                score_dict[k]["farthest_same_distance_sum"] += max(same_sense_distances)
                score_dict[k]["farthest_same_distance_count"] += 1

    if count == 0:
        print("No instances in this bin, skipping")
        return None, None

    ps_at_k = defaultdict(lambda: dict())
    recalls_at_k = defaultdict(lambda: dict())
    hits_at_k = defaultdict(lambda: dict())
    mrr_at_k = defaultdict(lambda: dict())
    nearest_same_distance_at_k = defaultdict(lambda: dict())
    farthest_same_distance_at_k = defaultdict(lambda: dict())

    for k in range(1, cfg.top_n + 1):
        for label_key in ["label", "lemma"]:
            ps_at_k[k][label_key] = score_dict[k][label_key] / score_dict[k]["total"]
            recalls_at_k[k][label_key] = score_dict[k][label_key] / score_dict[k][f"{label_key}_total"]

        hits_at_k[k]["label"] = score_dict[k]["hit"] / count
        mrr_at_k[k]["label"] = score_dict[k]["mrr"] / count

        nearest_count = score_dict[k]["nearest_same_distance_count"]
        farthest_count = score_dict[k]["farthest_same_distance_count"]
        nearest_same_distance_at_k[k]["label"] = (
            score_dict[k]["nearest_same_distance_sum"] / nearest_count if nearest_count else None
        )
        farthest_same_distance_at_k[k]["label"] = (
            score_dict[k]["farthest_same_distance_sum"] / farthest_count if farthest_count else None
        )

    oracle_recalls_at_k = defaultdict(lambda: dict())
    oracle_precisions_at_k = defaultdict(lambda: dict())
    for k in range(1, cfg.top_n + 1):
        oracle_recalls_at_k[k]["label"] = score_dict[k]["oracle_recall"] / score_dict[k]["label_total"]
        oracle_precisions_at_k[k]["label"] = score_dict[k]["oracle_precision"] / score_dict[k]["total"]

    def path_f(ext):
        return paths.bucketed_metric_at_k_path(
            cfg, min_train_freq, max_train_freq, min_rarity, max_rarity, ext, query_category=query_category, pos=pos
        )

    def dump_metric(metric, ext):
        metric = dict(metric)
        for key, value in metric.items():
            metric[key] = dict(value)
        pickle_write(metric, path_f(ext))
        return metric

    ps_at_k = dump_metric(ps_at_k, "prec")
    recalls_at_k = dump_metric(recalls_at_k, "rec")
    hits_at_k = dump_metric(hits_at_k, "hit")
    mrr_at_k = dump_metric(mrr_at_k, "mrr")
    nearest_same_distance_at_k = dump_metric(nearest_same_distance_at_k, "nsd")
    farthest_same_distance_at_k = dump_metric(farthest_same_distance_at_k, "fsd")
    oracle_recalls_at_k = dump_metric(oracle_recalls_at_k, "orec")
    oracle_precisions_at_k = dump_metric(oracle_precisions_at_k, "oprec")

    pickle_write(lemmas, path_f("lemmas"))
    pickle_write(count, path_f("count"))

    return ps_at_k, recalls_at_k


def geometry_by_bucket(
    cfg,
    train_dataset,
    label_freqs,
    lemma_freqs,
    lemma_f,
    min_train_freq,
    max_train_freq,
    min_rarity,
    max_rarity,
    query_category=None,
    pos=None,
):
    embeddings_by_label = defaultdict(list)
    label_to_lemma = {}

    for instance in train_dataset:
        label = instance["label"].label
        lemma = lemma_f(label)
        rarity = label_freqs[label] / lemma_freqs[lemma]

        if not (min_rarity <= rarity < max_rarity):
            continue
        if not (min_train_freq <= label_freqs[label] < max_train_freq):
            continue
        if "span_embeddings" not in instance:
            continue

        embeddings = np.asarray(instance["span_embeddings"].array, dtype=float)
        if embeddings.ndim == 2:
            embedding = embeddings.mean(axis=0)
        else:
            embedding = embeddings

        embeddings_by_label[label].append(embedding)
        label_to_lemma[label] = lemma

    if not embeddings_by_label:
        print("No train embeddings in this bin, skipping geometry")
        return None

    centroids = {label: np.mean(np.stack(vectors, axis=0), axis=0) for label, vectors in embeddings_by_label.items()}

    intra_total = 0.0
    intra_count = 0
    inter_total = 0.0
    inter_count = 0
    competitor_label_count = 0

    for label, vectors in embeddings_by_label.items():
        centroid = centroids[label]
        stack = np.stack(vectors, axis=0)
        intra_dists = np.linalg.norm(stack - centroid, axis=1)
        intra_total += float(intra_dists.sum())
        intra_count += int(intra_dists.shape[0])

        competitor_centroids = [
            other_centroid
            for other_label, other_centroid in centroids.items()
            if other_label != label and label_to_lemma[other_label] == label_to_lemma[label]
        ]
        if competitor_centroids:
            competitor_label_count += 1
            nearest_other = min(float(np.linalg.norm(centroid - other_centroid)) for other_centroid in competitor_centroids)
            inter_total += nearest_other * len(vectors)
            inter_count += len(vectors)

    intra_mean = intra_total / intra_count if intra_count else None
    inter_mean = inter_total / inter_count if inter_count else None

    geometry = {
        "instance_count": intra_count,
        "label_count": len(embeddings_by_label),
        "lemma_count": len(set(label_to_lemma.values())),
        "competitor_label_count": competitor_label_count,
        "intra_mean": intra_mean,
        "inter_nearest_mean": inter_mean,
        "inter_intra_ratio": (inter_mean / intra_mean) if inter_mean is not None and intra_mean not in (None, 0.0) else None,
        "centroid_margin": (inter_mean - intra_mean) if inter_mean is not None and intra_mean is not None else None,
    }

    pickle_write(
        geometry,
        paths.bucketed_metric_at_k_path(
            cfg, min_train_freq, max_train_freq, min_rarity, max_rarity, "geom", query_category=query_category, pos=pos
        ),
    )

    return geometry


def dataset_stats(split, dataset, directory, lemma_function):
    labels = Counter()
    lemmas = Counter()

    for instance in dataset:
        label = instance["label"].label
        lemma = lemma_function(label)
        labels[label] += 1
        lemmas[lemma] += 1

    with open(paths.freq_tsv_path(directory, split, "label"), "w", encoding="utf-8") as f:
        for item, freq in sorted(labels.items(), key=lambda x: -x[1]):
            f.write(f"{item}\t{freq}\n")
    with open(paths.freq_tsv_path(directory, split, "lemma"), "w", encoding="utf-8") as f:
        for item, freq in sorted(lemmas.items(), key=lambda x: -x[1]):
            f.write(f"{item}\t{freq}\n")

    return labels, lemmas
