#!/usr/bin/env python3
import argparse
import glob
import os
import pandas as pd
from typing import Optional

BUCKET_ORDER = [
    "l<500 | r<0.25",
    "l<500 | r>=0.25",
    "l>=500 | r<0.25",
    "l>=500 | r>=0.25",
]

DEFAULT_MODEL_ORDER = [
    "Baseline",
    "Oracle",
    "bert-base-cased",
    "distilbert-base-cased",
    "roberta-base",
    "distilroberta-base",
    "albert-base-v2",
    "xlnet-base-cased",
    "gpt2",
]

DISPLAY_NAMES = {
    "random baseline": "Baseline",
    "oracle": "Oracle",
}


def infer_sep(path: str) -> str:
    if path.endswith(".tsv") or path.endswith(".tscv"):
        return "\t"
    return ","


def detect_bucket(path: str) -> Optional[str]:
    base = os.path.basename(path)
    if "high_freq_low_rarity" in base:
        return "l>=500 | r<0.25"
    if "high_freq_all_rarity" in base:
        return "l>=500 | r>=0.25"
    if "low_freq_low_rarity" in base:
        return "l<500 | r<0.25"
    if "low_freq_all_rarity" in base:
        return "l<500 | r>=0.25"
    return None


def prettify_model(model_name: str) -> str:
    if not isinstance(model_name, str):
        return str(model_name)
    return DISPLAY_NAMES.get(model_name.strip().lower(), model_name)


def reorder_index(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [m for m in DEFAULT_MODEL_ORDER if m in df.index]
    remaining = [m for m in df.index if m not in ordered]
    return df.reindex(ordered + sorted(remaining))


def maybe_percentify(series: pd.Series) -> pd.Series:
    max_val = series.max(skipna=True)
    min_val = series.min(skipna=True)
    if pd.notna(max_val) and pd.notna(min_val) and min_val >= 0 and max_val <= 1.0:
        return series * 100.0
    return series


def build_setting_table(data: pd.DataFrame, model_col: str, count_col: str, metrics: list[str], setting: str, best_agg: str) -> pd.DataFrame:
    if setting == "no_finetune":
        work = data[data[count_col] == 0].copy()
        grouped = work.groupby([model_col, "bucket"], as_index=False)[metrics].mean(numeric_only=True)
    else:
        grouped_obj = data.groupby([model_col, "bucket"], as_index=False)[metrics]
        if best_agg == "max":
            grouped = grouped_obj.max(numeric_only=True)
        elif best_agg == "mean":
            grouped = grouped_obj.mean(numeric_only=True)
        else:
            grouped = grouped_obj.median(numeric_only=True)

    for metric in metrics:
        if metric in grouped.columns:
            grouped[metric] = maybe_percentify(grouped[metric])

    grouped["setting"] = setting
    return grouped


def write_dataframe_variants(df: pd.DataFrame, out_base: str):
    with open(out_base + ".md", "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    with open(out_base + ".tex", "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format="%.2f", na_rep="--"))
    df.to_csv(out_base + ".csv", index=False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", default="MVA/results/*.tsv", help='Ex: "MVA/results/*.tsv"')
    p.add_argument("--model-col", default="model")
    p.add_argument("--count-col", default="finetuning_count")
    p.add_argument("--corpus", default=None, help="Filter optional, ex: clres")
    p.add_argument("--metrics", nargs="+", required=True, help="Ex: mean_f1 mean_mrr")
    p.add_argument("--best-agg", default="max", choices=["max", "mean", "median"])
    p.add_argument("--out-prefix", default="MVA/tables/mva_buckets")
    args = p.parse_args()

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(args.glob, recursive=True))
    if not files:
        raise SystemExit(f"Aucun fichier trouvé avec le glob: {args.glob}")

    dfs: list[pd.DataFrame] = []
    for f in files:
        bucket = detect_bucket(f)
        if bucket is None:
            continue

        df = pd.read_csv(f, sep=infer_sep(f), engine="python")
        needed = [args.model_col, args.count_col]
        if any(col not in df.columns for col in needed):
            continue

        if args.corpus and "corpus" in df.columns:
            df = df[df["corpus"] == args.corpus]

        if df.empty:
            continue

        keep_metrics = [m for m in args.metrics if m in df.columns]
        if not keep_metrics:
            continue

        cols = [args.model_col, args.count_col] + keep_metrics
        part = df[cols].copy()
        part[args.model_col] = part[args.model_col].map(prettify_model)
        part["bucket"] = bucket
        dfs.append(part)

    if not dfs:
        raise SystemExit("Aucun fichier exploitable (colonnes manquantes ou métriques absentes).")

    data = pd.concat(dfs, ignore_index=True)

    for metric in args.metrics:
        if metric not in data.columns:
            continue

        work = data[[args.model_col, args.count_col, "bucket", metric]].dropna().copy()
        if work.empty:
            continue

        no_ft = work[work[args.count_col] == 0]
        if not no_ft.empty:
            no_ft_agg = no_ft.groupby([args.model_col, "bucket"], as_index=False)[metric].mean()
            no_ft_agg[metric] = maybe_percentify(no_ft_agg[metric])
            no_ft_piv = no_ft_agg.pivot(index=args.model_col, columns="bucket", values=metric)
            no_ft_piv = no_ft_piv.reindex(columns=BUCKET_ORDER)
            no_ft_piv = reorder_index(no_ft_piv).round(2)

            noft_md = f"{args.out_prefix}_{metric}_no_finetune.md"
            noft_tex = f"{args.out_prefix}_{metric}_no_finetune.tex"
            noft_csv = f"{args.out_prefix}_{metric}_no_finetune.csv"
            with open(noft_md, "w", encoding="utf-8") as f:
                f.write(no_ft_piv.to_markdown())
            with open(noft_tex, "w", encoding="utf-8") as f:
                f.write(no_ft_piv.to_latex(float_format="%.2f", na_rep="--"))
            no_ft_piv.to_csv(noft_csv, index=True)
            print(f"[ok] {metric} no-finetune: {noft_md} | {noft_tex} | {noft_csv}")

        grouped = work.groupby([args.model_col, "bucket"], as_index=False)[metric]
        if args.best_agg == "max":
            best_agg = grouped.max()
        elif args.best_agg == "mean":
            best_agg = grouped.mean()
        else:
            best_agg = grouped.median()

        best_agg[metric] = maybe_percentify(best_agg[metric])
        best_piv = best_agg.pivot(index=args.model_col, columns="bucket", values=metric)
        best_piv = best_piv.reindex(columns=BUCKET_ORDER)
        best_piv = reorder_index(best_piv).round(2)

        best_md = f"{args.out_prefix}_{metric}_best_trials.md"
        best_tex = f"{args.out_prefix}_{metric}_best_trials.tex"
        best_csv = f"{args.out_prefix}_{metric}_best_trials.csv"
        with open(best_md, "w", encoding="utf-8") as f:
            f.write(best_piv.to_markdown())
        with open(best_tex, "w", encoding="utf-8") as f:
            f.write(best_piv.to_latex(float_format="%.2f", na_rep="--"))
        best_piv.to_csv(best_csv, index=True)

        print(f"[ok] {metric} best-trials: {best_md} | {best_tex} | {best_csv}")

    # All-metrics recap tables.
    metrics_available = [m for m in args.metrics if m in data.columns]
    if metrics_available:
        no_ft_table = build_setting_table(
            data,
            model_col=args.model_col,
            count_col=args.count_col,
            metrics=metrics_available,
            setting="no_finetune",
            best_agg=args.best_agg,
        )
        best_table = build_setting_table(
            data,
            model_col=args.model_col,
            count_col=args.count_col,
            metrics=metrics_available,
            setting="best_trials",
            best_agg=args.best_agg,
        )

        recap = pd.concat([no_ft_table, best_table], ignore_index=True)
        recap = recap.rename(columns={args.model_col: "model"})

        model_rank = {m: i for i, m in enumerate(DEFAULT_MODEL_ORDER)}
        bucket_rank = {b: i for i, b in enumerate(BUCKET_ORDER)}
        setting_rank = {"no_finetune": 0, "best_trials": 1}
        recap["_model_rank"] = recap["model"].map(lambda x: model_rank.get(x, 999))
        recap["_bucket_rank"] = recap["bucket"].map(lambda x: bucket_rank.get(x, 999))
        recap["_setting_rank"] = recap["setting"].map(lambda x: setting_rank.get(x, 999))
        recap = recap.sort_values(["_setting_rank", "_model_rank", "_bucket_rank", "model", "bucket"]).drop(
            columns=["_model_rank", "_bucket_rank", "_setting_rank"]
        )
        recap = recap.round(2)

        recap_cols = ["setting", "model", "bucket"] + metrics_available
        recap = recap[recap_cols]

        recap_long_base = f"{args.out_prefix}_all_metrics_recap"
        write_dataframe_variants(recap, recap_long_base)

        recap_wide = recap.set_index(["setting", "model", "bucket"])[metrics_available].unstack("bucket")
        desired_cols = [
            (metric, bucket)
            for metric in metrics_available
            for bucket in BUCKET_ORDER
            if (metric, bucket) in recap_wide.columns
        ]
        recap_wide = recap_wide.loc[:, desired_cols]
        recap_wide.columns = [f"{metric} | {bucket}" for metric, bucket in recap_wide.columns]
        recap_wide = recap_wide.reset_index().round(2)

        recap_wide_base = f"{args.out_prefix}_all_metrics_wide"
        write_dataframe_variants(recap_wide, recap_wide_base)

        print(
            "[ok] all-metrics recap: "
            f"{recap_long_base}.md | {recap_long_base}.tex | {recap_long_base}.csv"
        )
        print(
            "[ok] all-metrics wide: "
            f"{recap_wide_base}.md | {recap_wide_base}.tex | {recap_wide_base}.csv"
        )

if __name__ == "__main__":
    main()