r"""Gold-count error analysis: single-gold vs. multi-gold bugs (Cassandra30 TEST, FINAL).

Splits every Cassandra30 TEST bug by how many gold (fix-commit-changed)
source files it has - ``single_gold`` (exactly 1) vs. ``multi_gold`` (2+) -
using the dataset's own ``gold_files`` list (the exact list
``candidate_scores.metrics_from_scores`` already uses as ``gold`` for every
final Test evaluation script). ``n_gold`` is not read from a precomputed
column; it's ``len(gold_files)`` directly, so this script can't silently
drift from the list the metrics themselves are computed against.

CPU-only: reads the three already-saved TEST candidate-score tables and never
loads a model or runs inference.

- ``save_tfidf_test_scores.py`` -> ``tfidf_test_candidate_scores.parquet``.
- ``save_finetuned_test_scores.py`` -> ``finetuned_epoch{N}_test_candidate_scores.parquet``.
- Hybrid is reconstructed exactly as ``hybrid_test_experiment.py`` does: join
  the two tables on (project, key, path), per-bug min-max normalize each
  score column (``hybrid_experiment.normalize_per_bug``, unchanged), combine
  at the Validation-selected, fixed alpha=0.5
  (``hybrid_test_experiment.hybrid_scores_at_alpha``, unchanged).

All ranking metrics reuse ``candidate_scores.metrics_from_scores`` /
``metrics.py`` unchanged, so hit@k/recall@k/MRR/MAP and gold-file semantics
are identical to the final Test evaluation scripts. Only TEST tables are
read here - Validation is never read.

Usage:
    python -m bug2code.localization.gold_count_error_analysis --config configs/cassandra30.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from bug2code.config import load_config
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.localization.candidate_scores import metrics_from_scores, ranked_paths
from bug2code.localization.hybrid_experiment import load_and_join, normalize_per_bug
from bug2code.localization.hybrid_test_experiment import ALPHA, hybrid_scores_at_alpha
from bug2code.localization.metrics import (
    average_precision,
    best_gold_rank,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from bug2code.localization.save_finetuned_test_scores import (
    OUTPUT_TEMPLATE as CODEBERT_SCORES_TEMPLATE,
)
from bug2code.localization.save_tfidf_test_scores import OUTPUT_NAME as TFIDF_SCORES_NAME
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

SUMMARY_NAME = "gold_count_error_analysis_summary.csv"
PER_BUG_NAME = "gold_count_error_analysis_per_bug.csv"
EXAMPLES_NAME = "gold_count_error_analysis_examples.csv"

TOP_N_PREDICTIONS = 10
METHODS = ("tfidf", "finetuned", "hybrid")
DIAGNOSTIC_K_VALUES = (10, 20)


def classify_gold_count(n_gold: int) -> str:
    """``single_gold`` for exactly 1 gold file, ``multi_gold`` for 2+."""
    return "single_gold" if n_gold == 1 else "multi_gold"


def bug_prediction_summary(
    bug_scores: pd.DataFrame, gold: list[str], top_n: int = TOP_N_PREDICTIONS
) -> dict:
    """Rank/RR/AP/Recall@{5,10,20}/Hit@10/top-N for one bug's raw score rows."""
    ranked = ranked_paths(bug_scores) if len(bug_scores) else []
    return {
        "rank": best_gold_rank(ranked, gold),
        "rr": reciprocal_rank(ranked, gold),
        "ap": average_precision(ranked, gold),
        "recall_at_5": recall_at_k(ranked, gold, 5),
        "recall_at_10": recall_at_k(ranked, gold, 10),
        "recall_at_20": recall_at_k(ranked, gold, 20),
        "hit_at_10": hit_at_k(ranked, gold, 10),
        "top_n": ";".join(ranked[:top_n]),
    }


def build_per_bug_table(
    test_bugs: pd.DataFrame,
    tfidf_scores: pd.DataFrame,
    codebert_scores: pd.DataFrame,
    hybrid_scores: pd.DataFrame,
) -> pd.DataFrame:
    """One row per Cassandra30 TEST bug: gold-count group, per-method ranks/metrics."""
    scores_by_method = {
        "tfidf": {k: g for k, g in tfidf_scores.groupby(["project", "key"])},
        "finetuned": {k: g for k, g in codebert_scores.groupby(["project", "key"])},
        "hybrid": {k: g for k, g in hybrid_scores.groupby(["project", "key"])},
    }

    rows = []
    for bug in test_bugs.itertuples():
        key = (bug.project, bug.key)
        gold = list(bug.gold_files)
        n_gold = len(gold)

        row = {
            "project": bug.project,
            "key": bug.key,
            "title": bug.title,
            "group": classify_gold_count(n_gold),
            "n_gold": n_gold,
            "gold_files": ";".join(gold),
        }

        for method in METHODS:
            bug_scores = scores_by_method[method].get(key, pd.DataFrame())
            pred = bug_prediction_summary(bug_scores, gold)
            row[f"{method}_rank"] = pred["rank"]
            row[f"{method}_rr"] = pred["rr"]
            row[f"{method}_ap"] = pred["ap"]
            row[f"{method}_recall_at_5"] = pred["recall_at_5"]
            row[f"{method}_recall_at_10"] = pred["recall_at_10"]
            row[f"{method}_recall_at_20"] = pred["recall_at_20"]
            row[f"{method}_hit_at_10"] = pred["hit_at_10"]
            row[f"{method}_top10"] = pred["top_n"]

        rows.append(row)
    return pd.DataFrame(rows)


def _method_metrics_with_meta(
    scores: pd.DataFrame, gold: pd.DataFrame, bug_meta: pd.DataFrame
) -> pd.DataFrame:
    """``metrics_from_scores`` output joined with (group, n_gold) per bug."""
    return metrics_from_scores(scores, gold).merge(bug_meta, on=["project", "key"])


def build_summary_table(
    tfidf_scores: pd.DataFrame,
    codebert_scores: pd.DataFrame,
    hybrid_scores: pd.DataFrame,
    gold: pd.DataFrame,
    bug_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Per (method, group) mean metrics + gold-count stats, plus MRR/MAP/Recall@10 deltas.

    Delta rows share the same columns but only populate ``mrr``, ``map``, and
    ``recall_at_10`` (the deltas asked for); everything else is ``NaN`` and
    ``n_bugs``/gold-count columns are ``NA`` - they compare already-computed
    means, not a distinct bug set.
    """
    scores_by_method = {
        "tfidf": tfidf_scores,
        "finetuned": codebert_scores,
        "hybrid": hybrid_scores,
    }

    rows = []
    for method, scores in scores_by_method.items():
        per_bug = _method_metrics_with_meta(scores, gold, bug_meta)
        exclude = ("project", "key", "n_candidates", "group", "n_gold")
        metric_cols = [c for c in per_bug.columns if c not in exclude]
        for group, g in per_bug.groupby("group"):
            row = {
                "method": method,
                "group": group,
                "n_bugs": len(g),
                "mean_n_gold": g["n_gold"].mean(),
                "median_n_gold": g["n_gold"].median(),
            }
            row |= g[metric_cols].mean().to_dict()
            rows.append(row)
    summary = pd.DataFrame(rows)

    delta_metrics = ("mrr", "map", "recall_at_10")
    pivots = {m: summary.pivot(index="group", columns="method", values=m) for m in delta_metrics}
    delta_rows = []
    for group in pivots["mrr"].index:
        for other in ("finetuned", "hybrid"):
            row = {
                "method": f"delta_{other}_minus_tfidf",
                "group": group,
                "n_bugs": pd.NA,
                "mean_n_gold": pd.NA,
                "median_n_gold": pd.NA,
            }
            for metric in delta_metrics:
                row[metric] = pivots[metric].loc[group, other] - pivots[metric].loc[group, "tfidf"]
            delta_rows.append(row)
    return pd.concat([summary, pd.DataFrame(delta_rows)], ignore_index=True)


def build_multigold_diagnostic(
    tfidf_scores: pd.DataFrame,
    codebert_scores: pd.DataFrame,
    hybrid_scores: pd.DataFrame,
    gold: pd.DataFrame,
    bug_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Multi-gold-only diagnostic: how much of the multi-file fix is actually surfaced.

    ``pct_at_least_one_top{k}`` is ``mean(hit_at_k) * 100`` (MRR/Hit@k only
    need one gold file). ``mean_recall_at_{k}`` is the mean fraction of ALL
    gold files retrieved in the top k. ``pct_all_in_top{k}`` is the percentage
    of bugs where ``recall_at_k == 1.0`` - every gold file surfaced, the
    complement of what MRR alone can ever show.
    """
    scores_by_method = {
        "tfidf": tfidf_scores,
        "finetuned": codebert_scores,
        "hybrid": hybrid_scores,
    }

    rows = []
    for method, scores in scores_by_method.items():
        per_bug = _method_metrics_with_meta(scores, gold, bug_meta)
        multi = per_bug[per_bug["group"] == "multi_gold"]
        row = {"method": method, "group": "multi_gold", "n_bugs": len(multi)}
        for k in DIAGNOSTIC_K_VALUES:
            row[f"pct_at_least_one_in_top{k}"] = multi[f"hit_at_{k}"].mean() * 100
            row[f"mean_recall_at_{k}"] = multi[f"recall_at_{k}"].mean()
            row[f"pct_all_in_top{k}"] = (multi[f"recall_at_{k}"] == 1.0).mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


def build_examples_table(per_bug: pd.DataFrame, n: int) -> pd.DataFrame:
    """Systematic (not cherry-picked) multi-gold qualitative cases, 4 categories."""
    multi = per_bug[per_bug["group"] == "multi_gold"].copy()

    frames = []

    # (a) MRR high, Recall@10 low: max (rr - recall_at_10) gap across the 3
    # methods, whichever method shows it most for that bug.
    gap_cols = {m: multi[f"{m}_rr"] - multi[f"{m}_recall_at_10"] for m in METHODS}
    gaps = pd.DataFrame(gap_cols)
    multi["mrr_high_recall_low_gap"] = gaps.max(axis=1)
    multi["mrr_high_recall_low_method"] = gaps.idxmax(axis=1)
    frames.append(
        multi.nlargest(n, "mrr_high_recall_low_gap").assign(example_category="mrr_high_recall_low")
    )

    # (b) Hybrid retrieves strictly more gold files (Recall@10) than TF-IDF.
    multi["hybrid_minus_tfidf_recall_at_10"] = (
        multi["hybrid_recall_at_10"] - multi["tfidf_recall_at_10"]
    )
    frames.append(
        multi.nlargest(n, "hybrid_minus_tfidf_recall_at_10").assign(
            example_category="hybrid_beats_tfidf_recall_at_10"
        )
    )

    # (c) TF-IDF retrieves strictly more gold files (Recall@10) than Fine-tuned.
    multi["tfidf_minus_finetuned_recall_at_10"] = (
        multi["tfidf_recall_at_10"] - multi["finetuned_recall_at_10"]
    )
    frames.append(
        multi.nlargest(n, "tfidf_minus_finetuned_recall_at_10").assign(
            example_category="tfidf_beats_finetuned_recall_at_10"
        )
    )

    # (d) optional: all three methods retrieve none of the gold files in
    # Top-10 - the hardest cases, ranked by n_gold (more missed files first).
    all_struggle = multi[
        (multi["tfidf_recall_at_10"] == 0)
        & (multi["finetuned_recall_at_10"] == 0)
        & (multi["hybrid_recall_at_10"] == 0)
    ]
    if len(all_struggle):
        frames.append(
            all_struggle.nlargest(n, "n_gold").assign(example_category="all_methods_struggle")
        )

    cols = [
        "example_category",
        "project",
        "key",
        "title",
        "n_gold",
        "gold_files",
        "tfidf_rank",
        "finetuned_rank",
        "hybrid_rank",
        "tfidf_rr",
        "finetuned_rr",
        "hybrid_rr",
        "tfidf_recall_at_10",
        "finetuned_recall_at_10",
        "hybrid_recall_at_10",
        "tfidf_top10",
        "finetuned_top10",
        "hybrid_top10",
    ]
    return pd.concat(frames, ignore_index=True)[cols]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument(
        "--epoch",
        type=int,
        default=3,
        help="which saved Fine-tuned CodeBERT TEST table (selected: 3)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=10,
        help="qualitative examples per category (not cherry-picked)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    dataset = pd.read_parquet(cfg.paths.data_processed / DATASET_NAME)
    dev_keys = pd.read_parquet(cfg.paths.data_processed / cfg.dev_subset_name)
    bugs = dataset.merge(dev_keys[["project", "key"]], on=["project", "key"])
    test_bugs = bugs[bugs["split"] == "test"]
    logger.info("%d Cassandra30 TEST bugs (Validation is never read here)", len(test_bugs))

    tfidf_path = cfg.paths.tables / TFIDF_SCORES_NAME
    codebert_path = cfg.paths.tables / CODEBERT_SCORES_TEMPLATE.format(epoch=args.epoch)
    for path, label in ((tfidf_path, "TF-IDF"), (codebert_path, "Fine-tuned CodeBERT")):
        if not path.exists():
            raise FileNotFoundError(
                f"{label} Cassandra30 TEST candidate-score table not found at {path} - this "
                "script never runs inference; copy the already-saved TEST score table from "
                "Colab into this path first."
            )

    tfidf_scores = pd.read_parquet(tfidf_path)
    codebert_scores = pd.read_parquet(codebert_path)

    merged = load_and_join(tfidf_path, codebert_path)
    merged = normalize_per_bug(merged, "score_tfidf", "tfidf_norm")
    merged = normalize_per_bug(merged, "score_codebert", "codebert_norm")
    hybrid_scores = hybrid_scores_at_alpha(merged, ALPHA)

    gold = test_bugs[["project", "key", "gold_files"]]

    per_bug = build_per_bug_table(test_bugs, tfidf_scores, codebert_scores, hybrid_scores)
    bug_meta = per_bug[["project", "key", "group", "n_gold"]]

    summary = build_summary_table(tfidf_scores, codebert_scores, hybrid_scores, gold, bug_meta)
    diagnostic = build_multigold_diagnostic(
        tfidf_scores, codebert_scores, hybrid_scores, gold, bug_meta
    )
    examples = build_examples_table(per_bug, args.n_examples)

    out_dir = ensure_dir(cfg.paths.tables)
    per_bug.to_csv(out_dir / PER_BUG_NAME, index=False)
    summary.to_csv(out_dir / SUMMARY_NAME, index=False)
    examples.to_csv(out_dir / EXAMPLES_NAME, index=False)

    print("\n=== Cassandra30 TEST gold-count group counts ===")
    print(per_bug["group"].value_counts().to_string())

    print(f"\n=== Cassandra30 TEST metrics by gold-count group (alpha={ALPHA} Hybrid) ===")
    print(summary.round(4).to_string(index=False))

    print("\n=== multi-gold diagnostic (Top-10 / Top-20 coverage) ===")
    print(diagnostic.round(4).to_string(index=False))

    print(f"\nwrote {out_dir / PER_BUG_NAME}")
    print(f"wrote {out_dir / SUMMARY_NAME}")
    print(f"wrote {out_dir / EXAMPLES_NAME}")


if __name__ == "__main__":
    main()
