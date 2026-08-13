r"""Lexical error analysis: identifier-rich vs. plain-prose bug reports (Cassandra30 TEST, FINAL).

Splits every Cassandra30 TEST bug into two groups using ONLY its title +
description text (never model rankings/scores/gold ranks), and compares
TF-IDF, Fine-tuned CodeBERT epoch-3, and Hybrid alpha=0.5 ranking quality
between the groups. CPU-only: reads the three already-saved TEST candidate-
score tables and never loads a model or runs inference.

- ``save_tfidf_test_scores.py`` -> ``tfidf_test_candidate_scores.parquet``.
- ``save_finetuned_test_scores.py`` -> ``finetuned_epoch{N}_test_candidate_scores.parquet``.
- Hybrid is reconstructed exactly as ``hybrid_test_experiment.py`` does: join
  the two tables above on (project, key, path), per-bug min-max normalize
  each score column (``hybrid_experiment.normalize_per_bug``, unchanged),
  combine at the Validation-selected, fixed alpha=0.5
  (``hybrid_test_experiment.hybrid_scores_at_alpha``, unchanged) - no re-sweep,
  no re-fit, no re-encoding.

Ranking metrics reuse ``candidate_scores.metrics_from_scores`` /
``metrics.py`` unchanged, so hit@k/recall@k/MRR/MAP and gold-file semantics
are identical to the final Test evaluation scripts. This script only ever
reads TEST tables - Validation is never read here.

### Classification rule (deterministic, transparent, text-only)

Six regex-based lexical signals are computed from ``title + " " + description``
(the same text the models see, via ``tfidf.bug_text``):

- ``has_stacktrace``: a Java-style stack frame (``at pkg.Class.method(...)``)
  or a ``Caused by:`` line.
- ``has_source_filename``: a token ending in a source-file extension
  (``.java``, ``.scala``, ``.py``, ...).
- ``has_method_pattern``: an identifier immediately followed by ``(...)``
  with no space before the ``(`` - excludes ordinary English parentheticals
  like "issue (see below)", which always have a space there.
- ``has_camelcase_identifier``: a PascalCase identifier with 2+ humps
  (``ConsistencyLevel``) or a lowerCamelCase identifier (``validateForWrite``)
  - single-hump capitalized words (``Java``, ``The``) never match.
- ``has_qualified_identifier``: a dotted identifier with 3+ segments
  (``org.apache.cassandra``) - 2-segment abbreviations like "e.g." can't
  match (both require a letter/underscore start with no space around dots).
- ``has_exception_error``: an identifier ending in ``Exception``/``Error``
  with a non-empty prefix - bare "Error"/"Exception" as ordinary English
  words can never match, since the pattern requires characters before the
  suffix that don't fit inside the literal suffix itself.

``identifier_count`` is the total number of regex matches across all six
patterns (occurrences, not deduplicated across categories).

**Final rule**: a bug is ``identifier_rich`` if any of the five "strong"
signals (``has_stacktrace``, ``has_source_filename``, ``has_method_pattern``,
``has_qualified_identifier``, ``has_exception_error``) is true, OR if 2 or
more CamelCase-identifier matches are found (a single incidental match, e.g.
a product name like "GitHub", is not enough on its own). Otherwise
``plain_prose``.

Usage:
    python -m bug2code.localization.lexical_error_analysis --config configs/cassandra30.yaml
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

from bug2code.config import load_config
from bug2code.data.split import OUTPUT_NAME as DATASET_NAME
from bug2code.localization.candidate_scores import metrics_from_scores, ranked_paths
from bug2code.localization.hybrid_experiment import load_and_join, normalize_per_bug
from bug2code.localization.hybrid_test_experiment import ALPHA, hybrid_scores_at_alpha
from bug2code.localization.metrics import best_gold_rank, hit_at_k, reciprocal_rank
from bug2code.localization.save_finetuned_test_scores import (
    OUTPUT_TEMPLATE as CODEBERT_SCORES_TEMPLATE,
)
from bug2code.localization.save_tfidf_test_scores import OUTPUT_NAME as TFIDF_SCORES_NAME
from bug2code.localization.tfidf import bug_text
from bug2code.logging_utils import get_logger, setup_logging
from bug2code.paths import ensure_dir

logger = get_logger(__name__)

SUMMARY_NAME = "lexical_error_analysis_summary.csv"
PER_BUG_NAME = "lexical_error_analysis_per_bug.csv"
EXAMPLES_NAME = "lexical_error_analysis_examples.csv"

TOP_N_PREDICTIONS = 5
METHODS = ("tfidf", "finetuned", "hybrid")

# --- lexical feature regexes ------------------------------------------------

_SOURCE_FILENAME_RE = re.compile(
    r"\b[\w][\w\-./]*\.(?:java|scala|py|cpp|cc|c|h|hpp|js|ts|go|rb|cs)\b"
)
# No \s* before "(": excludes "issue (see below)" (space before paren), keeps
# "foo(bar, baz)" and "foo()" (identifier directly attached to the paren).
_METHOD_PATTERN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\([^)\n]*\)")
# PascalCase with 2+ humps ("ConsistencyLevel") or lowerCamelCase ("validateForWrite").
_CAMELCASE_RE = re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}\b|\b[a-z]+[A-Z][A-Za-z0-9]*\b")
# 3+ dotted segments, each starting with a letter/underscore, no surrounding
# whitespace - excludes "e.g." (2 segments) and version numbers ("3.11.2").
_QUALIFIED_IDENTIFIER_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){2,}\b"
)
_STACKTRACE_RE = re.compile(r"\bat\s+[\w$]+(?:\.[\w$]+)+\([^)]*\)|\bCaused by:")
# Requires a prefix before Error/Exception, so plain English words never match.
_EXCEPTION_ERROR_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:Exception|Error)\b")


def extract_features(text: str) -> dict:
    """The six lexical signals (booleans + identifier_count) for one bug's text."""
    stacktrace = _STACKTRACE_RE.findall(text)
    filenames = _SOURCE_FILENAME_RE.findall(text)
    methods = _METHOD_PATTERN_RE.findall(text)
    camelcase = _CAMELCASE_RE.findall(text)
    qualified = _QUALIFIED_IDENTIFIER_RE.findall(text)
    exc_err = _EXCEPTION_ERROR_RE.findall(text)

    return {
        "has_stacktrace": len(stacktrace) > 0,
        "has_source_filename": len(filenames) > 0,
        "has_method_pattern": len(methods) > 0,
        "has_camelcase_identifier": len(camelcase) > 0,
        "has_qualified_identifier": len(qualified) > 0,
        "has_exception_error": len(exc_err) > 0,
        "identifier_count": sum(
            len(m) for m in (stacktrace, filenames, methods, camelcase, qualified, exc_err)
        ),
        "camelcase_count": len(camelcase),
    }


def classify(features: dict) -> str:
    """``identifier_rich`` if any strong signal fires, or 2+ CamelCase identifiers."""
    strong = (
        features["has_stacktrace"]
        or features["has_source_filename"]
        or features["has_method_pattern"]
        or features["has_qualified_identifier"]
        or features["has_exception_error"]
    )
    weak = features["camelcase_count"] >= 2
    return "identifier_rich" if (strong or weak) else "plain_prose"


def bug_prediction_summary(
    bug_scores: pd.DataFrame, gold: list[str], top_n: int = TOP_N_PREDICTIONS
) -> dict:
    """Rank/RR/hit@10/top-N for one bug's raw score rows under one method."""
    ranked = ranked_paths(bug_scores) if len(bug_scores) else []
    return {
        "rank": best_gold_rank(ranked, gold),
        "rr": reciprocal_rank(ranked, gold),
        "hit_at_10": hit_at_k(ranked, gold, 10),
        "top5": ";".join(ranked[:top_n]),
    }


def build_per_bug_table(
    test_bugs: pd.DataFrame,
    tfidf_scores: pd.DataFrame,
    codebert_scores: pd.DataFrame,
    hybrid_scores: pd.DataFrame,
) -> pd.DataFrame:
    """One row per Cassandra30 TEST bug: group, lexical features, per-method ranks."""
    scores_by_method = {
        "tfidf": {k: g for k, g in tfidf_scores.groupby(["project", "key"])},
        "finetuned": {k: g for k, g in codebert_scores.groupby(["project", "key"])},
        "hybrid": {k: g for k, g in hybrid_scores.groupby(["project", "key"])},
    }

    rows = []
    for bug in test_bugs.itertuples():
        key = (bug.project, bug.key)
        gold = list(bug.gold_files)
        features = extract_features(bug_text(bug))
        group = classify(features)

        row = {
            "project": bug.project,
            "key": bug.key,
            "title": bug.title,
            "group": group,
            "has_stacktrace": features["has_stacktrace"],
            "has_source_filename": features["has_source_filename"],
            "has_method_pattern": features["has_method_pattern"],
            "has_camelcase_identifier": features["has_camelcase_identifier"],
            "has_qualified_identifier": features["has_qualified_identifier"],
            "has_exception_error": features["has_exception_error"],
            "identifier_count": features["identifier_count"],
            "n_gold": len(gold),
            "gold_files": ";".join(gold),
        }

        for method in METHODS:
            bug_scores = scores_by_method[method].get(key, pd.DataFrame())
            pred = bug_prediction_summary(bug_scores, gold)
            row[f"{method}_rank"] = pred["rank"]
            row[f"{method}_rr"] = pred["rr"]
            row[f"{method}_hit_at_10"] = pred["hit_at_10"]
            row[f"{method}_top5"] = pred["top5"]

        rows.append(row)
    return pd.DataFrame(rows)


def build_summary_table(
    tfidf_scores: pd.DataFrame,
    codebert_scores: pd.DataFrame,
    hybrid_scores: pd.DataFrame,
    gold: pd.DataFrame,
    bug_groups: pd.DataFrame,
) -> pd.DataFrame:
    """Per (method, group) mean metrics, plus Fine-tuned/Hybrid vs. TF-IDF deltas.

    Delta rows share the same columns but only populate ``mrr``/``hit_at_10``
    (the two deltas asked for); the rest are left ``NaN`` and ``n_bugs`` is
    ``NA`` - they compare already-computed means, not a distinct bug set.
    """
    scores_by_method = {
        "tfidf": tfidf_scores,
        "finetuned": codebert_scores,
        "hybrid": hybrid_scores,
    }

    rows = []
    for method, scores in scores_by_method.items():
        per_bug = metrics_from_scores(scores, gold).merge(bug_groups, on=["project", "key"])
        exclude = ("project", "key", "n_candidates", "group")
        metric_cols = [c for c in per_bug.columns if c not in exclude]
        for group, g in per_bug.groupby("group"):
            row = {"method": method, "group": group, "n_bugs": len(g)}
            row |= g[metric_cols].mean().to_dict()
            rows.append(row)
    summary = pd.DataFrame(rows)

    mrr_p = summary.pivot(index="group", columns="method", values="mrr")
    hit10_p = summary.pivot(index="group", columns="method", values="hit_at_10")
    delta_rows = []
    for group in mrr_p.index:
        delta_rows.append(
            {
                "method": "delta_finetuned_minus_tfidf",
                "group": group,
                "n_bugs": pd.NA,
                "mrr": mrr_p.loc[group, "finetuned"] - mrr_p.loc[group, "tfidf"],
                "hit_at_10": hit10_p.loc[group, "finetuned"] - hit10_p.loc[group, "tfidf"],
            }
        )
        delta_rows.append(
            {
                "method": "delta_hybrid_minus_tfidf",
                "group": group,
                "n_bugs": pd.NA,
                "mrr": mrr_p.loc[group, "hybrid"] - mrr_p.loc[group, "tfidf"],
                "hit_at_10": hit10_p.loc[group, "hybrid"] - hit10_p.loc[group, "tfidf"],
            }
        )
    return pd.concat([summary, pd.DataFrame(delta_rows)], ignore_index=True)


def build_examples_table(per_bug: pd.DataFrame, n: int) -> pd.DataFrame:
    """Top-``n`` per category by RR delta magnitude - not manually cherry-picked."""
    df = per_bug.copy()
    df["tfidf_minus_finetuned_rr"] = df["tfidf_rr"] - df["finetuned_rr"]
    df["finetuned_minus_tfidf_rr"] = df["finetuned_rr"] - df["tfidf_rr"]
    df["hybrid_minus_tfidf_rr"] = df["hybrid_rr"] - df["tfidf_rr"]

    categories = [
        ("tfidf_beats_finetuned", "tfidf_minus_finetuned_rr"),
        ("finetuned_beats_tfidf", "finetuned_minus_tfidf_rr"),
        ("hybrid_beats_tfidf", "hybrid_minus_tfidf_rr"),
    ]
    frames = [df.nlargest(n, col).assign(example_category=cat) for cat, col in categories]

    cols = [
        "example_category",
        "project",
        "key",
        "title",
        "group",
        "gold_files",
        "tfidf_rank",
        "finetuned_rank",
        "hybrid_rank",
        "tfidf_rr",
        "finetuned_rr",
        "hybrid_rr",
        "tfidf_top5",
        "finetuned_top5",
        "hybrid_top5",
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
    summary = build_summary_table(
        tfidf_scores, codebert_scores, hybrid_scores, gold, per_bug[["project", "key", "group"]]
    )
    examples = build_examples_table(per_bug, args.n_examples)

    out_dir = ensure_dir(cfg.paths.tables)
    per_bug.to_csv(out_dir / PER_BUG_NAME, index=False)
    summary.to_csv(out_dir / SUMMARY_NAME, index=False)
    examples.to_csv(out_dir / EXAMPLES_NAME, index=False)

    print("\n=== Cassandra30 TEST lexical group counts ===")
    print(per_bug["group"].value_counts().to_string())

    print(f"\n=== Cassandra30 TEST metrics by lexical group (alpha={ALPHA} Hybrid) ===")
    print(summary.round(4).to_string(index=False))

    print(f"\nwrote {out_dir / PER_BUG_NAME}")
    print(f"wrote {out_dir / SUMMARY_NAME}")
    print(f"wrote {out_dir / EXAMPLES_NAME}")


if __name__ == "__main__":
    main()
