"""Describe the collected dataset: coverage, components, gold files, text lengths.

Everything here is descriptive. It answers the questions that decide whether the
experimental design is feasible before any model is trained: how many bugs
survive linking, how many files a fix touches, how often a gold file was created
by the fix itself, and — for the RQ4 component experiment — how much of the
search space a component filter removes and how much correct answer it removes
with it.

Usage:
    python -m bug2code.data.dataset_report [--config configs/exp.yaml]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bug2code.config import Config, load_config  # noqa: E402
from bug2code.data.collect_issues import OUTPUT_NAME as ISSUES_NAME  # noqa: E402
from bug2code.data.link_commits import FIX_COMMITS_NAME  # noqa: E402
from bug2code.logging_utils import get_logger, setup_logging  # noqa: E402
from bug2code.paths import ensure_dir  # noqa: E402

logger = get_logger(__name__)

FIGSIZE = (11, 7)


def load_joined(cfg: Config) -> pd.DataFrame:
    """Join issues with their selected fixing commit (left join keeps unlinked bugs)."""
    issues = pd.read_parquet(cfg.paths.data_raw / ISSUES_NAME)
    fixes = pd.read_parquet(cfg.paths.data_interim / FIX_COMMITS_NAME)

    fix_cols = [
        "key",
        "sha",
        "parent_sha",
        "commit_date",
        "source_files",
        "source_statuses",
        "n_files_total",
        "n_candidates",
    ]
    df = issues.merge(fixes[fix_cols], on="key", how="left")

    df["linked"] = df["sha"].notna()
    df["n_components"] = df["components"].apply(len)
    df["created_dt"] = pd.to_datetime(df["created"], utc=True, format="mixed")
    df["title_words"] = df["summary"].str.split().apply(len)
    df["desc_words"] = df["description"].str.split().apply(len)

    # Unlinked issues have no commit, so list columns come back as NaN.
    empty = pd.Series([[]] * len(df), index=df.index)
    df["source_files"] = df["source_files"].where(df["linked"], empty)
    df["source_statuses"] = df["source_statuses"].where(df["linked"], empty)

    df["n_gold"] = df["source_files"].apply(len)
    df["n_gold_new"] = df["source_statuses"].apply(
        lambda statuses: sum(1 for s in statuses if s.startswith("A"))
    )
    df["n_gold_existing"] = df["n_gold"] - df["n_gold_new"]
    return df


def summarise_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Per-project funnel from collected issues to bugs usable for localization.

    A bug is usable when it has a fixing commit and at least one gold file that
    already existed before the fix, since a file created by the fix cannot be
    ranked against a pre-fix snapshot.
    """
    rows = []
    for project, group in df.groupby("project"):
        linked = group[group["linked"]]
        usable = linked[linked["n_gold_existing"] > 0]

        rows.append(
            {
                "project": project,
                "issues": len(group),
                "with_component": int((group["n_components"] > 0).sum()),
                "single_component": int((group["n_components"] == 1).sum()),
                "components_total": int(group["components"].explode().nunique()),
                "linked": len(linked),
                "link_rate": round(len(linked) / len(group), 4),
                "usable": len(usable),
                "lost_all_gold_new": int((linked["n_gold_existing"] == 0).sum()),
                "usable_with_component": int((usable["n_components"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def component_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Issue count per (project, component) over single-component issues."""
    single = df[df["n_components"] == 1].copy()
    single["component"] = single["components"].str[0]
    out = (
        single.groupby(["project", "component"])
        .agg(issues=("key", "size"), linked=("linked", "sum"))
        .reset_index()
    )
    totals = out.groupby("project")["issues"].transform("sum")
    out["share"] = (out["issues"] / totals).round(4)
    return out.sort_values(["project", "issues"], ascending=[True, False])


def gold_file_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of gold files per linked bug, and how many are new files."""
    rows = []
    for project, group in df[df["linked"]].groupby("project"):
        n = group["n_gold"]
        rows.append(
            {
                "project": project,
                "linked_bugs": len(group),
                "gold_mean": round(n.mean(), 2),
                "gold_median": int(n.median()),
                "gold_p90": int(n.quantile(0.9)),
                "gold_max": int(n.max()),
                "pct_single_file": round((n == 1).mean(), 4),
                "pct_over_5_files": round((n > 5).mean(), 4),
                "gold_files_total": int(n.sum()),
                "gold_files_new": int(group["n_gold_new"].sum()),
                "pct_gold_files_new": round(group["n_gold_new"].sum() / max(n.sum(), 1), 4),
                "bugs_with_any_new": int((group["n_gold_new"] > 0).sum()),
                "bugs_all_gold_new": int((group["n_gold_existing"] == 0).sum()),
                "mean_candidate_commits": round(group["n_candidates"].mean(), 2),
            }
        )
    return pd.DataFrame(rows)


def extension_counts(df: pd.DataFrame) -> pd.DataFrame:
    """How the gold files split across source languages."""
    rows = []
    for project, group in df[df["linked"]].groupby("project"):
        counter: Counter[str] = Counter()
        for files in group["source_files"]:
            counter.update(Path(f).suffix for f in files)
        total = sum(counter.values())
        rows += [
            {"project": project, "extension": ext, "files": n, "share": round(n / total, 4)}
            for ext, n in counter.most_common()
        ]
    return pd.DataFrame(rows)


def text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Word-length statistics of titles and descriptions."""
    return (
        df.groupby("project")
        .agg(
            title_words_mean=("title_words", "mean"),
            title_words_p95=("title_words", lambda s: s.quantile(0.95)),
            desc_words_mean=("desc_words", "mean"),
            desc_words_median=("desc_words", "median"),
            desc_words_p95=("desc_words", lambda s: s.quantile(0.95)),
            empty_description=("desc_words", lambda s: (s == 0).sum()),
        )
        .round(1)
        .reset_index()
    )


def component_candidate_coverage(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Component candidate coverage and search-space reduction.

    Condition B ranks only files that some *training* bug of the same component
    touched. A gold file outside that set can never be retrieved, however good
    the ranker is, so coverage is the ceiling on Condition B — measurable before
    any model exists — and the search-space reduction is the other half of the
    trade-off RQ4 is about.

    The universe here is "files ever touched by a bug", not the full repository
    snapshot, which does not exist until phase 1 is finished. Coverage is
    unaffected by that choice; the reduction is conservative.
    """
    train_frac = cfg.split.train_frac
    test_start = train_frac + cfg.split.val_frac

    single = df[(df["n_components"] == 1) & df["linked"]].copy()
    single["component"] = single["components"].str[0]
    single["gold"] = [
        [f for f, s in zip(files, statuses, strict=True) if not s.startswith("A")]
        for files, statuses in zip(single["source_files"], single["source_statuses"], strict=True)
    ]
    single = single[single["gold"].apply(len) > 0]

    rows = []
    for project, group in single.groupby("project"):
        group = group.sort_values("created_dt")
        n = len(group)
        # round, not int: float rounding would truncate one bug too early
        train = group.iloc[: round(train_frac * n)]
        test = group.iloc[round(test_start * n) :]
        if not len(test):
            continue

        files_of: dict[str, set[str]] = {
            component: set().union(*sub["gold"]) for component, sub in train.groupby("component")
        }
        universe = set().union(*group["gold"])

        covered = [
            bool(set(bug.gold) & files_of.get(bug.component, set())) for bug in test.itertuples()
        ]
        sizes = [len(files_of.get(bug.component, set())) for bug in test.itertuples()]
        # A component unseen in training means an empty candidate set, not just a bad rank
        unseen = sum(1 for bug in test.itertuples() if bug.component not in files_of)

        rows.append(
            {
                "project": project,
                "test_bugs": len(test),
                "universe_files": len(universe),
                "mean_candidates": round(sum(sizes) / len(sizes), 1),
                "median_candidates": int(pd.Series(sizes).median()),
                "search_space_kept": round(sum(sizes) / len(sizes) / len(universe), 4),
                "candidate_coverage": round(sum(covered) / len(covered), 4),
                "unseen_component_rate": round(unseen / len(test), 4),
            }
        )
    return pd.DataFrame(rows)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("wrote %s", path)


def make_figures(df: pd.DataFrame, components: pd.DataFrame, figures: Path) -> None:
    """Write the four EDA figures used in the report and the deck."""
    projects = sorted(df["project"].unique())

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for project in projects:
        group = df[df["project"] == project]
        by_year = group.groupby(group["created_dt"].dt.year).size()
        ax.plot(by_year.index, by_year.values, marker="o", label=project)
    ax.set(xlabel="year created", ylabel="bug reports", title="Collected bugs per year")
    ax.legend()
    _save(fig, figures / "fig01_bugs_per_year.png")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, project in zip(axes.ravel(), projects, strict=False):
        top = components[components["project"] == project].head(15).iloc[::-1]
        ax.barh(top["component"], top["issues"])
        ax.set(title=f"{project}: top components", xlabel="issues")
        ax.tick_params(axis="y", labelsize=7)
    _save(fig, figures / "fig02_component_distribution.png")

    linked = df[df["linked"]]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(
        [linked.loc[linked["project"] == p, "n_gold"].clip(upper=20) for p in projects],
        bins=range(0, 22),
        label=projects,
    )
    ax.set(
        xlabel="gold source files per bug (clipped at 20)",
        ylabel="bugs",
        title="Gold files per linked bug",
    )
    ax.legend()
    _save(fig, figures / "fig03_files_per_bug.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(
        [df.loc[df["project"] == p, "title_words"] for p in projects],
        bins=range(0, 40, 2),
        label=projects,
    )
    axes[0].set(xlabel="title words", ylabel="bugs", title="Title length")
    axes[0].legend()
    axes[1].hist(
        [df.loc[df["project"] == p, "desc_words"].clip(upper=600) for p in projects],
        bins=30,
        label=projects,
    )
    axes[1].set(xlabel="description words (clipped at 600)", title="Description length")
    _save(fig, figures / "fig04_text_length.png")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="optional override YAML")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    cfg = load_config(args.config)

    df = load_joined(cfg)
    tables = ensure_dir(cfg.paths.tables)
    figures = ensure_dir(cfg.paths.figures)

    summary = summarise_dataset(df)
    components = component_distribution(df)
    gold = gold_file_stats(df)
    extensions = extension_counts(df)
    text = text_stats(df)
    coverage = component_candidate_coverage(df, cfg)

    summary.to_csv(tables / "05_dataset_summary.csv", index=False)
    components.to_csv(tables / "06_component_distribution.csv", index=False)
    gold.to_csv(tables / "07_gold_files.csv", index=False)
    extensions.to_csv(tables / "08_gold_extensions.csv", index=False)
    text.to_csv(tables / "09_text_stats.csv", index=False)
    coverage.to_csv(tables / "10_component_candidate_coverage.csv", index=False)

    make_figures(df, components, figures)

    for name, table in [
        ("dataset summary", summary),
        ("gold files", gold),
        ("extensions", extensions),
        ("text", text),
        ("component candidate coverage (RQ4 ceiling)", coverage),
    ]:
        print(f"\n=== {name} ===")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
