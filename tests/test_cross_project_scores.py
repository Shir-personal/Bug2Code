import types

import pandas as pd

from bug2code.localization import cross_project_scores
from bug2code.localization.cross_project_scores import (
    BENCHMARK_CHUNKS,
    BENCHMARK_SECONDS,
    CASSANDRA30_EXPERIMENTS_DIR,
    OUTPUT_TEMPLATE,
    count_chunks,
    default_checkpoint_path,
    estimate_gpu_seconds,
)
from bug2code.localization.train_codebert import CHECKPOINT_DIR
from bug2code.paths import resolve


def test_default_checkpoint_path_points_at_cassandra30_regardless_of_target_project():
    path = default_checkpoint_path(3)
    assert path == resolve(CASSANDRA30_EXPERIMENTS_DIR) / CHECKPOINT_DIR / "epoch_3.pt"


def test_default_checkpoint_path_uses_requested_epoch():
    assert default_checkpoint_path(5).name == "epoch_5.pt"


def test_estimate_gpu_seconds_matches_the_measured_benchmark_at_its_own_chunk_count():
    assert estimate_gpu_seconds(BENCHMARK_CHUNKS) == BENCHMARK_SECONDS


def test_estimate_gpu_seconds_scales_linearly():
    assert estimate_gpu_seconds(BENCHMARK_CHUNKS * 2) == BENCHMARK_SECONDS * 2


def test_output_template_names_val_and_test_distinctly():
    val_name = OUTPUT_TEMPLATE.format(epoch=3, project="hbase", split="val")
    test_name = OUTPUT_TEMPLATE.format(epoch=3, project="hbase", split="test")
    assert val_name == "cassandra_epoch3_zeroshot_hbase30_val_candidate_scores.parquet"
    assert test_name == "cassandra_epoch3_zeroshot_hbase30_test_candidate_scores.parquet"
    assert val_name != test_name


class _FakeTokenizer:
    """Tokenizes by whitespace so chunk counts are easy to predict."""

    def __call__(self, text, add_special_tokens=True, truncation=False):
        return {"input_ids": text.split()}


def test_count_chunks_never_loads_a_model_and_reports_exact_workload(monkeypatch):
    # Two bugs, two candidate files shared between them (dedup: 2 distinct files, 4 pairs).
    index = pd.DataFrame(
        {
            "project": ["hbase", "hbase", "hbase", "hbase"],
            "key": ["HBASE-1", "HBASE-1", "HBASE-2", "HBASE-2"],
            "file_version_id": [10, 20, 10, 20],
        }
    )
    versions = pd.DataFrame({"id": [10, 20], "path": ["a.java", "b.java"]})
    val_bugs = pd.DataFrame({"project": ["hbase", "hbase"], "key": ["HBASE-1", "HBASE-2"]})

    monkeypatch.setattr(
        cross_project_scores,
        "read_versions",
        lambda cfg, versions, ids: {10: "one two three", 20: "four five"},
    )
    monkeypatch.setattr(
        cross_project_scores.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer()
    )

    cfg = types.SimpleNamespace(
        localization={"code_max_length": 4, "chunk_stride": 1, "max_chunks_per_file": 12}
    )

    stats = count_chunks(cfg, "hbase", versions, index, val_bugs)

    assert stats["project"] == "hbase"
    assert stats["n_bugs"] == 2
    assert stats["n_bug_candidate_pairs"] == 4
    assert stats["n_distinct_candidate_files"] == 2
    # inner = max_length-2 = 2 tokens/chunk; 3-token file -> 2 chunks, 2-token file -> 1 chunk.
    assert stats["n_chunks"] == 3


def test_count_chunks_ignores_other_projects_in_the_index(monkeypatch):
    index = pd.DataFrame(
        {
            "project": ["hbase", "spark"],
            "key": ["HBASE-1", "SPARK-1"],
            "file_version_id": [10, 99],
        }
    )
    versions = pd.DataFrame({"id": [10, 99], "path": ["a.java", "b.scala"]})
    val_bugs = pd.DataFrame({"project": ["hbase"], "key": ["HBASE-1"]})

    monkeypatch.setattr(
        cross_project_scores, "read_versions", lambda cfg, versions, ids: {10: "one"}
    )
    monkeypatch.setattr(
        cross_project_scores.AutoTokenizer, "from_pretrained", lambda name: _FakeTokenizer()
    )

    cfg = types.SimpleNamespace(
        localization={"code_max_length": 4, "chunk_stride": 4, "max_chunks_per_file": 12}
    )

    stats = count_chunks(cfg, "hbase", versions, index, val_bugs)

    assert stats["n_bug_candidate_pairs"] == 1
    assert stats["n_distinct_candidate_files"] == 1
