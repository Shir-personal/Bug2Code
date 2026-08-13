import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from bug2code.localization.train_codebert import (
    build_training_pairs,
    epoch_examples,
    find_resume_epoch,
    mnr_loss,
    run_benchmark,
    run_epoch,
    wrap_chunks,
)


class _FakeTokenizer:
    """Minimal stand-in for an HF tokenizer: no network, no real vocabulary."""

    cls_token_id = 0
    sep_token_id = 1
    pad_token_id = 2

    def __call__(self, texts, padding=None, truncation=None, max_length=None):
        ids, masks = [], []
        for text in texts:
            tok_ids = [3 + (ord(c) % 5) for c in text][:max_length]
            pad_len = max_length - len(tok_ids)
            ids.append(tok_ids + [self.pad_token_id] * pad_len)
            masks.append([1] * len(tok_ids) + [0] * pad_len)
        return {"input_ids": ids, "attention_mask": masks}


class _FakeDualEncoder(nn.Module):
    """Tiny learnable stand-in for DualEncoder: same encode_query/encode_code
    interface, no pretrained weights, so tests stay fast and fully offline."""

    def __init__(self, vocab_size: int = 16, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)

    def _pool(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embed(ids)
        mask_f = mask.unsqueeze(-1).float()
        pooled = (emb * mask_f).sum(1) / mask_f.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, dim=1)

    def encode_query(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._pool(ids, mask)

    def encode_code(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._pool(ids, mask)


def _tiny_setup(n_pairs: int = 6):
    """A small fake pairs/chunks/cfg fixture, enough to drive run_epoch for real."""
    pairs = pd.DataFrame(
        {
            "file_version_id": [i % 3 for i in range(n_pairs)],
            "query_text": [f"bug report {i}" for i in range(n_pairs)],
        }
    )
    chunks_by_id = {0: [[3, 4, 5]], 1: [[4, 5, 6], [5, 6, 7]], 2: [[6, 7]]}
    cfg = types.SimpleNamespace(
        seed=0,
        localization={
            "code_max_length": 8,
            "query_max_length": 8,
            "encoder": {"batch_size": 2, "scale": 20.0},
        },
    )
    model = _FakeDualEncoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer, _FakeTokenizer(), pairs, chunks_by_id, cfg


def _hf_offline_env_after_import(preset: dict[str, str]) -> dict[str, str | None]:
    """Import the module in a clean subprocess and report the two HF env vars.

    A subprocess is required: importing in-process would leak whatever the
    surrounding test session already set into ``os.environ``, and module
    imports are cached so a second import wouldn't re-run the module body.
    """
    script = (
        "import os, json\n"
        "import bug2code.localization.train_codebert\n"
        "print(json.dumps({"
        "'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE'), "
        "'TRANSFORMERS_OFFLINE': os.environ.get('TRANSFORMERS_OFFLINE')"
        "}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=preset,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_importing_train_codebert_does_not_force_offline_mode():
    env = dict(os.environ)
    env.pop("HF_HUB_OFFLINE", None)
    env.pop("TRANSFORMERS_OFFLINE", None)
    result = _hf_offline_env_after_import(env)
    assert result == {"HF_HUB_OFFLINE": None, "TRANSFORMERS_OFFLINE": None}


def test_importing_train_codebert_preserves_explicit_offline_mode():
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    result = _hf_offline_env_after_import(env)
    assert result == {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def test_mnr_loss_lower_for_aligned_pairs_than_random():
    aligned = torch.eye(4)
    torch.manual_seed(0)
    random_vecs = torch.nn.functional.normalize(torch.randn(4, 4), dim=1)
    aligned_loss = mnr_loss(aligned, aligned, scale=20.0).item()
    random_loss = mnr_loss(aligned, random_vecs, scale=20.0).item()
    assert aligned_loss < random_loss


def test_find_resume_epoch_picks_highest(tmp_path: Path):
    for name in ("epoch_1.pt", "epoch_3.pt", "epoch_2.pt"):
        (tmp_path / name).touch()
    assert find_resume_epoch(tmp_path) == 3


def test_find_resume_epoch_empty_dir_is_zero(tmp_path: Path):
    assert find_resume_epoch(tmp_path) == 0


def test_wrap_chunks_adds_cls_sep_and_pads():
    ids, masks = wrap_chunks([[5, 6, 7]], max_len=8, cls_id=0, sep_id=2, pad_id=1)
    assert ids == [[0, 5, 6, 7, 2, 1, 1, 1]]
    assert masks == [[1, 1, 1, 1, 1, 0, 0, 0]]


def test_epoch_examples_deterministic_for_same_seed():
    pairs = pd.DataFrame({"file_version_id": [1, 2, 3], "query_text": ["a", "b", "c"]})
    chunks_by_id = {1: [[1], [2], [3]], 2: [[4], [5]], 3: [[6]]}
    p1, c1 = epoch_examples(pairs, chunks_by_id, seed=42)
    p2, c2 = epoch_examples(pairs, chunks_by_id, seed=42)
    assert p1["file_version_id"].tolist() == p2["file_version_id"].tolist()
    assert c1 == c2


def test_epoch_examples_differs_across_epoch_seeds():
    pairs = pd.DataFrame({"file_version_id": [1] * 5, "query_text": list("abcde")})
    chunks_by_id = {1: [[i] for i in range(20)]}
    _, c1 = epoch_examples(pairs, chunks_by_id, seed=1)
    _, c2 = epoch_examples(pairs, chunks_by_id, seed=2)
    assert c1 != c2


def test_build_training_pairs_one_row_per_bug_gold_file_pair():
    train_bugs = pd.DataFrame(
        {
            "project": ["p", "p"],
            "key": ["BUG-1", "BUG-2"],
            "title": ["t1", "t2"],
            "description": ["d1", None],
            "gold_files": [["a.java", "b.java"], ["a.java"]],
        }
    )
    versions = pd.DataFrame({"id": [10, 11], "project": ["p", "p"], "path": ["a.java", "b.java"]})
    index = pd.DataFrame(
        {
            "project": ["p", "p", "p"],
            "key": ["BUG-1", "BUG-1", "BUG-2"],
            "file_version_id": [10, 11, 10],
        }
    )

    pairs = build_training_pairs(train_bugs, versions, index)

    assert len(pairs) == 3  # D2: not deduped, one row per (bug, gold-file)
    assert set(pairs["query_text"]) == {"t1 d1", "t2 "}
    assert sorted(pairs["file_version_id"].tolist()) == [10, 10, 11]


def test_build_training_pairs_raises_on_missing_gold_file():
    train_bugs = pd.DataFrame(
        {
            "project": ["p"],
            "key": ["BUG-1"],
            "title": ["t1"],
            "description": ["d1"],
            "gold_files": [["missing.java"]],
        }
    )
    versions = pd.DataFrame({"id": [], "project": [], "path": []})
    index = pd.DataFrame({"project": [], "key": [], "file_version_id": []})

    with pytest.raises(ValueError, match="data integrity"):
        build_training_pairs(train_bugs, versions, index)


def test_run_epoch_default_runs_every_batch():
    model, optimizer, tokenizer, pairs, chunks_by_id, cfg = _tiny_setup(n_pairs=6)
    calls = []
    run_epoch(
        model, optimizer, tokenizer, pairs, chunks_by_id, cfg, epoch=1, device="cpu",
        on_step=lambda step, elapsed, loss: calls.append(step),
    )
    # batch_size=2, 6 pairs -> 3 full-epoch batches when max_steps is not given
    assert calls == [0, 1, 2]


def test_run_epoch_max_steps_stops_early_without_changing_earlier_steps():
    model, optimizer, tokenizer, pairs, chunks_by_id, cfg = _tiny_setup(n_pairs=6)
    calls = []
    mean_loss = run_epoch(
        model, optimizer, tokenizer, pairs, chunks_by_id, cfg, epoch=1, device="cpu",
        max_steps=2, on_step=lambda step, elapsed, loss: calls.append((step, loss)),
    )
    assert [s for s, _ in calls] == [0, 1]
    assert mean_loss == pytest.approx(sum(loss for _, loss in calls) / 2)


def test_run_benchmark_reports_expected_fields_on_cpu():
    model, optimizer, tokenizer, pairs, chunks_by_id, cfg = _tiny_setup(n_pairs=6)
    report = run_benchmark(model, optimizer, tokenizer, pairs, chunks_by_id, cfg, "cpu", n_steps=2)

    assert report["device"] == "cpu"
    assert report["batch_size"] == 2
    assert report["steps_completed"] == 2
    assert report["warmup_steps_excluded"] == 0  # fewer steps than WARMUP_STEPS
    assert report["avg_s_per_step"] >= 0
    assert report["total_benchmark_s"] >= 0
    assert report["cuda_memory_allocated_mb"] is None  # no CUDA device in this test
    assert report["cuda_memory_reserved_mb"] is None
    assert report["cuda_peak_memory_allocated_mb"] is None
    assert isinstance(report["loss_start"], float)
    assert isinstance(report["loss_end"], float)
    assert report["n_batches_per_epoch"] == 3  # ceil(6 pairs / batch_size 2)
    assert report["estimated_epoch_s"] == pytest.approx(report["avg_s_per_step"] * 3)


def test_run_benchmark_excludes_warmup_steps_once_past_threshold():
    model, optimizer, tokenizer, pairs, chunks_by_id, cfg = _tiny_setup(n_pairs=20)
    report = run_benchmark(model, optimizer, tokenizer, pairs, chunks_by_id, cfg, "cpu", n_steps=7)
    assert report["steps_completed"] == 7
    assert report["warmup_steps_excluded"] == 5
