import csv
import json

from bug2code.utils.experiment_log import (
    LOG_COLUMNS,
    ExperimentRecord,
    log_experiment,
    make_run_id,
)


def _record(**kwargs):
    defaults = dict(
        run_id="test-run",
        task="localization",
        model="tfidf_cosine",
        dataset="spark",
        split="temporal",
        seed=42,
    )
    return ExperimentRecord(**{**defaults, **kwargs})


def test_make_run_id_is_readable_and_includes_parts():
    run_id = make_run_id("localization", "codebert", "frozen")
    assert run_id.endswith("_localization_codebert_frozen")


def test_log_experiment_writes_json_and_csv(tmp_path):
    record = _record(val_metrics={"mrr": 0.5}, notes="smoke")
    run_dir = log_experiment(record, tmp_path, artefacts={"per_class": {"SQL": 0.6}})

    payload = json.loads((run_dir / "run.json").read_text())
    assert payload["val_metrics"] == {"mrr": 0.5}
    assert payload["artefacts"]["per_class"]["SQL"] == 0.6

    rows = list(csv.DictReader((tmp_path / "experiment_log.csv").open()))
    assert len(rows) == 1
    assert list(rows[0]) == LOG_COLUMNS
    assert rows[0]["notes"] == "smoke"


def test_log_experiment_appends_without_duplicate_header(tmp_path):
    log_experiment(_record(run_id="a"), tmp_path)
    log_experiment(_record(run_id="b"), tmp_path)
    rows = list(csv.DictReader((tmp_path / "experiment_log.csv").open()))
    assert [r["run_id"] for r in rows] == ["a", "b"]


def test_test_metrics_default_to_empty():
    assert _record().test_metrics == {}
