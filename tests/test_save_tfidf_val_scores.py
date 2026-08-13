import pandas as pd
import pytest

from bug2code.localization.save_tfidf_val_scores import EXPECTED_METRICS, verify_against_expected


def test_verify_against_expected_passes_on_matching_metrics():
    verify_against_expected(pd.Series(EXPECTED_METRICS))


def test_verify_against_expected_raises_on_mismatch():
    bad = dict(EXPECTED_METRICS)
    bad["mrr"] = 0.0
    with pytest.raises(AssertionError, match="do not match"):
        verify_against_expected(pd.Series(bad))
