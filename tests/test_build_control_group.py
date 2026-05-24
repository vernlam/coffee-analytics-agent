"""Tests for src/tools/build_control_group.py."""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.tools.build_control_group import _get_weekly_revenue, _match, _normalize


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_daily_df():
    """Three stores, two weeks of daily data (14 days each)."""
    rows = []
    for merchant_id in ["M001", "M002", "M003"]:
        for day_offset in range(14):
            txn_date = date(2024, 1, 1 + day_offset)
            # M001: 100/day, M002: 200/day, M003: 150/day
            amount = {"M001": 100.0, "M002": 200.0, "M003": 150.0}[merchant_id]
            rows.append({
                "merchant_id": merchant_id,
                "txn_date":    txn_date,
                "amount":      amount,
            })
    return pd.DataFrame(rows)


def make_spark(df: pd.DataFrame) -> MagicMock:
    mock = MagicMock()
    mock.table.return_value.filter.return_value.withColumn.return_value \
        .groupBy.return_value.agg.return_value.orderBy.return_value \
        .toPandas.return_value = df
    return mock


# ---------------------------------------------------------------------------
# _get_weekly_revenue
# ---------------------------------------------------------------------------

skip_no_pyspark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyspark") is None,
    reason="pyspark not installed locally",
)


@skip_no_pyspark
def test_weekly_revenue_sums_correctly():
    daily_df = make_daily_df()
    mock_spark = make_spark(
        daily_df.rename(columns={"txn_date": "week_start", "amount": "weekly_revenue"})
    )
    result = _get_weekly_revenue(mock_spark, date(2024, 1, 1), date(2024, 1, 14))
    assert "merchant_id" in result.columns
    assert "weekly_revenue" in result.columns


@skip_no_pyspark
def test_weekly_revenue_has_correct_stores():
    daily_df = make_daily_df()
    mock_spark = make_spark(
        daily_df.rename(columns={"txn_date": "week_start", "amount": "weekly_revenue"})
    )
    result = _get_weekly_revenue(mock_spark, date(2024, 1, 1), date(2024, 1, 14))
    assert set(result["merchant_id"].unique()) == {"M001", "M002", "M003"}


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

def make_weekly_df():
    """Two weeks of weekly data for three stores."""
    return pd.DataFrame([
        {"merchant_id": "M001", "week_start": date(2024, 1, 1),  "weekly_revenue": 700.0},
        {"merchant_id": "M001", "week_start": date(2024, 1, 8),  "weekly_revenue": 700.0},
        {"merchant_id": "M002", "week_start": date(2024, 1, 1),  "weekly_revenue": 1400.0},
        {"merchant_id": "M002", "week_start": date(2024, 1, 8),  "weekly_revenue": 1400.0},
        {"merchant_id": "M003", "week_start": date(2024, 1, 1),  "weekly_revenue": 1050.0},
        {"merchant_id": "M003", "week_start": date(2024, 1, 8),  "weekly_revenue": 1050.0},
    ])


def test_normalize_mean_is_one():
    pivot = _normalize(make_weekly_df())
    for col in pivot.columns:
        assert abs(pivot[col].mean() - 1.0) < 1e-9, f"{col} mean is not 1.0"


def test_normalize_returns_pivot():
    pivot = _normalize(make_weekly_df())
    assert set(pivot.columns) == {"M001", "M002", "M003"}
    assert len(pivot) == 2  # two weeks


def test_normalize_preserves_shape():
    weekly_df = make_weekly_df()
    pivot = _normalize(weekly_df)
    assert pivot.shape == (2, 3)  # 2 weeks, 3 stores


# ---------------------------------------------------------------------------
# _match
# ---------------------------------------------------------------------------

def make_pivot():
    """
    M001 and M002 have identical series.
    M003 is very different.
    Treated = [M001]. Best match should be M002.
    """
    return pd.DataFrame({
        "M001": [1.0, 1.0, 1.0, 1.0],
        "M002": [1.0, 1.0, 1.0, 1.0],  # identical to M001
        "M003": [2.0, 0.5, 2.0, 0.5],  # very different
    })


def test_match_best_match_is_most_similar():
    pivot = make_pivot()
    result = _match(pivot, treated_ids=["M001"], n_matches=2)
    top = result.sort_values("similarity_score", ascending=False).iloc[0]
    assert top["control_merchant_id"] == "M002"


def test_match_treated_excluded_from_controls():
    pivot = make_pivot()
    result = _match(pivot, treated_ids=["M001"], n_matches=2)
    assert "M001" not in result["control_merchant_id"].values


def test_match_similarity_score_between_0_and_1():
    pivot = make_pivot()
    result = _match(pivot, treated_ids=["M001"], n_matches=2)
    assert (result["similarity_score"] >= 0).all()
    assert (result["similarity_score"] <= 1).all()


def test_match_identical_series_has_score_1():
    pivot = make_pivot()
    result = _match(pivot, treated_ids=["M001"], n_matches=2)
    perfect = result[result["control_merchant_id"] == "M002"].iloc[0]
    assert perfect["similarity_score"] == 1.0


def test_match_returns_correct_columns():
    pivot = make_pivot()
    result = _match(pivot, treated_ids=["M001"], n_matches=2)
    assert set(result.columns) == {"treated_merchant_id", "control_merchant_id", "similarity_score"}


def test_match_respects_n_matches():
    pivot = make_pivot()
    result = _match(pivot, treated_ids=["M001"], n_matches=1)
    assert len(result) == 1
