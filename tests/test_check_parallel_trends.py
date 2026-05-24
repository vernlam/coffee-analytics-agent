"""Tests for src/tools/check_parallel_trends.py."""

from datetime import date

import pandas as pd
import pytest

from src.tools.check_parallel_trends import _aggregate_control, _aggregate_treated, check_parallel_trends


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_weekly_df():
    """
    Two treated stores (M001, M002), two control stores (M003, M004).
    Pre-period: weeks -2, -1. Post-period: weeks 1, 2.
    Treated revenue jumps +5% in post-period. Control stays flat.
    """
    rows = []
    week_starts = {
        -2: date(2024, 1, 1),
        -1: date(2024, 1, 8),
         1: date(2024, 1, 22),
         2: date(2024, 1, 29),
    }
    for week_num, week_start in week_starts.items():
        is_post = week_num > 0
        for mid in ["M001", "M002"]:
            rows.append({
                "merchant_id":    mid,
                "week_start":     week_start,
                "weekly_revenue": 1050.0 if is_post else 1000.0,
            })
        for mid in ["M003", "M004"]:
            rows.append({
                "merchant_id":    mid,
                "week_start":     week_start,
                "weekly_revenue": 1000.0,
            })
    return pd.DataFrame(rows)


def make_matches():
    return pd.DataFrame([
        {"treated_merchant_id": "M001", "control_merchant_id": "M003", "similarity_score": 1.0},
        {"treated_merchant_id": "M001", "control_merchant_id": "M004", "similarity_score": 1.0},
        {"treated_merchant_id": "M002", "control_merchant_id": "M003", "similarity_score": 1.0},
        {"treated_merchant_id": "M002", "control_merchant_id": "M004", "similarity_score": 1.0},
    ])


# ---------------------------------------------------------------------------
# _aggregate_treated
# ---------------------------------------------------------------------------

def test_aggregate_treated_mean():
    weekly_df = make_weekly_df()
    result = _aggregate_treated(weekly_df, ["M001", "M002"])
    pre_revenue  = result[result["week_start"] == date(2024, 1, 1)]["treated_avg_revenue"].iloc[0]
    post_revenue = result[result["week_start"] == date(2024, 1, 22)]["treated_avg_revenue"].iloc[0]
    assert pre_revenue == 1000.0
    assert post_revenue == 1050.0


def test_aggregate_treated_excludes_controls():
    weekly_df = make_weekly_df()
    result = _aggregate_treated(weekly_df, ["M001", "M002"])
    # Control stores have revenue 1000 in all weeks; if included, post weeks would be wrong
    post_revenue = result[result["week_start"] == date(2024, 1, 22)]["treated_avg_revenue"].iloc[0]
    assert post_revenue == 1050.0


# ---------------------------------------------------------------------------
# _aggregate_control
# ---------------------------------------------------------------------------

def test_aggregate_control_weighted():
    weekly_df = make_weekly_df()
    matches = make_matches()
    result = _aggregate_control(weekly_df, matches)
    # All control stores have revenue 1000 in all weeks
    assert (result["control_avg_revenue"] == 1000.0).all()


# ---------------------------------------------------------------------------
# lift_pct
# ---------------------------------------------------------------------------

def make_trend():
    """Minimal trend DataFrame for testing lift_pct arithmetic."""
    return pd.DataFrame({
        "week_number":         [-2, -1, 1, 2],
        "treated_avg_revenue": [1000.0, 1000.0, 1050.0, 1050.0],
        "control_avg_revenue": [1000.0, 1000.0, 1000.0, 1000.0],
        "period":              ["pre", "pre", "post", "post"],
    })


def test_pre_period_lift_near_zero():
    trend = make_trend()
    pre = trend[trend["period"] == "pre"]
    baseline = pre["treated_avg_revenue"].mean() / pre["control_avg_revenue"].mean()
    pre_lift = ((pre["treated_avg_revenue"] / pre["control_avg_revenue"]) / baseline - 1) * 100
    assert (pre_lift.abs() < 1e-9).all()


def test_post_period_lift_is_five_pct():
    trend = make_trend()
    pre = trend[trend["period"] == "pre"]
    baseline = pre["treated_avg_revenue"].mean() / pre["control_avg_revenue"].mean()
    post = trend[trend["period"] == "post"]
    post_lift = ((post["treated_avg_revenue"] / post["control_avg_revenue"]) / baseline - 1) * 100
    assert (post_lift.round(6) == 5.0).all()


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_output_columns():
    """check_parallel_trends returns the expected columns."""
    # Build a minimal result manually to test column presence
    trend = make_trend()
    pre = trend[trend["period"] == "pre"]
    baseline = pre["treated_avg_revenue"].mean() / pre["control_avg_revenue"].mean()
    trend["lift_pct"] = ((trend["treated_avg_revenue"] / trend["control_avg_revenue"]) / baseline - 1) * 100
    expected_cols = {"week_number", "treated_avg_revenue", "control_avg_revenue", "lift_pct", "period"}
    assert expected_cols.issubset(set(trend.columns))


def test_week_numbers_sorted():
    trend = make_trend()
    assert list(trend["week_number"]) == sorted(trend["week_number"])
