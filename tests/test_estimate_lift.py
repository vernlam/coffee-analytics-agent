"""Tests for src/tools/estimate_lift.py."""

import pandas as pd
import pytest
from scipy import stats

from src.tools.estimate_lift import estimate_lift


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_trends(post_lifts: list[float], pre_lifts: list[float] = None) -> pd.DataFrame:
    """Build a minimal trends DataFrame with given post-period lift values."""
    if pre_lifts is None:
        pre_lifts = [0.0, 0.0]

    rows = []
    for i, lift in enumerate(pre_lifts):
        rows.append({"week_number": -(len(pre_lifts) - i), "lift_pct": lift, "period": "pre"})
    for i, lift in enumerate(post_lifts):
        rows.append({"week_number": i + 1, "lift_pct": lift, "period": "post"})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lift and CI calculation
# ---------------------------------------------------------------------------

def test_lift_pct_is_mean_of_post_weeks():
    trends = make_trends(post_lifts=[4.0, 6.0, 5.0, 5.0])
    result = estimate_lift(trends)
    assert result["lift_pct"].iloc[0] == 5.0


def test_ci_computed_correctly():
    post_lifts = [4.0, 6.0, 5.0, 5.0]
    trends = make_trends(post_lifts=post_lifts)
    result = estimate_lift(trends)

    import numpy as np
    n       = len(post_lifts)
    mean    = sum(post_lifts) / n
    se      = stats.sem(post_lifts)
    ci      = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)

    assert abs(result["ci_lower"].iloc[0] - round(ci[0], 4)) < 1e-6
    assert abs(result["ci_upper"].iloc[0] - round(ci[1], 4)) < 1e-6


def test_n_weeks_is_correct():
    trends = make_trends(post_lifts=[5.0, 5.0, 5.0])
    result = estimate_lift(trends)
    assert result["n_weeks"].iloc[0] == 3


def test_returns_single_row():
    trends = make_trends(post_lifts=[5.0, 5.0])
    result = estimate_lift(trends)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def test_significant_when_ci_above_zero():
    # Consistent positive lift with some variance → CI above zero
    trends = make_trends(post_lifts=[9.0, 10.0, 11.0, 10.0, 9.5, 10.5, 10.0, 9.8])
    result = estimate_lift(trends)
    assert result["significant"].iloc[0] == True


def test_not_significant_when_ci_crosses_zero():
    # Mixed lift values → wide CI crossing zero
    trends = make_trends(post_lifts=[-5.0, 5.0, -5.0, 5.0])
    result = estimate_lift(trends)
    assert result["significant"].iloc[0] == False


# ---------------------------------------------------------------------------
# Window filtering
# ---------------------------------------------------------------------------

def test_first_n_weeks_restricts_post_period():
    trends = make_trends(post_lifts=[10.0, 10.0, 0.0, 0.0])
    result = estimate_lift(trends, first_n_weeks=2)
    assert result["n_weeks"].iloc[0] == 2
    assert result["lift_pct"].iloc[0] == 10.0


def test_week_start_filters_correctly():
    trends = make_trends(post_lifts=[0.0, 0.0, 10.0, 10.0])
    result = estimate_lift(trends, week_start=3)
    assert result["n_weeks"].iloc[0] == 2
    assert result["lift_pct"].iloc[0] == 10.0


def test_week_end_filters_correctly():
    trends = make_trends(post_lifts=[10.0, 10.0, 0.0, 0.0])
    result = estimate_lift(trends, week_end=2)
    assert result["n_weeks"].iloc[0] == 2
    assert result["lift_pct"].iloc[0] == 10.0


def test_week_start_and_end_combined():
    trends = make_trends(post_lifts=[0.0, 10.0, 10.0, 0.0])
    result = estimate_lift(trends, week_start=2, week_end=3)
    assert result["n_weeks"].iloc[0] == 2
    assert result["lift_pct"].iloc[0] == 10.0


def test_raises_with_fewer_than_two_weeks():
    trends = make_trends(post_lifts=[5.0])
    with pytest.raises(ValueError, match="at least 2"):
        estimate_lift(trends)
