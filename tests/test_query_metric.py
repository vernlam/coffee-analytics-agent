"""Tests for src/tools/query_metric.py.

Runs entirely locally — no Spark or Databricks connection needed.
Validation and SQL-construction tests call internal helpers directly.
End-to-end tests pass a MagicMock as the spark argument.
"""

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.tools.query_metric import (
    _build_sql,
    _validate,
    query_metric,
)

DATE_RANGE = (date(2024, 1, 1), date(2024, 3, 31))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_spark(df: pd.DataFrame) -> MagicMock:
    """Return a mock SparkSession whose .sql().toPandas() returns df."""
    mock = MagicMock()
    mock.sql.return_value.toPandas.return_value = df
    return mock


# ---------------------------------------------------------------------------
# Validation — invalid inputs
# ---------------------------------------------------------------------------

def test_invalid_metric():
    with pytest.raises(ValueError, match="Invalid metric"):
        _validate("sales", None, {}, DATE_RANGE)


def test_invalid_group_by():
    with pytest.raises(ValueError, match="Invalid group_by"):
        _validate("revenue", "merchant_id", {}, DATE_RANGE)


def test_invalid_filter_key():
    with pytest.raises(ValueError, match="Invalid filter key"):
        _validate("revenue", None, {"country": "US"}, DATE_RANGE)


def test_invalid_filter_value():
    with pytest.raises(ValueError, match="Invalid filter value"):
        _validate("revenue", None, {"location_type": "airport"}, DATE_RANGE)


def test_inverted_date_range():
    with pytest.raises(ValueError, match="must be <="):
        _validate("revenue", None, {}, (date(2024, 6, 1), date(2024, 1, 1)))


def test_same_day_date_range_is_valid():
    _validate("revenue", None, {}, (date(2024, 1, 1), date(2024, 1, 1)))


# ---------------------------------------------------------------------------
# SQL construction
# ---------------------------------------------------------------------------

def test_sql_no_groupby_no_filters():
    sql = _build_sql("revenue", None, {}, DATE_RANGE)
    assert "SUM(amount)" in sql
    assert "WHERE txn_date BETWEEN '2024-01-01' AND '2024-03-31'" in sql
    assert "GROUP BY" not in sql


def test_sql_with_groupby():
    sql = _build_sql("revenue", "region", {}, DATE_RANGE)
    assert "region" in sql
    assert "GROUP BY region" in sql
    assert "ORDER BY region" in sql


def test_sql_with_filter():
    sql = _build_sql("transaction_count", None, {"location_type": "urban"}, DATE_RANGE)
    assert "location_type = 'urban'" in sql


def test_sql_multiple_filters():
    sql = _build_sql("revenue", None, {"location_type": "urban", "size_band": "large"}, DATE_RANGE)
    assert "location_type = 'urban'" in sql
    assert "size_band = 'large'" in sql


def test_sql_avg_basket():
    sql = _build_sql("avg_basket", None, {}, DATE_RANGE)
    assert "SUM(amount)" in sql
    assert "SUM(txn_count)" in sql


def test_sql_active_merchants():
    sql = _build_sql("active_merchants", None, {}, DATE_RANGE)
    assert "COUNT(DISTINCT merchant_id)" in sql


# ---------------------------------------------------------------------------
# End-to-end with mock Spark
# ---------------------------------------------------------------------------

def test_returns_dataframe():
    expected = pd.DataFrame({"revenue": [42318.50]})
    result = query_metric("revenue", DATE_RANGE, spark=make_spark(expected))
    pd.testing.assert_frame_equal(result, expected)


def test_spark_receives_correct_sql():
    mock_spark = make_spark(pd.DataFrame({"revenue": [0.0]}))
    query_metric(
        "revenue",
        DATE_RANGE,
        filters={"location_type": "urban"},
        spark=mock_spark,
    )
    sql_sent = mock_spark.sql.call_args[0][0]
    assert "location_type = 'urban'" in sql_sent
    assert "transactions_enriched" in sql_sent


def test_groupby_end_to_end():
    expected = pd.DataFrame({
        "region":  ["midwest", "northeast", "southeast", "west"],
        "revenue": [110000.0, 125000.0, 118000.0, 121000.0],
    })
    mock_spark = make_spark(expected)
    result = query_metric("revenue", DATE_RANGE, group_by="region", spark=mock_spark)
    assert list(result.columns) == ["region", "revenue"]
    assert len(result) == 4


def test_validation_error_before_spark_call():
    mock_spark = make_spark(pd.DataFrame())
    with pytest.raises(ValueError):
        query_metric("bad_metric", DATE_RANGE, spark=mock_spark)
    mock_spark.sql.assert_not_called()
