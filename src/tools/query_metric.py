"""
query_metric — descriptive analytics tool for the coffee analytics agent.

WHEN TO USE:
    Descriptive ("what happened") questions: totals, averages, trends, segment
    breakdowns. Any question that can be answered by aggregating transaction or
    merchant data over a date range.

WHEN NOT TO USE:
    Causal questions — "did the intervention work?", "what was the lift?",
    "was the mobile pilot effective?". Route those to the causal tools:
    build_control_group → check_parallel_trends → estimate_lift.

Examples
--------
    # Total revenue for Q1 2024
    query_metric(
        metric="revenue",
        date_range=(date(2024, 1, 1), date(2024, 3, 31)),
    )
    #    revenue
    # 0  452318.42

    # Transaction count by region, urban locations only, full year 2023
    query_metric(
        metric="transaction_count",
        group_by="region",
        filters={"location_type": "urban"},
        date_range=(date(2023, 1, 1), date(2023, 12, 31)),
    )
    #        region  transaction_count
    # 0   midwest             184201
    # 1  northeast             210443
    # 2  southeast             197832
    # 3       west             201109

    # Average basket size by size band
    query_metric(
        metric="avg_basket",
        group_by="size_band",
        date_range=(date(2024, 1, 1), date(2024, 6, 30)),
    )
    #   size_band  avg_basket
    # 0     large        8.21
    # 1       mid        7.18
    # 2     small        6.44
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

import pandas as pd

from src.config import TRANSACTIONS_ENRICHED

# ---------------------------------------------------------------------------
# Allowed values — validated strictly so the LLM can't inject arbitrary SQL
# ---------------------------------------------------------------------------

VALID_METRICS = {"revenue", "transaction_count", "avg_basket", "active_merchants"}

VALID_GROUP_BY = {"location_type", "region", "size_band", "brand"}

VALID_FILTER_KEYS = {"location_type", "region", "size_band", "brand"}

VALID_FILTER_VALUES: Dict[str, set] = {
    "location_type": {"urban", "suburban", "highway", "mall", "campus"},
    "region":        {"northeast", "southeast", "midwest", "west"},
    "size_band":     {"small", "mid", "large"},
    "brand":         {"BrandA", "BrandB", "BrandC", "BrandD"},
}

# ---------------------------------------------------------------------------
# SQL fragments per metric
# ---------------------------------------------------------------------------

_METRIC_SQL = {
    "revenue":           "ROUND(SUM(amount), 2)          AS revenue",
    "transaction_count": "SUM(txn_count)                 AS transaction_count",
    "avg_basket":        "ROUND(SUM(amount) / SUM(txn_count), 2) AS avg_basket",
    "active_merchants":  "COUNT(DISTINCT merchant_id)    AS active_merchants",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_metric(
    metric: str,
    date_range: tuple[date, date],
    group_by: Optional[str] = None,
    filters: Optional[Dict[str, str]] = None,
    spark: Any = None,
) -> pd.DataFrame:
    """
    Aggregate a single metric from gold.transactions_enriched.

    Parameters
    ----------
    metric : str
        What to measure. One of:
        - "revenue"           — sum of daily revenue (dollars)
        - "transaction_count" — sum of daily transaction counts
        - "avg_basket"        — revenue / transaction_count (average ticket size)
        - "active_merchants"  — count of distinct merchants with any activity
    date_range : tuple[date, date]
        Inclusive (start, end) date range to query.
    group_by : str, optional
        Dimension to segment by. One of:
        "location_type", "region", "size_band", "brand".
        If None, returns a single aggregate row.
    filters : dict, optional
        Equality filters on merchant attributes. Keys and values must be from
        the allowed sets. Example: {"location_type": "urban", "size_band": "large"}.
    spark : SparkSession, optional
        Active Spark session. If None, resolves via SparkSession.getActiveSession().

    Returns
    -------
    pd.DataFrame
        One row per group_by value (or one row if group_by is None),
        sorted by group_by. Columns: [group_by (if set), metric].

    Raises
    ------
    ValueError
        If metric, group_by, filter keys, or filter values are not in the
        allowed sets, or if date_range is inverted.
    """
    filters = filters or {}

    _validate(metric, group_by, filters, date_range)

    sql = _build_sql(metric, group_by, filters, date_range)

    spark = _resolve_spark(spark)
    return spark.sql(sql).toPandas()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(
    metric: str,
    group_by: Optional[str],
    filters: Dict[str, str],
    date_range: tuple[date, date],
) -> None:
    if metric not in VALID_METRICS:
        raise ValueError(
            f"Invalid metric {metric!r}. Valid values: {sorted(VALID_METRICS)}"
        )

    if group_by is not None and group_by not in VALID_GROUP_BY:
        raise ValueError(
            f"Invalid group_by {group_by!r}. Valid values: {sorted(VALID_GROUP_BY)}"
        )

    for key, val in filters.items():
        if key not in VALID_FILTER_KEYS:
            raise ValueError(
                f"Invalid filter key {key!r}. Valid keys: {sorted(VALID_FILTER_KEYS)}"
            )
        allowed_vals = VALID_FILTER_VALUES[key]
        if val not in allowed_vals:
            raise ValueError(
                f"Invalid filter value {val!r} for key {key!r}. "
                f"Valid values: {sorted(allowed_vals)}"
            )

    start, end = date_range
    if start > end:
        raise ValueError(
            f"date_range start ({start}) must be <= end ({end})"
        )


# ---------------------------------------------------------------------------
# SQL construction
# ---------------------------------------------------------------------------

def _build_sql(
    metric: str,
    group_by: Optional[str],
    filters: Dict[str, str],
    date_range: tuple[date, date],
) -> str:
    start, end = date_range

    select_cols = []
    if group_by:
        select_cols.append(group_by)
    select_cols.append(_METRIC_SQL[metric])

    where_clauses = [
        f"txn_date BETWEEN '{start}' AND '{end}'"
    ]
    for key, val in filters.items():
        where_clauses.append(f"{key} = '{val}'")

    sql = (
        f"SELECT {', '.join(select_cols)}\n"
        f"FROM {TRANSACTIONS_ENRICHED}\n"
        f"WHERE {' AND '.join(where_clauses)}\n"
    )

    if group_by:
        sql += f"GROUP BY {group_by}\n"
        sql += f"ORDER BY {group_by}\n"

    return sql


# ---------------------------------------------------------------------------
# Spark resolution
# ---------------------------------------------------------------------------

def _resolve_spark(spark: Any) -> Any:
    if spark is not None:
        return spark
    try:
        from pyspark.sql import SparkSession
        session = SparkSession.getActiveSession()
        if session is None:
            raise RuntimeError("No active SparkSession found.")
        return session
    except ImportError:
        raise RuntimeError(
            "pyspark is not available. Pass a spark session explicitly or "
            "run this function in a Databricks environment."
        )
