from datetime import date, timedelta
from typing import Any

import pandas as pd


def build_control_group(
    intervention_id: str,
    spark: Any = None,
    n_matches: int = 10,
) -> pd.DataFrame:
    """
    Proposes a control group for a given intervention using nearest-neighbor
    matching on normalized weekly pre-period revenue.

    WHEN TO USE:
        When the user asks about causal impact of an intervention. Call this
        first to build the control group before checking parallel trends or
        estimating lift.

    WHEN NOT TO USE:
        Descriptive questions — use query_metric instead.

    Parameters
    ----------
    intervention_id : str
        ID of the intervention to analyze. Looks up treated stores and
        start date from interventions_agent_view.
    n_matches : int
        Number of control stores to match per treated store. Default 10.
    spark : SparkSession, optional
        Active Spark session. Resolved automatically in Databricks.

    Returns
    -------
    pd.DataFrame
        Columns: treated_merchant_id, control_merchant_id, similarity_score.
        One row per treated-control pair.
    """
    spark = _resolve_spark(spark)
    treated_ids, start_date = _get_intervention(intervention_id, spark)

    pre_start = start_date - timedelta(weeks=52)
    pre_end   = start_date - timedelta(days=1)

    weekly_df = _get_weekly_revenue(spark, pre_start, pre_end)
    pivot     = _normalize(weekly_df)
    matches   = _match(pivot, treated_ids, n_matches)

    return matches


def _get_intervention(intervention_id: str, spark: Any) -> tuple[list[str], date]:
    """Returns (treated_merchant_ids, start_date) from interventions_agent_view."""
    row = (
        spark.table("main.coffee_analytics.interventions_agent_view")
        .filter(f"intervention_id = '{intervention_id}'")
        .collect()[0]
    )
    return row["treated_merchant_ids"], row["start_date"]


def _get_weekly_revenue(spark: Any, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: merchant_id, week_start, weekly_revenue.
    One row per store per week over the pre-period.
    """
    from pyspark.sql import functions as F

    return (
        spark.table("main.coffee_analytics_gold.transactions_enriched")
        .filter(f"txn_date BETWEEN '{start_date}' AND '{end_date}'")
        .withColumn("week_start", F.date_trunc("week", "txn_date"))
        .groupBy("merchant_id", "week_start")
        .agg(F.sum("amount").alias("weekly_revenue"))
        .orderBy("merchant_id", "week_start")
        .toPandas()
    )


def _normalize(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Divides each store's weekly_revenue by its mean.
    Returns a pivot table: rows = week_start, columns = merchant_id, values = normalized revenue.
    """
    mean_revenue = weekly_df.groupby("merchant_id")["weekly_revenue"].transform("mean")
    weekly_df = weekly_df.copy()
    weekly_df["normalized_revenue"] = weekly_df["weekly_revenue"] / mean_revenue

    return weekly_df.pivot(index="week_start", columns="merchant_id", values="normalized_revenue")


def _match(
    pivot: pd.DataFrame,
    treated_ids: list[str],
    n_matches: int = 10,
) -> pd.DataFrame:
    """
    For each treated store, finds the n_matches closest control stores
    by sum of absolute residuals on the normalized weekly revenue series.
    Returns DataFrame with: treated_merchant_id, control_merchant_id, similarity_score.
    """
    control_ids = [c for c in pivot.columns if c not in treated_ids]
    rows = []

    for treated_id in treated_ids:
        treated_series = pivot[treated_id]

        residuals = (
            pivot[control_ids]
            .subtract(treated_series, axis=0)
            .abs()
            .sum()
        )

        top_matches = residuals.nsmallest(n_matches)

        for control_id, residual_sum in top_matches.items():
            rows.append({
                "treated_merchant_id": treated_id,
                "control_merchant_id": control_id,
                "similarity_score":    round(1 / (1 + residual_sum), 4),
            })

    return pd.DataFrame(rows)


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
