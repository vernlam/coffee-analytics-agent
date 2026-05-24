from datetime import date
from typing import Any

import pandas as pd


def check_parallel_trends(
    matches: pd.DataFrame,
    intervention_id: str,
    spark: Any = None,
) -> pd.DataFrame:
    """
    Computes weekly lift % relative to pre-period baseline, for pre- and post-period.

    WHEN TO USE:
        After build_control_group. Pass its output here to produce the parallel
        trends evidence for the human-in-the-loop checkpoint.

    WHEN NOT TO USE:
        Descriptive questions — use query_metric instead.

    Parameters
    ----------
    matches : pd.DataFrame
        Output of build_control_group. Columns: treated_merchant_id,
        control_merchant_id, similarity_score.
    intervention_id : str
        Used to look up start_date, pre_window_days, post_window_days.
    spark : SparkSession, optional
        Active Spark session. Resolved automatically in Databricks.

    Returns
    -------
    pd.DataFrame
        Columns: week_number, treated_avg_revenue, control_avg_revenue,
        lift_pct, period.
        week_number is relative to intervention start (negative = pre, positive = post).
        lift_pct is normalized against the pre-period baseline ratio.
    """
    spark = _resolve_spark(spark)
    start_date, pre_window_days, post_window_days = _get_intervention_dates(intervention_id, spark)

    pre_start = pd.Timestamp(start_date) - pd.Timedelta(days=pre_window_days)
    post_end  = pd.Timestamp(start_date) + pd.Timedelta(days=post_window_days)

    treated_ids = matches["treated_merchant_id"].unique().tolist()
    control_ids = matches["control_merchant_id"].unique().tolist()
    all_ids     = list(set(treated_ids + control_ids))

    weekly_df     = _get_weekly_revenue(spark, pre_start.date(), post_end.date(), all_ids)
    treated_trend = _aggregate_treated(weekly_df, treated_ids)
    control_trend = _aggregate_control(weekly_df, matches)

    trend = treated_trend.merge(control_trend, on="week_start", how="inner")

    # Relative week number — negative = pre, positive = post
    intervention_ts = pd.Timestamp(start_date)
    trend["week_number"] = (
        (pd.to_datetime(trend["week_start"]) - intervention_ts)
        .dt.days // 7
    )
    trend["period"] = trend["week_number"].apply(lambda w: "pre" if w < 0 else "post")

    # Baseline ratio over the entire pre-period
    pre = trend[trend["period"] == "pre"]
    baseline = pre["treated_avg_revenue"].mean() / pre["control_avg_revenue"].mean()

    # lift_pct: deviation from baseline ratio, expressed as %
    trend["lift_pct"] = (
        (trend["treated_avg_revenue"] / trend["control_avg_revenue"]) / baseline - 1
    ) * 100

    return (
        trend[["week_number", "treated_avg_revenue", "control_avg_revenue", "lift_pct", "period"]]
        .sort_values("week_number")
        .reset_index(drop=True)
    )


def _get_intervention_dates(intervention_id: str, spark: Any) -> tuple[date, int, int]:
    """Returns (start_date, pre_window_days, post_window_days)."""
    row = (
        spark.table("main.coffee_analytics.interventions_agent_view")
        .filter(f"intervention_id = '{intervention_id}'")
        .collect()[0]
    )
    return row["start_date"], row["pre_window_days"], row["post_window_days"]


def _get_weekly_revenue(
    spark: Any,
    start_date: date,
    end_date: date,
    merchant_ids: list[str],
) -> pd.DataFrame:
    """Weekly revenue per store over the full window (pre + post)."""
    from pyspark.sql import functions as F

    ids_str = ", ".join(f"'{m}'" for m in merchant_ids)

    return (
        spark.table("main.coffee_analytics_gold.transactions_enriched")
        .filter(f"txn_date BETWEEN '{start_date}' AND '{end_date}'")
        .filter(f"merchant_id IN ({ids_str})")
        .withColumn("week_start", F.date_trunc("week", "txn_date"))
        .groupBy("merchant_id", "week_start")
        .agg(F.sum("amount").alias("weekly_revenue"))
        .toPandas()
    )


def _aggregate_treated(weekly_df: pd.DataFrame, treated_ids: list[str]) -> pd.DataFrame:
    """Simple average weekly revenue across all treated stores."""
    return (
        weekly_df[weekly_df["merchant_id"].isin(treated_ids)]
        .groupby("week_start")["weekly_revenue"]
        .mean()
        .reset_index()
        .rename(columns={"weekly_revenue": "treated_avg_revenue"})
    )


def _aggregate_control(weekly_df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Similarity-weighted average weekly revenue across matched control stores."""
    weights = (
        matches.groupby("control_merchant_id")["similarity_score"]
        .mean()
        .reset_index()
        .rename(columns={"control_merchant_id": "merchant_id", "similarity_score": "weight"})
    )

    control_df = weekly_df.merge(weights, on="merchant_id", how="inner")
    control_df["weighted_revenue"] = control_df["weekly_revenue"] * control_df["weight"]

    return (
        control_df.groupby("week_start")
        .apply(lambda g: g["weighted_revenue"].sum() / g["weight"].sum())
        .reset_index()
        .rename(columns={0: "control_avg_revenue"})
    )


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
