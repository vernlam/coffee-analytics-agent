import os
from typing import Optional

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState


def _execute_sql(sql: str) -> pd.DataFrame:
    w = WorkspaceClient()
    warehouse_id = os.environ["DATABRICKS_WAREHOUSE_ID"]
    response = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
        wait_timeout="50s",
    )
    if response.status.state != StatementState.SUCCEEDED:
        error_msg = (
            response.status.error.message
            if response.status.error
            else str(response.status.state)
        )
        raise RuntimeError(f"Query failed: {error_msg}")
    columns = [col.name for col in response.manifest.schema.columns]
    rows = response.result.data_array if response.result and response.result.data_array else []
    return pd.DataFrame(rows, columns=columns)


def _to_markdown(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join([header, sep] + rows)


def call_build_control_group(intervention_id: str, n_matches: int = 10) -> pd.DataFrame:
    return _execute_sql(
        f"SELECT * FROM main.coffee_analytics_gold.build_control_group('{intervention_id}', {n_matches})"
    )


def call_check_parallel_trends(intervention_id: str, n_matches: int = 10) -> pd.DataFrame:
    return _execute_sql(
        f"SELECT * FROM main.coffee_analytics_gold.check_parallel_trends('{intervention_id}', {n_matches})"
        f" ORDER BY week_number"
    )


def call_estimate_lift(
    intervention_id: str,
    n_matches: int = 10,
    first_n_weeks: Optional[int] = None,
    week_start: Optional[int] = None,
    week_end: Optional[int] = None,
) -> pd.DataFrame:
    def _sql_val(v):
        return str(v) if v is not None else "NULL"

    return _execute_sql(
        f"SELECT * FROM main.coffee_analytics_gold.estimate_lift("
        f"'{intervention_id}', {n_matches}, "
        f"{_sql_val(first_n_weeks)}, {_sql_val(week_start)}, {_sql_val(week_end)})"
    )


def call_query_metric(
    metric: str,
    start_date: str,
    end_date: str,
    group_by: Optional[str] = None,
    filter_location_type: Optional[str] = None,
    filter_region: Optional[str] = None,
    filter_size_band: Optional[str] = None,
    filter_brand: Optional[str] = None,
) -> pd.DataFrame:
    def _sql_str(v):
        return f"'{v}'" if v is not None else "NULL"

    return _execute_sql(
        f"SELECT * FROM main.coffee_analytics_gold.query_metric("
        f"'{metric}', '{start_date}', '{end_date}', "
        f"{_sql_str(group_by)}, {_sql_str(filter_location_type)}, "
        f"{_sql_str(filter_region)}, {_sql_str(filter_size_band)}, {_sql_str(filter_brand)})"
    )
