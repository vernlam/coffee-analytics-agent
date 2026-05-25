import os
import time
from typing import Optional

import pandas as pd
import requests


def _execute_sql(sql: str) -> pd.DataFrame:
    host = os.environ["DATABRICKS_HOST"].rstrip("/")
    token = os.environ["DATABRICKS_TOKEN"]
    warehouse_id = os.environ["DATABRICKS_WAREHOUSE_ID"]
    headers = {"Authorization": f"Bearer {token}"}

    response = requests.post(
        f"{host}/api/2.0/sql/statements",
        headers=headers,
        json={"warehouse_id": warehouse_id, "statement": sql, "wait_timeout": "50s"},
    )
    response.raise_for_status()
    data = response.json()

    while data["status"]["state"] in ("PENDING", "RUNNING"):
        time.sleep(1)
        data = requests.get(
            f"{host}/api/2.0/sql/statements/{data['statement_id']}",
            headers=headers,
        ).json()

    if data["status"]["state"] != "SUCCEEDED":
        raise RuntimeError(f"Query failed: {data['status'].get('error', {}).get('message', data['status']['state'])}")

    columns = [col["name"] for col in data["manifest"]["schema"]["columns"]]
    rows = data.get("result", {}).get("data_array") or []
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


def call_lookup_intervention(name_query: str) -> str | None:
    safe = name_query.replace("'", "").replace("%", "")
    pattern = "%" + "%".join(safe.split()) + "%"
    df = _execute_sql(
        f"SELECT intervention_id, name "
        f"FROM main.coffee_analytics.interventions_agent_view "
        f"WHERE LOWER(name) LIKE LOWER('{pattern}') LIMIT 1"
    )
    return df.iloc[0]["intervention_id"] if not df.empty else None


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
