# Databricks notebook source
# MAGIC %md
# MAGIC # 10 — Register Tools
# MAGIC
# MAGIC Registers agent tools as Unity Catalog functions so the Mosaic AI Agent
# MAGIC Framework can discover and call them.
# MAGIC
# MAGIC Tools registered here:
# MAGIC - `query_metric` — descriptive analytics (Mode 1)
# MAGIC
# MAGIC Idempotent: safe to re-run (uses CREATE OR REPLACE).

# COMMAND ----------

CATALOG = "main"
SCHEMA  = "coffee_analytics_gold"

# COMMAND ----------

# MAGIC %md
# MAGIC ## query_metric

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.query_metric(
    metric              STRING  COMMENT 'What to measure. One of: revenue, transaction_count, avg_basket, active_merchants.',
    start_date          DATE    COMMENT 'Inclusive start of the date range.',
    end_date            DATE    COMMENT 'Inclusive end of the date range.',
    group_by            STRING  COMMENT 'Optional dimension to segment by. One of: location_type, region, size_band, brand. Pass NULL for a single aggregate row.',
    filter_location_type STRING COMMENT 'Optional equality filter on location_type. One of: urban, suburban, highway, mall, campus. Pass NULL to include all.',
    filter_region        STRING COMMENT 'Optional equality filter on region. One of: northeast, southeast, midwest, west. Pass NULL to include all.',
    filter_size_band     STRING COMMENT 'Optional equality filter on size_band. One of: small, mid, large. Pass NULL to include all.',
    filter_brand         STRING COMMENT 'Optional equality filter on brand. One of: BrandA, BrandB, BrandC, BrandD. Pass NULL to include all.'
)
RETURNS TABLE
COMMENT 'Aggregate a descriptive metric from transactions_enriched over a date range.
USE FOR: "what happened" questions — totals, averages, trends, segment breakdowns.
DO NOT USE FOR: causal questions about whether an intervention worked. Route those to build_control_group.'
LANGUAGE PYTHON
AS $$
METRIC_SQL = {{
    "revenue":           "ROUND(SUM(amount), 2) AS revenue",
    "transaction_count": "SUM(txn_count) AS transaction_count",
    "avg_basket":        "ROUND(SUM(amount) / SUM(txn_count), 2) AS avg_basket",
    "active_merchants":  "COUNT(DISTINCT merchant_id) AS active_merchants",
}}
TABLE = "main.coffee_analytics_gold.transactions_enriched"

def query_metric(metric, start_date, end_date, group_by, filter_location_type, filter_region, filter_size_band, filter_brand):
    select_cols = []
    if group_by:
        select_cols.append(group_by)
    select_cols.append(METRIC_SQL[metric])

    where = [f"txn_date BETWEEN '{{start_date}}' AND '{{end_date}}'"]
    for key, val in {{"location_type": filter_location_type, "region": filter_region, "size_band": filter_size_band, "brand": filter_brand}}.items():
        if val is not None:
            where.append(f"{{key}} = '{{val}}'")

    sql = f"SELECT {{', '.join(select_cols)}} FROM {{TABLE}} WHERE {{' AND '.join(where)}}"
    if group_by:
        sql += f" GROUP BY {{group_by}} ORDER BY {{group_by}}"

    return sql
$$
""")
print("✓ registered: query_metric")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke tests

# COMMAND ----------

# Total revenue, full year 2024
display(spark.sql(f"""
    SELECT * FROM {CATALOG}.{SCHEMA}.query_metric(
        'revenue', '2024-01-01', '2024-12-31', NULL, NULL, NULL, NULL, NULL
    )
"""))

# COMMAND ----------

# Revenue by region, urban stores only, Q1 2024
display(spark.sql(f"""
    SELECT * FROM {CATALOG}.{SCHEMA}.query_metric(
        'revenue', '2024-01-01', '2024-03-31', 'region', 'urban', NULL, NULL, NULL
    )
"""))

# COMMAND ----------

# Average basket by size band, full history
display(spark.sql(f"""
    SELECT * FROM {CATALOG}.{SCHEMA}.query_metric(
        'avg_basket', '2023-01-01', '2024-12-31', 'size_band', NULL, NULL, NULL, NULL
    )
"""))
