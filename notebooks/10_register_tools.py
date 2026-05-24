# Databricks notebook source
# MAGIC %md
# MAGIC # 10 — Register Tools
# MAGIC
# MAGIC Registers agent tools as Unity Catalog functions so the Mosaic AI Agent
# MAGIC Framework can discover and call them.
# MAGIC
# MAGIC Tools registered here:
# MAGIC - `query_metric` — descriptive analytics (Mode 1)
# MAGIC - `build_control_group` - causal impact analysis (Mode 2)
# MAGIC
# MAGIC Idempotent: safe to re-run (uses CREATE OR REPLACE).

# COMMAND ----------

CATALOG = "main"
SCHEMA  = "coffee_analytics_gold"

# COMMAND ----------

# MAGIC %md
# MAGIC ## query_metric

# COMMAND ----------

# Drop the existing Python function first
spark.sql(f"DROP FUNCTION IF EXISTS {CATALOG}.{SCHEMA}.query_metric")

# Now create the SQL version
spark.sql(f"""
CREATE FUNCTION {CATALOG}.{SCHEMA}.query_metric(
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
RETURN 
    SELECT 
        CASE WHEN group_by IS NOT NULL THEN 
            CASE group_by
                WHEN 'location_type' THEN location_type
                WHEN 'region' THEN region
                WHEN 'size_band' THEN size_band
                WHEN 'brand' THEN brand
            END
        END AS dimension,
        CASE metric
            WHEN 'revenue' THEN ROUND(SUM(amount), 2)
            WHEN 'transaction_count' THEN SUM(txn_count)
            WHEN 'avg_basket' THEN ROUND(SUM(amount) / NULLIF(SUM(txn_count), 0), 2)
            WHEN 'active_merchants' THEN COUNT(DISTINCT merchant_id)
        END AS metric_value
    FROM {CATALOG}.{SCHEMA}.transactions_enriched
    WHERE txn_date BETWEEN start_date AND end_date
        AND (filter_location_type IS NULL OR location_type = filter_location_type)
        AND (filter_region IS NULL OR region = filter_region)
        AND (filter_size_band IS NULL OR size_band = filter_size_band)
        AND (filter_brand IS NULL OR brand = filter_brand)
    GROUP BY 
        CASE WHEN group_by IS NOT NULL THEN 
            CASE group_by
                WHEN 'location_type' THEN location_type
                WHEN 'region' THEN region
                WHEN 'size_band' THEN size_band
                WHEN 'brand' THEN brand
            END
        END
    ORDER BY dimension
""")
print("✓ registered: query_metric")

# COMMAND ----------

# MAGIC %md
# MAGIC ###Smoke tests

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

# COMMAND ----------

# MAGIC %md
# MAGIC ## build_control_group

# COMMAND ----------

spark.sql(f"DROP FUNCTION IF EXISTS {CATALOG}.{SCHEMA}.build_control_group")

spark.sql(f"""
CREATE FUNCTION {CATALOG}.{SCHEMA}.build_control_group(
    intervention_id STRING COMMENT 'ID of the intervention to analyze (e.g. INT_001).',
    n_matches       INT    COMMENT 'Number of control stores to match per treated store. Default 10.'
)
RETURNS TABLE
COMMENT 'Build a matched control group for a causal analysis of a given intervention.
USE FOR: causal questions — did an intervention work, what was the lift, did the pilot increase revenue.
DO NOT USE FOR: descriptive questions about general sales data — use query_metric instead.

Matches control stores to treated stores using normalized weekly pre-period revenue similarity.
Returns one row per treated-control pair with a similarity score (0-1, higher is better).'
RETURN
  WITH intervention_meta AS (
    SELECT 
      intervention_id,
      treated_merchant_ids,
      start_date,
      DATE_SUB(start_date, 364) AS pre_start,
      DATE_SUB(start_date, 1) AS pre_end
    FROM main.coffee_analytics.interventions_agent_view
    WHERE intervention_id = build_control_group.intervention_id
  ),
  -- Explode treated merchant IDs array into rows
  treated_merchants AS (
    SELECT 
      EXPLODE(treated_merchant_ids) AS merchant_id,
      pre_start,
      pre_end
    FROM intervention_meta
  ),
  -- Get weekly revenue for all merchants in pre-period
  weekly_revenue AS (
    SELECT 
      t.merchant_id,
      DATE_TRUNC('week', t.txn_date) AS week_start,
      SUM(t.amount) AS weekly_revenue
    FROM {CATALOG}.{SCHEMA}.transactions_enriched t
    CROSS JOIN intervention_meta i
    WHERE t.txn_date BETWEEN i.pre_start AND i.pre_end
    GROUP BY t.merchant_id, DATE_TRUNC('week', t.txn_date)
  ),
  -- Calculate mean revenue per merchant
  merchant_means AS (
    SELECT 
      merchant_id,
      AVG(weekly_revenue) AS mean_revenue
    FROM weekly_revenue
    GROUP BY merchant_id
  ),
  -- Normalize weekly revenue by merchant mean
  normalized_revenue AS (
    SELECT 
      w.merchant_id,
      w.week_start,
      w.weekly_revenue / NULLIF(m.mean_revenue, 0) AS normalized_revenue
    FROM weekly_revenue w
    JOIN merchant_means m ON w.merchant_id = m.merchant_id
  ),
  -- Create treated-control pairs and calculate similarity
  similarity_scores AS (
    SELECT 
      treated.merchant_id AS treated_merchant_id,
      control.merchant_id AS control_merchant_id,
      SUM(ABS(treated.normalized_revenue - control.normalized_revenue)) AS total_residual
    FROM normalized_revenue treated
    JOIN normalized_revenue control 
      ON treated.week_start = control.week_start
    WHERE treated.merchant_id IN (SELECT merchant_id FROM treated_merchants)
      AND control.merchant_id NOT IN (SELECT merchant_id FROM treated_merchants)
    GROUP BY treated.merchant_id, control.merchant_id
  ),
  -- Rank control merchants by similarity (lower residual = better match)
  ranked_matches AS (
    SELECT 
      treated_merchant_id,
      control_merchant_id,
      ROUND(1.0 / (1.0 + total_residual), 4) AS similarity_score,
      ROW_NUMBER() OVER (PARTITION BY treated_merchant_id ORDER BY total_residual ASC) AS rank
    FROM similarity_scores
  )
  SELECT 
    treated_merchant_id,
    control_merchant_id,
    similarity_score
  FROM ranked_matches
  WHERE rank <= build_control_group.n_matches
  ORDER BY treated_merchant_id, similarity_score DESC
""")
print("✓ registered: build_control_group")

# COMMAND ----------

# Smoke test — build control group for the mobile order pilot
result = spark.sql(f"""
    SELECT * FROM {CATALOG}.{SCHEMA}.build_control_group('INT_001', 10)
""")
display(result)


# COMMAND ----------

# Confirm shape — should have 25 treated stores * 10 matches = 250 rows
n = spark.sql(f"""
    SELECT COUNT(*) as n FROM {CATALOG}.{SCHEMA}.build_control_group('INT_001', 10)
""").collect()[0]["n"]
print(f"Row count: {n} (expected 250)")
assert n == 250, f"Expected 250, got {n}"


# COMMAND ----------

# Confirm similarity scores are between 0 and 1
spark.sql(f"""
    SELECT MIN(similarity_score), MAX(similarity_score)
    FROM {CATALOG}.{SCHEMA}.build_control_group('INT_001', 10)
""").show()

