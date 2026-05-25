# Databricks notebook source

CATALOG = "main"
SCHEMA  = "coffee_analytics_gold"

spark.sql("CREATE SCHEMA IF NOT EXISTS main.coffee_analytics_temp")

# COMMAND ----------

spark.sql(f"DROP FUNCTION IF EXISTS {CATALOG}.{SCHEMA}.query_metric")

spark.sql(f"""
CREATE FUNCTION {CATALOG}.{SCHEMA}.query_metric(
    metric               STRING,
    start_date           DATE,
    end_date             DATE,
    group_by             STRING,
    filter_location_type STRING,
    filter_region        STRING,
    filter_size_band     STRING,
    filter_brand         STRING
)
RETURNS TABLE
RETURN
    SELECT
        CASE WHEN group_by IS NOT NULL THEN
            CASE group_by
                WHEN 'location_type' THEN location_type
                WHEN 'region'        THEN region
                WHEN 'size_band'     THEN size_band
                WHEN 'brand'         THEN brand
            END
        END AS dimension,
        CASE metric
            WHEN 'revenue'           THEN ROUND(SUM(amount), 2)
            WHEN 'transaction_count' THEN SUM(txn_count)
            WHEN 'avg_basket'        THEN ROUND(SUM(amount) / NULLIF(SUM(txn_count), 0), 2)
            WHEN 'active_merchants'  THEN COUNT(DISTINCT merchant_id)
        END AS metric_value
    FROM {CATALOG}.{SCHEMA}.transactions_enriched
    WHERE txn_date BETWEEN start_date AND end_date
        AND (filter_location_type IS NULL OR location_type = filter_location_type)
        AND (filter_region        IS NULL OR region        = filter_region)
        AND (filter_size_band     IS NULL OR size_band     = filter_size_band)
        AND (filter_brand         IS NULL OR brand         = filter_brand)
    GROUP BY
        CASE WHEN group_by IS NOT NULL THEN
            CASE group_by
                WHEN 'location_type' THEN location_type
                WHEN 'region'        THEN region
                WHEN 'size_band'     THEN size_band
                WHEN 'brand'         THEN brand
            END
        END
    ORDER BY dimension
""")
print("✓ registered: query_metric")

# COMMAND ----------

display(spark.sql(f"""
    SELECT * FROM {CATALOG}.{SCHEMA}.query_metric(
        'revenue', '2024-01-01', '2024-12-31', NULL, NULL, NULL, NULL, NULL
    )
"""))

# COMMAND ----------

spark.sql(f"DROP FUNCTION IF EXISTS {CATALOG}.{SCHEMA}.build_control_group")

spark.sql(f"""
CREATE FUNCTION {CATALOG}.{SCHEMA}.build_control_group(
    intervention_id STRING,
    n_matches       INT
)
RETURNS TABLE
RETURN
  WITH intervention_meta AS (
    SELECT
      treated_merchant_ids,
      start_date,
      DATE_SUB(start_date, 364) AS pre_start,
      DATE_SUB(start_date, 1)   AS pre_end
    FROM main.coffee_analytics.interventions_agent_view
    WHERE intervention_id = build_control_group.intervention_id
  ),
  treated_merchants AS (
    SELECT EXPLODE(treated_merchant_ids) AS merchant_id, pre_start, pre_end
    FROM intervention_meta
  ),
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
  merchant_means AS (
    SELECT merchant_id, AVG(weekly_revenue) AS mean_revenue
    FROM weekly_revenue
    GROUP BY merchant_id
  ),
  normalized_revenue AS (
    SELECT
      w.merchant_id,
      w.week_start,
      w.weekly_revenue / NULLIF(m.mean_revenue, 0) AS normalized_revenue
    FROM weekly_revenue w
    JOIN merchant_means m ON w.merchant_id = m.merchant_id
  ),
  similarity_scores AS (
    SELECT
      treated.merchant_id AS treated_merchant_id,
      control.merchant_id AS control_merchant_id,
      SUM(ABS(treated.normalized_revenue - control.normalized_revenue)) AS total_residual
    FROM normalized_revenue treated
    JOIN normalized_revenue control ON treated.week_start = control.week_start
    WHERE treated.merchant_id IN (SELECT merchant_id FROM treated_merchants)
      AND control.merchant_id NOT IN (SELECT merchant_id FROM treated_merchants)
    GROUP BY treated.merchant_id, control.merchant_id
  ),
  ranked_matches AS (
    SELECT
      treated_merchant_id,
      control_merchant_id,
      ROUND(1.0 / (1.0 + total_residual), 4) AS similarity_score,
      ROW_NUMBER() OVER (PARTITION BY treated_merchant_id ORDER BY total_residual ASC) AS rank
    FROM similarity_scores
  )
  SELECT treated_merchant_id, control_merchant_id, similarity_score
  FROM ranked_matches
  WHERE rank <= build_control_group.n_matches
  ORDER BY treated_merchant_id, similarity_score DESC
""")
print("✓ registered: build_control_group")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.build_control_group('INT_001', 10)"))

# COMMAND ----------

spark.sql(f"DROP FUNCTION IF EXISTS {CATALOG}.{SCHEMA}.check_parallel_trends")

spark.sql(f"""
CREATE FUNCTION {CATALOG}.{SCHEMA}.check_parallel_trends(
    intervention_id STRING,
    n_matches       INT
)
RETURNS TABLE
RETURN
  WITH intervention_meta AS (
    SELECT
      treated_merchant_ids,
      start_date,
      DATE_SUB(start_date, 364) AS match_pre_start,
      DATE_SUB(start_date, 1)   AS match_pre_end,
      DATE_SUB(start_date, 364) AS trend_pre_start,
      DATE_ADD(start_date, CAST(post_window_days AS INT)) AS trend_post_end
    FROM main.coffee_analytics.interventions_agent_view
    WHERE intervention_id = check_parallel_trends.intervention_id
  ),
  treated_merchants AS (
    SELECT EXPLODE(treated_merchant_ids) AS merchant_id
    FROM intervention_meta
  ),
  weekly_revenue_match AS (
    SELECT
      t.merchant_id,
      DATE_TRUNC('week', t.txn_date) AS week_start,
      SUM(t.amount) AS weekly_revenue
    FROM main.coffee_analytics_gold.transactions_enriched t
    CROSS JOIN intervention_meta i
    WHERE t.txn_date BETWEEN i.match_pre_start AND i.match_pre_end
    GROUP BY t.merchant_id, DATE_TRUNC('week', t.txn_date)
  ),
  merchant_means AS (
    SELECT merchant_id, AVG(weekly_revenue) AS mean_revenue
    FROM weekly_revenue_match
    GROUP BY merchant_id
  ),
  normalized_revenue AS (
    SELECT
      w.merchant_id,
      w.week_start,
      w.weekly_revenue / NULLIF(m.mean_revenue, 0) AS normalized_revenue
    FROM weekly_revenue_match w
    JOIN merchant_means m ON w.merchant_id = m.merchant_id
  ),
  similarity_scores AS (
    SELECT
      treated.merchant_id AS treated_merchant_id,
      control.merchant_id AS control_merchant_id,
      SUM(ABS(treated.normalized_revenue - control.normalized_revenue)) AS total_residual
    FROM normalized_revenue treated
    JOIN normalized_revenue control ON treated.week_start = control.week_start
    WHERE treated.merchant_id IN (SELECT merchant_id FROM treated_merchants)
      AND control.merchant_id NOT IN (SELECT merchant_id FROM treated_merchants)
    GROUP BY treated.merchant_id, control.merchant_id
  ),
  ranked_matches AS (
    SELECT
      treated_merchant_id,
      control_merchant_id,
      ROUND(1.0 / (1.0 + total_residual), 4) AS similarity_score,
      ROW_NUMBER() OVER (PARTITION BY treated_merchant_id ORDER BY total_residual ASC) AS rank
    FROM similarity_scores
  ),
  top_matches AS (
    SELECT treated_merchant_id, control_merchant_id, similarity_score
    FROM ranked_matches
    WHERE rank <= check_parallel_trends.n_matches
  ),
  all_merchant_ids AS (
    SELECT merchant_id FROM treated_merchants
    UNION
    SELECT control_merchant_id AS merchant_id FROM top_matches
  ),
  weekly_revenue_full AS (
    SELECT
      t.merchant_id,
      DATE_TRUNC('week', t.txn_date) AS week_start,
      SUM(t.amount) AS weekly_revenue
    FROM main.coffee_analytics_gold.transactions_enriched t
    CROSS JOIN intervention_meta i
    WHERE t.txn_date BETWEEN i.trend_pre_start AND i.trend_post_end
      AND t.merchant_id IN (SELECT merchant_id FROM all_merchant_ids)
    GROUP BY t.merchant_id, DATE_TRUNC('week', t.txn_date)
  ),
  treated_agg AS (
    SELECT week_start, AVG(weekly_revenue) AS treated_avg_revenue
    FROM weekly_revenue_full
    WHERE merchant_id IN (SELECT merchant_id FROM treated_merchants)
    GROUP BY week_start
  ),
  control_weights AS (
    SELECT control_merchant_id AS merchant_id, AVG(similarity_score) AS weight
    FROM top_matches
    GROUP BY control_merchant_id
  ),
  control_agg AS (
    SELECT
      w.week_start,
      SUM(w.weekly_revenue * cw.weight) / SUM(cw.weight) AS control_avg_revenue
    FROM weekly_revenue_full w
    JOIN control_weights cw ON w.merchant_id = cw.merchant_id
    GROUP BY w.week_start
  ),
  trends AS (
    SELECT
      ta.week_start,
      CAST(FLOOR(DATEDIFF(ta.week_start, i.start_date) / 7.0) AS INT) AS week_number,
      ta.treated_avg_revenue,
      ca.control_avg_revenue,
      CASE WHEN ta.week_start < i.start_date THEN 'pre' ELSE 'post' END AS period
    FROM treated_agg ta
    JOIN control_agg ca ON ta.week_start = ca.week_start
    CROSS JOIN intervention_meta i
  ),
  baseline AS (
    SELECT AVG(treated_avg_revenue / NULLIF(control_avg_revenue, 0)) AS baseline_ratio
    FROM trends WHERE period = 'pre'
  )
  SELECT
    week_number,
    ROUND(treated_avg_revenue, 2) AS treated_avg_revenue,
    ROUND(control_avg_revenue, 2) AS control_avg_revenue,
    ROUND(((treated_avg_revenue / NULLIF(control_avg_revenue, 0)) / NULLIF(baseline_ratio, 0) - 1) * 100, 4) AS lift_pct,
    period
  FROM trends CROSS JOIN baseline
  ORDER BY week_number
""")
print("✓ registered: check_parallel_trends")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.check_parallel_trends('INT_001', 10)"))

# COMMAND ----------

spark.sql(f"DROP FUNCTION IF EXISTS {CATALOG}.{SCHEMA}.estimate_lift")

spark.sql(f"""
CREATE FUNCTION {CATALOG}.{SCHEMA}.estimate_lift(
    intervention_id STRING,
    n_matches       INT,
    first_n_weeks   INT,
    week_start      INT,
    week_end        INT
)
RETURNS TABLE
RETURN
  WITH intervention_meta AS (
    SELECT
      treated_merchant_ids,
      start_date,
      DATE_SUB(start_date, 364)                           AS match_pre_start,
      DATE_SUB(start_date, 1)                             AS match_pre_end,
      DATE_SUB(start_date, 364)                           AS trend_pre_start,
      DATE_ADD(start_date, CAST(post_window_days AS INT)) AS trend_post_end
    FROM main.coffee_analytics.interventions_agent_view
    WHERE intervention_id = estimate_lift.intervention_id
  ),
  treated_merchants AS (
    SELECT EXPLODE(treated_merchant_ids) AS merchant_id
    FROM intervention_meta
  ),
  weekly_revenue_match AS (
    SELECT
      t.merchant_id,
      DATE_TRUNC('week', t.txn_date) AS week_start,
      SUM(t.amount) AS weekly_revenue
    FROM main.coffee_analytics_gold.transactions_enriched t
    CROSS JOIN intervention_meta i
    WHERE t.txn_date BETWEEN i.match_pre_start AND i.match_pre_end
    GROUP BY t.merchant_id, DATE_TRUNC('week', t.txn_date)
  ),
  merchant_means AS (
    SELECT merchant_id, AVG(weekly_revenue) AS mean_revenue
    FROM weekly_revenue_match
    GROUP BY merchant_id
  ),
  normalized_revenue AS (
    SELECT
      w.merchant_id,
      w.week_start,
      w.weekly_revenue / NULLIF(m.mean_revenue, 0) AS normalized_revenue
    FROM weekly_revenue_match w
    JOIN merchant_means m ON w.merchant_id = m.merchant_id
  ),
  similarity_scores AS (
    SELECT
      treated.merchant_id AS treated_merchant_id,
      control.merchant_id AS control_merchant_id,
      SUM(ABS(treated.normalized_revenue - control.normalized_revenue)) AS total_residual
    FROM normalized_revenue treated
    JOIN normalized_revenue control ON treated.week_start = control.week_start
    WHERE treated.merchant_id IN (SELECT merchant_id FROM treated_merchants)
      AND control.merchant_id NOT IN (SELECT merchant_id FROM treated_merchants)
    GROUP BY treated.merchant_id, control.merchant_id
  ),
  ranked_matches AS (
    SELECT
      treated_merchant_id,
      control_merchant_id,
      ROUND(1.0 / (1.0 + total_residual), 4) AS similarity_score,
      ROW_NUMBER() OVER (PARTITION BY treated_merchant_id ORDER BY total_residual ASC) AS rank
    FROM similarity_scores
  ),
  top_matches AS (
    SELECT treated_merchant_id, control_merchant_id, similarity_score
    FROM ranked_matches
    WHERE rank <= estimate_lift.n_matches
  ),
  all_merchant_ids AS (
    SELECT merchant_id FROM treated_merchants
    UNION
    SELECT control_merchant_id AS merchant_id FROM top_matches
  ),
  weekly_revenue_full AS (
    SELECT
      t.merchant_id,
      DATE_TRUNC('week', t.txn_date) AS week_start,
      SUM(t.amount) AS weekly_revenue
    FROM main.coffee_analytics_gold.transactions_enriched t
    CROSS JOIN intervention_meta i
    WHERE t.txn_date BETWEEN i.trend_pre_start AND i.trend_post_end
      AND t.merchant_id IN (SELECT merchant_id FROM all_merchant_ids)
    GROUP BY t.merchant_id, DATE_TRUNC('week', t.txn_date)
  ),
  treated_agg AS (
    SELECT week_start, AVG(weekly_revenue) AS treated_avg_revenue
    FROM weekly_revenue_full
    WHERE merchant_id IN (SELECT merchant_id FROM treated_merchants)
    GROUP BY week_start
  ),
  control_weights AS (
    SELECT control_merchant_id AS merchant_id, AVG(similarity_score) AS weight
    FROM top_matches
    GROUP BY control_merchant_id
  ),
  control_agg AS (
    SELECT
      w.week_start,
      SUM(w.weekly_revenue * cw.weight) / SUM(cw.weight) AS control_avg_revenue
    FROM weekly_revenue_full w
    JOIN control_weights cw ON w.merchant_id = cw.merchant_id
    GROUP BY w.week_start
  ),
  trends AS (
    SELECT
      CAST(FLOOR(DATEDIFF(ta.week_start, i.start_date) / 7.0) AS INT) AS week_number,
      CASE WHEN ta.week_start < i.start_date THEN 'pre' ELSE 'post' END AS period,
      ta.treated_avg_revenue,
      ca.control_avg_revenue
    FROM treated_agg ta
    JOIN control_agg ca ON ta.week_start = ca.week_start
    CROSS JOIN intervention_meta i
  ),
  baseline AS (
    SELECT AVG(treated_avg_revenue / NULLIF(control_avg_revenue, 0)) AS baseline_ratio
    FROM trends WHERE period = 'pre'
  ),
  with_lift AS (
    SELECT
      t.week_number,
      t.period,
      ((t.treated_avg_revenue / NULLIF(t.control_avg_revenue, 0)) / NULLIF(b.baseline_ratio, 0) - 1) * 100 AS lift_pct
    FROM trends t CROSS JOIN baseline b
  ),
  post_lift AS (
    SELECT
      week_number,
      lift_pct,
      ROW_NUMBER() OVER (ORDER BY week_number ASC) AS rn
    FROM with_lift
    WHERE period = 'post'
  ),
  filtered_post AS (
    SELECT week_number, lift_pct
    FROM post_lift
    WHERE CASE
        WHEN estimate_lift.first_n_weeks IS NOT NULL
            THEN rn <= estimate_lift.first_n_weeks
        ELSE (estimate_lift.week_start IS NULL OR week_number >= estimate_lift.week_start)
         AND (estimate_lift.week_end   IS NULL OR week_number <= estimate_lift.week_end)
    END
  ),
  lift_stats AS (
    SELECT
      AVG(lift_pct)         AS mean_lift,
      STDDEV_SAMP(lift_pct) AS std_lift,
      COUNT(*)              AS n_weeks
    FROM filtered_post
  ),
  t_critical_lookup AS (
    SELECT  1 AS df, 12.706 AS t_val UNION ALL SELECT  2,  4.303 UNION ALL
    SELECT  3,  3.182 UNION ALL SELECT  4,  2.776 UNION ALL SELECT  5,  2.571 UNION ALL
    SELECT  6,  2.447 UNION ALL SELECT  7,  2.365 UNION ALL SELECT  8,  2.306 UNION ALL
    SELECT  9,  2.262 UNION ALL SELECT 10,  2.228 UNION ALL SELECT 11,  2.201 UNION ALL
    SELECT 12,  2.179 UNION ALL SELECT 13,  2.160 UNION ALL SELECT 14,  2.145 UNION ALL
    SELECT 15,  2.131 UNION ALL SELECT 16,  2.120 UNION ALL SELECT 17,  2.110 UNION ALL
    SELECT 18,  2.101 UNION ALL SELECT 19,  2.093 UNION ALL SELECT 20,  2.086 UNION ALL
    SELECT 21,  2.080 UNION ALL SELECT 22,  2.074 UNION ALL SELECT 23,  2.069 UNION ALL
    SELECT 24,  2.064 UNION ALL SELECT 25,  2.060 UNION ALL SELECT 26,  2.056 UNION ALL
    SELECT 27,  2.052 UNION ALL SELECT 28,  2.048 UNION ALL SELECT 29,  2.045 UNION ALL
    SELECT 30,  2.042 UNION ALL SELECT 31,  2.040 UNION ALL SELECT 32,  2.037 UNION ALL
    SELECT 33,  2.035 UNION ALL SELECT 34,  2.032 UNION ALL SELECT 35,  2.030 UNION ALL
    SELECT 36,  2.028 UNION ALL SELECT 37,  2.026 UNION ALL SELECT 38,  2.024 UNION ALL
    SELECT 39,  2.023 UNION ALL SELECT 40,  2.021 UNION ALL SELECT 41,  2.020 UNION ALL
    SELECT 42,  2.018 UNION ALL SELECT 43,  2.017 UNION ALL SELECT 44,  2.015 UNION ALL
    SELECT 45,  2.014 UNION ALL SELECT 46,  2.013 UNION ALL SELECT 47,  2.012 UNION ALL
    SELECT 48,  2.011 UNION ALL SELECT 49,  2.010 UNION ALL SELECT 50,  2.009 UNION ALL
    SELECT 51,  2.008 UNION ALL SELECT 52,  2.007 UNION ALL SELECT 53,  2.006 UNION ALL
    SELECT 54,  2.005 UNION ALL SELECT 55,  2.004 UNION ALL SELECT 56,  2.003 UNION ALL
    SELECT 57,  2.002 UNION ALL SELECT 58,  2.002 UNION ALL SELECT 59,  2.001 UNION ALL
    SELECT 60,  2.000 UNION ALL SELECT 61,  1.960
  ),
  t_selected AS (
    SELECT tcl.t_val
    FROM lift_stats ls
    JOIN t_critical_lookup tcl
      ON tcl.df = CASE WHEN CAST(ls.n_weeks AS INT) - 1 <= 60
                       THEN CAST(ls.n_weeks AS INT) - 1
                       ELSE 61 END
  )
  SELECT
    ROUND(ls.mean_lift, 4)                                              AS lift_pct,
    ROUND(ls.mean_lift - ts.t_val * ls.std_lift / SQRT(ls.n_weeks), 4) AS ci_lower,
    ROUND(ls.mean_lift + ts.t_val * ls.std_lift / SQRT(ls.n_weeks), 4) AS ci_upper,
    CAST(ls.n_weeks AS INT)                                             AS n_weeks,
    CASE WHEN (ls.mean_lift - ts.t_val * ls.std_lift / SQRT(ls.n_weeks)) > 0
              OR (ls.mean_lift + ts.t_val * ls.std_lift / SQRT(ls.n_weeks)) < 0
         THEN TRUE ELSE FALSE END                                       AS significant
  FROM lift_stats ls CROSS JOIN t_selected ts
""")
print("✓ registered: estimate_lift")

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.estimate_lift('INT_001', 10, NULL, NULL, NULL)"))

# COMMAND ----------

spark.sql("""
GRANT USE CATALOG ON CATALOG main TO `account users`;
GRANT USE SCHEMA ON SCHEMA main.coffee_analytics_gold TO `account users`;
GRANT EXECUTE ON SCHEMA main.coffee_analytics_gold TO `account users`;
""")
print("✓ grants applied")
