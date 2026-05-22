# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Gold Layer
# MAGIC
# MAGIC Produces `transactions_enriched`: silver transactions left-joined with silver
# MAGIC merchants. Merchant-day grain. All columns in one place so agent tools never
# MAGIC need to join at query time.
# MAGIC
# MAGIC Reads from `main.coffee_analytics_silver`. Writes to `main.coffee_analytics_gold`.
# MAGIC Idempotent: safe to re-run.

# COMMAND ----------

SILVER_CATALOG = "main"
SILVER_SCHEMA  = "coffee_analytics_silver"
GOLD_CATALOG   = "main"
GOLD_SCHEMA    = "coffee_analytics_gold"

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {GOLD_CATALOG}.{GOLD_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## transactions_enriched

# COMMAND ----------

transactions = spark.table(f"{SILVER_CATALOG}.{SILVER_SCHEMA}.transactions")
merchants    = spark.table(f"{SILVER_CATALOG}.{SILVER_SCHEMA}.merchants")

enriched = (
    transactions
    .join(merchants, on="merchant_id", how="left")
    .select(
        # keys
        "merchant_id",
        "txn_date",
        # measures
        "amount",
        "txn_count",
        # calendar
        "year",
        "month",
        "day_of_week",
        "is_weekend",
        # merchant attributes
        "location_type",
        "region",
        "size_band",
        "brand",
        "onboarded_date",
    )
)

print(f"transactions_enriched: {enriched.count():,} rows")
enriched.printSchema()

(
    enriched.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{GOLD_CATALOG}.{GOLD_SCHEMA}.transactions_enriched")
)
print("✓ written: transactions_enriched")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

n = spark.table(f"{GOLD_CATALOG}.{GOLD_SCHEMA}.transactions_enriched").count()
print(f"  transactions_enriched: {n:,} rows")

# Nulls in merchant columns would mean a broken join
from pyspark.sql import functions as F
null_check = (
    spark.table(f"{GOLD_CATALOG}.{GOLD_SCHEMA}.transactions_enriched")
    .select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in ["location_type", "region", "size_band", "brand"]])
)
null_check.show()
print("✓ gold layer complete (all counts above should be 0)")
