# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Silver Layer
# MAGIC
# MAGIC Transforms bronze tables into silver:
# MAGIC - **transactions**: cast txn_date to DateType, add calendar features
# MAGIC - **merchants**: pass-through with explicit type enforcement
# MAGIC
# MAGIC Reads from `main.coffee_analytics` (bronze). Writes to `main.coffee_analytics_silver`.
# MAGIC Idempotent: safe to re-run.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType, IntegerType, StringType

BRONZE_CATALOG = "main"
BRONZE_SCHEMA  = "coffee_analytics"
SILVER_CATALOG = "main"
SILVER_SCHEMA  = "coffee_analytics_silver"

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SILVER_CATALOG}.{SILVER_SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merchants

# COMMAND ----------

merchants = (
    spark.table(f"{BRONZE_CATALOG}.{BRONZE_SCHEMA}.merchants")
    .select(
        F.col("merchant_id").cast(StringType()),
        F.col("location_type").cast(StringType()),
        F.col("region").cast(StringType()),
        F.col("size_band").cast(StringType()),
        F.col("brand").cast(StringType()),
        F.col("onboarded_date").cast(DateType()),
    )
)
print(f"merchants: {merchants.count():,} rows")
merchants.printSchema()

(
    merchants.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{SILVER_CATALOG}.{SILVER_SCHEMA}.merchants")
)
print("✓ written: merchants")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transactions

# COMMAND ----------

# day_of_week: ISO 8601 (1=Mon, 7=Sun) — matches the generator's weekday() convention where >= 5 is weekend
transactions = (
    spark.table(f"{BRONZE_CATALOG}.{BRONZE_SCHEMA}.transactions")
    .select(
        F.col("merchant_id").cast(StringType()),
        F.col("txn_date").cast(DateType()),
        F.col("amount").cast(DoubleType()),
        F.col("txn_count").cast(IntegerType()),
    )
    .withColumn("year",        F.year("txn_date"))
    .withColumn("month",       F.month("txn_date"))
    .withColumn("day_of_week", F.date_format("txn_date", "u").cast(IntegerType()))
    .withColumn("is_weekend",  F.col("day_of_week") >= 6)
)
print(f"transactions: {transactions.count():,} rows")
transactions.printSchema()

(
    transactions.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{SILVER_CATALOG}.{SILVER_SCHEMA}.transactions")
)
print("✓ written: transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

for tbl in ["merchants", "transactions"]:
    n = spark.table(f"{SILVER_CATALOG}.{SILVER_SCHEMA}.{tbl}").count()
    print(f"  {tbl}: {n:,} rows")

sample = spark.table(f"{SILVER_CATALOG}.{SILVER_SCHEMA}.transactions").limit(5)
sample.select("txn_date", "year", "month", "day_of_week", "is_weekend").show()
print("✓ silver layer complete")
