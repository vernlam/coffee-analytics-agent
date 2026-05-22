# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Bronze Layer Ingest
# MAGIC
# MAGIC Reads the three source Parquet files from the Unity Catalog Volume and
# MAGIC writes them as Delta tables.
# MAGIC
# MAGIC Also creates `interventions_agent_view`, which excludes the `true_effect`
# MAGIC oracle column. Every agent tool queries this view. The raw `interventions`
# MAGIC table (oracle included) is used only by the evaluation harness.
# MAGIC
# MAGIC Idempotent: safe to re-run; tables are replaced on each execution.

# COMMAND ----------

CATALOG     = "main"
SCHEMA      = "coffee_analytics"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw"

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merchants

# COMMAND ----------

merchants = spark.read.parquet(f"{VOLUME_PATH}/merchants.parquet")
print(f"merchants: {merchants.count():,} rows")
merchants.printSchema()

(
    merchants.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.merchants")
)
print("✓ written: merchants")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transactions

# COMMAND ----------

transactions = spark.read.parquet(f"{VOLUME_PATH}/transactions.parquet")
print(f"transactions: {transactions.count():,} rows")
transactions.printSchema()

(
    transactions.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.transactions")
)
print("✓ written: transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interventions (full table — oracle included)

# COMMAND ----------

interventions = spark.read.parquet(f"{VOLUME_PATH}/interventions.parquet")
print(f"interventions: {interventions.count():,} rows")
interventions.printSchema()

(
    interventions.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.{SCHEMA}.interventions")
)
print("✓ written: interventions (true_effect oracle present — do not expose to agent)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent view — isolate the oracle
# MAGIC
# MAGIC Explicit column list (not SELECT *) so a schema change can never
# MAGIC accidentally surface `true_effect` through the view.

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE VIEW {CATALOG}.{SCHEMA}.interventions_agent_view AS
    SELECT
        intervention_id,
        name,
        treated_merchant_ids,
        start_date,
        pre_window_days,
        post_window_days
    FROM {CATALOG}.{SCHEMA}.interventions
""")
print("✓ created: interventions_agent_view (true_effect excluded)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation

# COMMAND ----------

for tbl in ["merchants", "transactions", "interventions", "interventions_agent_view"]:
    n = spark.table(f"{CATALOG}.{SCHEMA}.{tbl}").count()
    print(f"  {tbl}: {n:,} rows")

view_cols = spark.table(f"{CATALOG}.{SCHEMA}.interventions_agent_view").columns
assert "true_effect" not in view_cols, "FAIL: true_effect is visible in agent view"
print("\n✓ oracle isolation confirmed: true_effect absent from interventions_agent_view")
