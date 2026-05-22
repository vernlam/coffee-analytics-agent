# Databricks notebook source
# MAGIC %md
<<<<<<< Updated upstream
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
=======
# MAGIC # 01 — Ingest synthetic data into Unity Catalog
# MAGIC
# MAGIC Reads the three parquet files produced by `data_generation/generate_data.py`
# MAGIC from a Unity Catalog Volume and writes them as Delta tables.
# MAGIC
# MAGIC Also creates `interventions_agent_view`, which excludes the `true_effect`
# MAGIC column. The agent's tools query this view; the eval harness queries the
# MAGIC raw `interventions` table. Keeping the oracle isolated is critical for the
# MAGIC evaluation to be honest.
# MAGIC
# MAGIC **Re-runnable:** all writes are `overwrite` and view creation uses
# MAGIC `CREATE OR REPLACE`. Safe to run repeatedly.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

CATALOG = "main"               # change if your workspace uses a different catalog
SCHEMA  = "coffee_analytics"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw"   # where the parquet files live

print(f"Target: {CATALOG}.{SCHEMA}")
print(f"Source: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the schema (if needed)
>>>>>>> Stashed changes

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
<<<<<<< Updated upstream
=======
print(f"schema {CATALOG}.{SCHEMA} ready")
>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
<<<<<<< Updated upstream
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
=======
# MAGIC ## Parquet -> Delta tables

# COMMAND ----------

for table in ["merchants", "transactions", "interventions"]:
    df = spark.read.parquet(f"{VOLUME_PATH}/{table}.parquet")
    (df.write
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOG}.{SCHEMA}.{table}"))
    print(f"wrote {CATALOG}.{SCHEMA}.{table}: {df.count():,} rows")
>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
<<<<<<< Updated upstream
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
=======
# MAGIC ## Create the agent-visible view (oracle hidden)
# MAGIC
# MAGIC The `true_effect` column in `interventions` is the ground truth used to
# MAGIC score the agent. The agent's tools must never see it. They query this
# MAGIC view instead.
>>>>>>> Stashed changes

# COMMAND ----------

spark.sql(f"""
<<<<<<< Updated upstream
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
=======
CREATE OR REPLACE VIEW {CATALOG}.{SCHEMA}.interventions_agent_view AS
SELECT intervention_id,
       name,
       treated_merchant_ids,
       start_date,
       pre_window_days,
       post_window_days
FROM {CATALOG}.{SCHEMA}.interventions
""")
print(f"created {CATALOG}.{SCHEMA}.interventions_agent_view (true_effect excluded)")
>>>>>>> Stashed changes

# COMMAND ----------

# MAGIC %md
<<<<<<< Updated upstream
# MAGIC ## Validation

# COMMAND ----------

for tbl in ["merchants", "transactions", "interventions", "interventions_agent_view"]:
    n = spark.table(f"{CATALOG}.{SCHEMA}.{tbl}").count()
    print(f"  {tbl}: {n:,} rows")

view_cols = spark.table(f"{CATALOG}.{SCHEMA}.interventions_agent_view").columns
assert "true_effect" not in view_cols, "FAIL: true_effect is visible in agent view"
print("\n✓ oracle isolation confirmed: true_effect absent from interventions_agent_view")
=======
# MAGIC ## Smoke tests
# MAGIC
# MAGIC Expected:
# MAGIC - merchants: 500 rows
# MAGIC - transactions: 365,500 rows
# MAGIC - interventions_agent_view: 1 row, NO `true_effect` column

# COMMAND ----------

display(spark.sql(f"SELECT COUNT(*) AS n_merchants    FROM {CATALOG}.{SCHEMA}.merchants"))

# COMMAND ----------

display(spark.sql(f"SELECT COUNT(*) AS n_transactions FROM {CATALOG}.{SCHEMA}.transactions"))

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {CATALOG}.{SCHEMA}.interventions_agent_view"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confirm the oracle is hidden
# MAGIC
# MAGIC Compare the schemas: the raw table has `true_effect`, the view does not.

# COMMAND ----------

print("RAW interventions table schema:")
display(spark.sql(f"DESCRIBE {CATALOG}.{SCHEMA}.interventions"))

# COMMAND ----------

print("AGENT VIEW schema (should NOT contain true_effect):")
display(spark.sql(f"DESCRIBE {CATALOG}.{SCHEMA}.interventions_agent_view"))
>>>>>>> Stashed changes
