# Coffee Analytics Agent

A conversational analytics agent for a synthetic coffee shop network. Built to learn how to design and deploy AI agents with Databricks as the data and compute backend.

---

## What it is

A Streamlit chat interface that lets you ask natural language questions about a 500-store coffee shop network — either descriptive ("what was revenue in Q1 2024?") or causal ("did the mobile order pilot work?"). The agent routes the question, queries Databricks, and responds in plain English.

The data is synthetic but realistic: two years of daily merchant transactions with a single controlled intervention (a mobile ordering pilot) with a known ground-truth lift, used for evaluation.

---

## Architecture

```
User (Streamlit chat)
        │
        ▼
  router_node  ←── Claude Haiku (classify: descriptive or causal)
        │
   ┌────┴──────────┐
   ▼               ▼
text_to_sql  build_control_group
   │               │
   │          check_parallel_trends
   │               │
   │           hitl_node  ←── interrupt: user approves or rejects
   │               │
   │          estimate_lift
   │               │
   └────┬──────────┘
        ▼
  Databricks SQL Warehouse
  (transactions_enriched, gold layer functions)
```

**Descriptive path:** The LLM receives the table schema and writes a SQL `SELECT` statement. That SQL runs on a Databricks SQL warehouse, and a second LLM call interprets the result in plain English.

**Causal path:** A three-step matching pipeline inspired by Rosenbaum & Rubin (1983) and the difference-in-differences framework (Card & Krueger, 1994):
1. **Build control group** — nearest-neighbour matching on normalised pre-period weekly revenue
2. **Check parallel trends** — validates that treated and control stores moved together before the intervention
3. **Estimate lift** — post-period DiD estimate with confidence interval

A human-in-the-loop (HITL) interrupt sits between steps 2 and 3. The user reviews the parallel trends chart and approves before the lift estimate runs.

---

## Data pipeline

The synthetic data is generated locally and loaded into Databricks via a medallion architecture:

| Layer | Schema | Contents |
|-------|--------|----------|
| Bronze | `coffee_analytics` | Raw parquet files ingested as Delta tables |
| Silver | `coffee_analytics_silver` | Typed and cleaned merchants + transactions |
| Gold | `coffee_analytics_gold` | `transactions_enriched` (denormalised join) + registered SQL functions |

The gold layer SQL functions (`build_control_group`, `check_parallel_trends`, `estimate_lift`) encode the causal methodology and are what the agent calls at runtime.

---

## Tech stack

| Component | Tool |
|-----------|------|
| Agent framework | LangGraph |
| LLM (routing + SQL + interpretation) | Claude Haiku via Anthropic API |
| Data + compute | Databricks (Unity Catalog, SQL Warehouse) |
| UI | Streamlit |
| Data format | Delta Lake |

---

## Running locally

1. Clone the repo
2. Create a `.env` file in the root:
```
DATABRICKS_HOST=https://your-workspace.azuredatabricks.net
DATABRICKS_TOKEN=dapiXXXXXXXXXXXXXXXX
DATABRICKS_WAREHOUSE_ID=your_warehouse_id
ANTHROPIC_API_KEY=sk-ant-XXXXXXXX
```
3. Run:
```bash
uv run streamlit run streamlit_app.py
```

---

## Further revisions

**SQL validation before execution**
The text-to-SQL step currently sends whatever the LLM generates straight to Databricks. A validation step that checks the SQL only references allowed tables and columns would make it more robust and prevent unexpected queries from running.

**HITL for text-to-SQL**
Show the user the generated SQL before running it and let them approve or edit. This mirrors the causal HITL pattern and gives users more trust and transparency into what the agent is actually querying.

**Evaluation harness**
The synthetic data includes a `true_effect` oracle column — the ground truth lift for INT_001. An evaluation harness could run the full causal pipeline and compare the agent's lift estimate against the known answer to measure methodological accuracy.

**Multiple interventions**
Currently only one intervention (INT_001) exists in the data. Generating additional interventions with varying effect sizes and store characteristics would make the causal pipeline more interesting to demo and stress-test.

**Streamlit Cloud deployment**
The app currently runs locally only. Deploying to Streamlit Cloud would make it publicly accessible from a URL with no local setup required — better for sharing and portfolio presentation.

**Query caching**
Repeated identical questions hit the Databricks warehouse every time. A simple cache layer would reduce cost and latency for common queries.

---

## What I learnt

**LangGraph**
Building a deterministic agent with a state machine rather than an open-ended ReAct loop. The key concepts were: `TypedDict` state, nodes as pure functions that return state updates, conditional edges for branching, and `interrupt()` for pausing execution mid-graph to wait for human input. The `MemorySaver` checkpointer persists the graph state between Streamlit re-renders — this is what makes both HITL and conversation history work. State accumulates across turns so the agent remembers what was asked before, which is essential for descriptive analytics where users naturally follow up on previous results.

**Text-to-SQL**
How to implement text-to-SQL using two LLM calls. The first call receives the table schema in the system prompt and the user's question, and writes a SQL query. That SQL is sent to the Databricks warehouse via the REST API, which executes it and returns the result as JSON. The second LLM call receives the original question and the query result, and interprets it into a plain English response for the user. Feeding the schema in the system prompt is what prevents the LLM from hallucinating table or column names — it can only reference what it's been told exists. Conversation history is passed to the SQL generation step so follow-up questions like "how about just January?" or "break that down by region" resolve correctly without the user having to repeat context.

**Databricks**
How the medallion architecture (bronze → silver → gold) separates ingestion, transformation, and serving concerns. How to ingest raw data into Databricks and promote it through layers using Delta tables. What Unity Catalog is and how it provides governance — a single place to manage tables, schemas, and functions with access controls. How SQL warehouses provide serverless compute that scales on demand. How Unity Catalog functions encapsulate analytics logic (matching, parallel trends, lift estimation) so the application layer never needs to know the implementation — it just calls a function. How to connect a local Streamlit application to Databricks via the SQL Statements REST API, authenticating with a PAT token and sending SQL as JSON.

**Agent design**
The current agent is deliberately simple — the router makes a single binary decision: descriptive or causal. That's it. There's no complex tool selection or multi-step reasoning at the routing level. For descriptive questions, the LLM generates SQL freely against the schema. For causal questions, the pipeline is hardcoded because the methodology has a specific sequence of steps (control group → parallel trends → HITL → lift estimate) that shouldn't be improvised by an LLM. The rigidity is intentional — causal inference requires methodological discipline that open-ended reasoning can't guarantee.

**Human-in-the-loop**
How to implement a HITL layer using LangGraph's `interrupt()`. In causal analysis, users need to review whether the matched control group is a reasonable counterfactual before trusting the lift estimate — this is a judgement call that shouldn't be automated. The agent pauses after the parallel trends check, shows the user a chart of test versus control group trends over time, and waits for explicit approval before proceeding to the lift estimate. If the trends don't look parallel in the pre-period, the user can reject and the pipeline stops.

![App screenshot](/assets/HITL.png)
