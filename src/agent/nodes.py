import json
import os
import re

import anthropic
from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from .state import AgentState
from .tools import (
    _execute_sql,
    _to_markdown,
    call_build_control_group,
    call_check_parallel_trends,
    call_estimate_lift,
    call_lookup_intervention,
)

_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _to_api_messages(state_messages) -> list[dict]:
    result = []
    for m in state_messages:
        if hasattr(m, "type"):
            role = _ROLE_MAP.get(m.type, "user")
            content = _extract_text(m.content)
        else:
            role = m.get("role", "user")
            content = _extract_text(m.get("content", ""))
        result.append({"role": role, "content": content})
    return result


def _llm(system: str, messages: list[dict], max_tokens: int = 500) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system=system,
        messages=[m for m in messages if m["role"] != "system"],
    )
    return response.content[0].text


_ROUTER_SYSTEM = """You are a router for a coffee shop analytics agent.

Classify the user's question. Respond with valid JSON only — no markdown, no explanation.

Route as "descriptive" for: totals, averages, trends, breakdowns, what happened.
Route as "causal" for: did an intervention work, what was the lift, did a pilot increase revenue.

For causal, also extract:
- intervention_id: format INT_001, INT_002, etc. (null if not provided)
- intervention_name: plain-language name the user used (null if they gave an ID directly)

JSON schema:
{
  "route": "descriptive" | "causal",
  "intervention_id": string | null,
  "intervention_name": string | null
}"""

_SCHEMA = """You have access to one table in Databricks SQL:

Table: main.coffee_analytics_gold.transactions_enriched
Grain: one row per merchant per day
Columns:
  merchant_id      string   — unique store identifier
  txn_date         date     — transaction date
  amount           double   — daily revenue in dollars
  txn_count        int      — number of transactions
  year             int
  month            int      (1–12)
  day_of_week      int      (0=Monday, 6=Sunday)
  is_weekend       boolean
  location_type    string   — one of: urban, suburban, highway, mall, campus
  region           string   — one of: northeast, southeast, midwest, west
  size_band        string   — one of: small, mid, large
  brand            string   — one of: BrandA, BrandB, BrandC, BrandD
  onboarded_date   date

Write a single SQL SELECT statement to answer the user's question.
Return SQL only — no markdown, no explanation, no code fences."""


def router_node(state: AgentState) -> dict:
    messages = [{"role": "system", "content": _ROUTER_SYSTEM}] + _to_api_messages(state["messages"])
    raw = _llm(_ROUTER_SYSTEM, messages, max_tokens=200)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
        result = json.loads(match.group(1)) if match else {"route": "descriptive"}

    intervention_id = result.get("intervention_id")
    if not intervention_id and result.get("intervention_name"):
        intervention_id = call_lookup_intervention(result["intervention_name"])

    updates = {
        "route": result.get("route"),
        "intervention_id": intervention_id if result.get("route") == "causal" else None,
    }

    if result.get("route") == "causal" and not intervention_id:
        updates["messages"] = [AIMessage(
            content="I need an intervention ID or name to run the causal analysis. "
                    "Try something like 'did the mobile order pilot work' or 'did INT_001 work'."
        )]

    return updates


def text_to_sql_node(state: AgentState) -> dict:
    conversation = _to_api_messages(state["messages"])

    sql = _llm(_SCHEMA, conversation, max_tokens=500).strip()

    try:
        df = _execute_sql(sql)
    except Exception as e:
        return {"messages": [AIMessage(content=f"I couldn't run that query: {e}")]}

    table = _to_markdown(df)

    last_user_msg = next(
        (_extract_text(m.content) for m in reversed(state["messages"])
         if hasattr(m, "type") and m.type == "human"),
        "the user's question"
    )

    answer = _llm(
        "You are a data analyst. Answer the user's question in 1-2 natural sentences based on the query results. Be concise and use plain English.",
        [{"role": "user", "content": f"Question: {last_user_msg}\n\nResults:\n{table}"}],
        max_tokens=200,
    )

    return {"messages": [AIMessage(content=f"{answer}\n\n```sql\n{sql}\n```")]}


def build_control_group_node(state: AgentState) -> dict:
    n_matches = state.get("n_matches") or 10
    df = call_build_control_group(state["intervention_id"], n_matches)
    return {
        "control_group_df_json": df.to_json(orient="records"),
        "messages": [AIMessage(
            content=f"**Control group for {state['intervention_id']}** — {len(df)} treated-control pairs. Download below."
        )],
    }


def check_parallel_trends_node(state: AgentState) -> dict:
    n_matches = state.get("n_matches") or 10
    df = call_check_parallel_trends(state["intervention_id"], n_matches)

    df["lift_pct"] = df["lift_pct"].astype(float)
    pre_period_lift = float(df[df["period"] == "pre"]["lift_pct"].abs().mean())
    verdict = "parallel trends holds ✓" if pre_period_lift < 1.0 else "parallel trends may be violated ⚠️"

    return {
        "parallel_trends_df_json": df.to_json(orient="records"),
        "pre_period_lift": pre_period_lift,
        "messages": [AIMessage(
            content=f"**Parallel trends check:** pre-period avg absolute lift: {pre_period_lift:.2f}% — {verdict}. "
                    f"See chart and download below."
        )],
    }


def hitl_node(state: AgentState) -> dict:
    pre_lift = state.get("pre_period_lift") or 0.0
    verdict = "parallel trends holds ✓" if pre_lift < 1.0 else "may be violated ⚠️"

    decision = interrupt(
        f"Pre-period avg absolute lift: {pre_lift:.2f}% — {verdict}\n\n"
        "Do you approve proceeding with the lift estimate? (yes / no)"
    )

    approved = str(decision).lower().strip() in ("yes", "y", "approve", "approved", "proceed")
    return {"approved": approved}


def estimate_lift_node(state: AgentState) -> dict:
    n_matches = state.get("n_matches") or 10
    df = call_estimate_lift(state["intervention_id"], n_matches)
    row = df.iloc[0]

    summary = (
        f"**Lift estimate for {state['intervention_id']}:**\n\n"
        f"| lift_pct | ci_lower | ci_upper | n_weeks | significant |\n"
        f"|----------|----------|----------|---------|-------------|\n"
        f"| {row['lift_pct']}% | {row['ci_lower']}% | {row['ci_upper']}% "
        f"| {row['n_weeks']} | {str(row['significant']).lower()} |"
    )

    return {"lift_result": summary, "messages": [AIMessage(content=summary)]}
