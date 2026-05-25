import json
import re
from typing import Optional

from databricks.sdk import WorkspaceClient
from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from .state import AgentState
from .tools import (
    _to_markdown,
    call_build_control_group,
    call_check_parallel_trends,
    call_estimate_lift,
    call_query_metric,
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
        if hasattr(m, "type"):  # LangChain message object
            role = _ROLE_MAP.get(m.type, "user")
            content = _extract_text(m.content)
        else:  # plain dict
            role = m.get("role", "user")
            content = _extract_text(m.get("content", ""))
        result.append({"role": role, "content": content})
    return result


_ROUTER_SYSTEM = """You are a router for a coffee shop analytics agent.

Classify the user's question and extract parameters. Respond with valid JSON only — no markdown, no explanation.

Route as "descriptive" for: totals, averages, trends, breakdowns, what happened.
Route as "causal" for: did an intervention work, what was the lift, did a pilot increase revenue.

For descriptive, extract:
- metric: one of revenue, transaction_count, avg_basket, active_merchants
- start_date / end_date: ISO format YYYY-MM-DD
- group_by: one of location_type, region, size_band, brand (or null)
- filter_location_type: urban, suburban, highway, mall, campus (or null)
- filter_region: northeast, southeast, midwest, west (or null)
- filter_size_band: small, mid, large (or null)
- filter_brand: BrandA, BrandB, BrandC, BrandD (or null)

For causal, extract:
- intervention_id: format INT_001, INT_002, etc. (null if not provided)

JSON schema:
{
  "route": "descriptive" | "causal",
  "intervention_id": string | null,
  "metric": string | null,
  "start_date": string | null,
  "end_date": string | null,
  "group_by": string | null,
  "filter_location_type": string | null,
  "filter_region": string | null,
  "filter_size_band": string | null,
  "filter_brand": string | null
}"""


def _call_llm(messages: list[dict]) -> dict:
    w = WorkspaceClient()
    response = w.api_client.do(
        "POST",
        "/serving-endpoints/databricks-meta-llama-3-3-70b-instruct/invocations",
        body={"messages": messages, "temperature": 0, "max_tokens": 500},
    )
    content = response["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Router returned non-JSON: {content}")


def router_node(state: AgentState) -> dict:
    messages = [{"role": "system", "content": _ROUTER_SYSTEM}] + _to_api_messages(state["messages"])
    result = _call_llm(messages)

    updates = {
        "route": result.get("route"),
        "intervention_id": result.get("intervention_id"),
        "metric": result.get("metric"),
        "start_date": result.get("start_date"),
        "end_date": result.get("end_date"),
        "group_by": result.get("group_by"),
        "filter_location_type": result.get("filter_location_type"),
        "filter_region": result.get("filter_region"),
        "filter_size_band": result.get("filter_size_band"),
        "filter_brand": result.get("filter_brand"),
    }

    if result.get("route") == "causal":
        updates.update({
            "metric": None, "start_date": None, "end_date": None,
            "group_by": None, "filter_location_type": None,
            "filter_region": None, "filter_size_band": None, "filter_brand": None,
        })
    else:
        updates["intervention_id"] = None

    if result.get("route") == "causal" and not result.get("intervention_id"):
        updates["messages"] = [AIMessage(
            content="I need an intervention ID to run the causal analysis. "
                    "What's the ID? It should be in the format INT_001, INT_002, etc."
        )]

    return updates


def build_control_group_node(state: AgentState) -> dict:
    n_matches = state.get("n_matches") or 10
    df = call_build_control_group(state["intervention_id"], n_matches)
    table = _to_markdown(df)
    return {
        "control_group_result": table,
        "messages": [AIMessage(
            content=f"**Proposed control group for {state['intervention_id']}:**\n\n{table}"
        )],
    }


def check_parallel_trends_node(state: AgentState) -> dict:
    n_matches = state.get("n_matches") or 10
    df = call_check_parallel_trends(state["intervention_id"], n_matches)

    df["lift_pct"] = df["lift_pct"].astype(float)
    pre_period_lift = float(df[df["period"] == "pre"]["lift_pct"].abs().mean())
    verdict = "parallel trends holds ✓" if pre_period_lift < 1.0 else "parallel trends may be violated ⚠️"
    table = _to_markdown(df)

    return {
        "parallel_trends_result": table,
        "pre_period_lift": pre_period_lift,
        "messages": [AIMessage(
            content=f"**Parallel trends check:**\n\n{table}\n\n"
                    f"Pre-period avg absolute lift: {pre_period_lift:.2f}% — {verdict}"
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

    return {
        "lift_result": summary,
        "messages": [AIMessage(content=summary)],
    }


def query_metric_node(state: AgentState) -> dict:
    df = call_query_metric(
        metric=state.get("metric"),
        start_date=state.get("start_date"),
        end_date=state.get("end_date"),
        group_by=state.get("group_by"),
        filter_location_type=state.get("filter_location_type"),
        filter_region=state.get("filter_region"),
        filter_size_band=state.get("filter_size_band"),
        filter_brand=state.get("filter_brand"),
    )
    table = _to_markdown(df)
    return {
        "query_metric_result": table,
        "messages": [AIMessage(content=table)],
    }
