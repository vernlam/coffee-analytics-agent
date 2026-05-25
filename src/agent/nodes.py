from typing import Literal, Optional

import os

from databricks.sdk import WorkspaceClient
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel

from .state import AgentState
from .tools import (
    _to_markdown,
    call_build_control_group,
    call_check_parallel_trends,
    call_estimate_lift,
    call_query_metric,
)


def _make_llm() -> ChatOpenAI:
    w = WorkspaceClient()
    return ChatOpenAI(
        model="databricks-meta-llama-3-3-70b-instruct",
        openai_api_base=f"{w.config.host}/serving-endpoints",
        openai_api_key=w.config.token or "token",
    )


_llm = _make_llm()

_ROUTER_SYSTEM = """You are a router for a coffee shop analytics agent.

Classify the user's question and extract parameters.

Route as "descriptive" for: totals, averages, trends, breakdowns, what happened.
Route as "causal" for: did an intervention work, what was the lift, did a pilot increase revenue.

For descriptive, extract:
- metric: one of revenue, transaction_count, avg_basket, active_merchants
- start_date / end_date: ISO format YYYY-MM-DD
- group_by: one of location_type, region, size_band, brand (or null)
- filters: location_type (urban/suburban/highway/mall/campus), region (northeast/southeast/midwest/west),
  size_band (small/mid/large), brand (BrandA/BrandB/BrandC/BrandD) — null if not specified

For causal, extract:
- intervention_id: format INT_001, INT_002, etc. (null if not provided by user)
"""


class _RouterOutput(BaseModel):
    route: Literal["descriptive", "causal"]
    intervention_id: Optional[str] = None
    metric: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    group_by: Optional[str] = None
    filter_location_type: Optional[str] = None
    filter_region: Optional[str] = None
    filter_size_band: Optional[str] = None
    filter_brand: Optional[str] = None


_router_llm = _llm.with_structured_output(_RouterOutput)


def router_node(state: AgentState) -> dict:
    messages = [{"role": "system", "content": _ROUTER_SYSTEM}] + state["messages"]
    result: _RouterOutput = _router_llm.invoke(messages)

    updates = {
        "route": result.route,
        "intervention_id": result.intervention_id,
        "metric": result.metric,
        "start_date": result.start_date,
        "end_date": result.end_date,
        "group_by": result.group_by,
        "filter_location_type": result.filter_location_type,
        "filter_region": result.filter_region,
        "filter_size_band": result.filter_size_band,
        "filter_brand": result.filter_brand,
    }

    if result.route == "causal" and not result.intervention_id:
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

    # interrupt() pauses the graph here and returns control to the caller.
    # When the graph is resumed, decision = whatever the human typed.
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
