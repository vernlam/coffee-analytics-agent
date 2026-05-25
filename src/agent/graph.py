from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    build_control_group_node,
    check_parallel_trends_node,
    estimate_lift_node,
    hitl_node,
    query_metric_node,
    router_node,
)
from .state import AgentState


def _route_after_router(state: AgentState) -> str:
    if state["route"] == "causal" and state.get("intervention_id"):
        return "build_control_group"
    elif state["route"] == "causal":
        return END  # asked user for intervention_id — wait for next message
    else:
        return "query_metric"


def _route_after_hitl(state: AgentState) -> str:
    return "estimate_lift" if state.get("approved") else END


def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("router", router_node)
    builder.add_node("build_control_group", build_control_group_node)
    builder.add_node("check_parallel_trends", check_parallel_trends_node)
    builder.add_node("hitl", hitl_node)
    builder.add_node("estimate_lift", estimate_lift_node)
    builder.add_node("query_metric", query_metric_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", _route_after_router)
    builder.add_edge("build_control_group", "check_parallel_trends")
    builder.add_edge("check_parallel_trends", "hitl")
    builder.add_conditional_edges("hitl", _route_after_hitl)
    builder.add_edge("estimate_lift", END)
    builder.add_edge("query_metric", END)

    return builder.compile(checkpointer=MemorySaver())


graph = build_graph()
