from typing import Annotated, Literal, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    # Conversation history — add_messages appends instead of overwriting
    messages: Annotated[list, add_messages]

    # Set by router
    route: Optional[Literal["descriptive", "causal"]]

    # Causal pipeline
    intervention_id: Optional[str]
    n_matches: Optional[int]
    control_group_result: Optional[str]    # markdown table
    parallel_trends_result: Optional[str]  # markdown table
    pre_period_lift: Optional[float]
    approved: Optional[bool]
    lift_result: Optional[str]             # markdown summary

    # Descriptive pipeline
    metric: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    group_by: Optional[str]
    filter_location_type: Optional[str]
    filter_region: Optional[str]
    filter_size_band: Optional[str]
    filter_brand: Optional[str]
    query_metric_result: Optional[str]     # markdown table
