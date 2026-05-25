from typing import Annotated, Literal, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

    route: Optional[Literal["descriptive", "causal"]]

    # Causal pipeline
    intervention_id: Optional[str]
    n_matches: Optional[int]
    control_group_df_json: Optional[str]
    parallel_trends_df_json: Optional[str]
    pre_period_lift: Optional[float]
    approved: Optional[bool]
    lift_result: Optional[str]
