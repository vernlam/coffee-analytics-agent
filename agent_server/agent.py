import logging
from typing import AsyncGenerator

import mlflow
from agents import Agent, Runner, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks.sdk import WorkspaceClient
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.utils import (
    build_mcp_url,
    get_session_id,
    process_agent_stream_events,
)

logger = logging.getLogger(__name__)

set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api("chat_completions")
set_trace_processors([])
mlflow.openai.autolog()
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)


async def init_mcp_server(workspace_client: WorkspaceClient):
    return McpServer(
        url=build_mcp_url("/api/2.0/mcp/functions/main/coffee_analytics_gold", workspace_client=workspace_client),
        name="system.ai UC function MCP server",
        workspace_client=workspace_client,
    )


def create_agent(mcp_servers: list[McpServer] | None = None) -> Agent:
    return Agent(
        name="Analytics Agent",
        instructions="""
        You are an analytics agent for a coffee shop chain with 500 stores and two years of daily transaction data.

        ## Tools
        - query_metric — descriptive questions: totals, averages, trends, segment breakdowns
        - build_control_group — causal step 1: proposes matched control stores
        - check_parallel_trends — causal step 2: verifies pre-period parallel trends and returns weekly lift data
        - estimate_lift — causal step 3: estimates average lift % with 95% confidence interval

        ## Routing
        Descriptive (what happened, totals, breakdowns) → query_metric
        Causal (did an intervention work, what was the lift) → causal pipeline below

        ## Causal workflow
        1. Call build_control_group(intervention_id, n_matches=10)
        2. Show the user the proposed control stores as a markdown table (treated_merchant_id, control_merchant_id, similarity_score)
        3. Call check_parallel_trends(intervention_id, n_matches=10)
        4. Format the results as a markdown table with columns: week_number, treated_avg_revenue, control_avg_revenue, lift_pct, period
        5. Compute the average absolute lift_pct over pre-period weeks only
        6. Tell the user: "Pre-period avg absolute lift: X% — parallel trends holds ✓" (if < 1%) or "may be violated ⚠️" (if ≥ 1%)
        7. Ask: "Do you approve proceeding with the lift estimate?"
        8. STOP and wait for explicit approval
        9. On approval, call estimate_lift(intervention_id, n_matches=10) — omit first_n_weeks, week_start, week_end
        10. Report exact lift_pct, ci_lower, ci_upper, n_weeks, significant

        ## Rules
        - Always report exact values returned by tools. Never round, infer, or fabricate numbers.
        - For query_metric: valid group_by values are location_type, region, size_band, brand.
        - If the user asks a causal question without an intervention ID, ask them for it. IDs follow the format INT_001, INT_002, etc.

        ## Examples
        "What were sales in Q1 2024?" → query_metric
        "Did the mobile pilot increase revenue?" → causal pipeline, ask for intervention ID if not given
        "Break down revenue by region" → query_metric with group_by='region'
        """,
        model="databricks-meta-llama-3-3-70b-instruct",
        mcp_servers=mcp_servers or [],
    )


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
    try:
        async with await init_mcp_server(WorkspaceClient()) as mcp_server:
            agent = create_agent(mcp_servers=[mcp_server])
            messages = [i.model_dump() for i in request.input]
            result = await Runner.run(agent, messages)
            return ResponsesAgentResponse(output=[item.to_input_item() for item in result.new_items])
    except Exception:
        logger.warning("MCP server unavailable. Continuing without MCP tools.", exc_info=True)
        agent = create_agent()
        messages = [i.model_dump() for i in request.input]
        result = await Runner.run(agent, messages)
        return ResponsesAgentResponse(output=[item.to_input_item() for item in result.new_items])


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    if session_id := get_session_id(request):
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
    try:
        async with await init_mcp_server(WorkspaceClient()) as mcp_server:
            agent = create_agent(mcp_servers=[mcp_server])
            messages = [i.model_dump() for i in request.input]
            result = Runner.run_streamed(agent, input=messages)
            async for event in process_agent_stream_events(result.stream_events()):
                yield event
    except Exception:
        logger.warning("MCP server unavailable. Continuing without MCP tools.", exc_info=True)
        agent = create_agent()
        messages = [i.model_dump() for i in request.input]
        result = Runner.run_streamed(agent, input=messages)
        async for event in process_agent_stream_events(result.stream_events()):
            yield event
