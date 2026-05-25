import asyncio
import logging
import uuid
from typing import AsyncGenerator

import mlflow
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.utils import get_session_id
from src.agent import graph

logger = logging.getLogger(__name__)
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content)


def _normalize_messages(raw_messages: list) -> list:
    """Convert Responses API messages to LangChain HumanMessage objects."""
    result = []
    for m in raw_messages:
        role = m.get("role", "user")
        content = _extract_text(m.get("content", ""))
        if content and role == "user":
            result.append(HumanMessage(content=content))
    return result


def _run_graph(messages: list, config: dict) -> tuple[list[str], str | None]:
    current_state = graph.get_state(config)
    prev_count = len(current_state.values.get("messages", [])) if current_state.values else 0

    if current_state.next and current_state.tasks and current_state.tasks[0].interrupts:
        last_user_msg = _extract_text(messages[-1].get("content", "")) if messages else ""
        result = graph.invoke(Command(resume=last_user_msg), config=config)
    else:
        normalized = _normalize_messages(messages)
        result = graph.invoke({"messages": normalized}, config=config)

    new_texts = []
    for m in result.get("messages", [])[prev_count:]:
        if isinstance(m, AIMessage):
            new_texts.append(_extract_text(m.content))

    interrupt_text = None
    new_state = graph.get_state(config)
    if new_state.next and new_state.tasks and new_state.tasks[0].interrupts:
        interrupt_text = str(new_state.tasks[0].interrupts[0].value)

    return new_texts, interrupt_text


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    session_id = get_session_id(request)
    if session_id:
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    config = {"configurable": {"thread_id": session_id or "default"}}
    messages = [i.model_dump() for i in request.input]
    new_texts, interrupt_text = await asyncio.to_thread(_run_graph, messages, config)

    parts = [t for t in new_texts if t]
    if interrupt_text:
        parts.append(interrupt_text)

    return ResponsesAgentResponse(output=[{
        "type": "message",
        "id": f"msg_{uuid.uuid4().hex}",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "\n\n".join(parts) or "I couldn't process your request."}],
    }])


@stream()
async def stream_handler(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    session_id = get_session_id(request)
    if session_id:
        mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

    config = {"configurable": {"thread_id": session_id or "default"}}
    messages = [i.model_dump() for i in request.input]
    new_texts, interrupt_text = await asyncio.to_thread(_run_graph, messages, config)

    parts = [t for t in new_texts if t]
    if interrupt_text:
        parts.append(interrupt_text)

    for part in parts:
        yield ResponsesAgentStreamEvent(
            type="response.output_text.delta",
            delta=part,
        )
