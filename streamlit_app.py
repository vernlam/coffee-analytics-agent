import io
import uuid

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

from src.agent import graph  # noqa: E402


def _run_graph(user_input: str, config: dict) -> tuple[list[str], str | None, dict]:
    current_state = graph.get_state(config)
    prev_count = len(current_state.values.get("messages", [])) if current_state.values else 0

    if current_state.next and current_state.tasks and current_state.tasks[0].interrupts:
        result = graph.invoke(Command(resume=user_input), config=config)
    else:
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

    new_texts = [
        m.content for m in result.get("messages", [])[prev_count:]
        if isinstance(m, AIMessage)
    ]

    interrupt_text = None
    new_state = graph.get_state(config)
    if new_state.next and new_state.tasks and new_state.tasks[0].interrupts:
        interrupt_text = str(new_state.tasks[0].interrupts[0].value)

    extra_data = {
        "control_group_df_json": new_state.values.get("control_group_df_json"),
        "parallel_trends_df_json": new_state.values.get("parallel_trends_df_json"),
    }

    return new_texts, interrupt_text, extra_data


def _render_parallel_trends_chart(df: pd.DataFrame):
    df = df.copy()
    df["treated_avg_revenue"] = df["treated_avg_revenue"].astype(float)
    df["control_avg_revenue"] = df["control_avg_revenue"].astype(float)
    df["week_number"] = df["week_number"].astype(int)

    pre = df[df["period"] == "pre"]
    treated_base = pre["treated_avg_revenue"].mean()
    control_base = pre["control_avg_revenue"].mean()

    df["treated_index"] = df["treated_avg_revenue"] / treated_base
    df["control_index"] = df["control_avg_revenue"] / control_base

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["week_number"], y=df["treated_index"],
        name="Test group", line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["week_number"], y=df["control_index"],
        name="Control group", line=dict(color="#ff7f0e", width=2, dash="dash"),
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color="lightgray")
    fig.add_vline(x=0, line_dash="dot", line_color="gray", annotation_text="Intervention start")
    fig.update_layout(
        title="Weekly Revenue Index: Test vs Control (pre-period = 1.0)",
        xaxis_title="Week (0 = intervention start)",
        yaxis_title="Revenue index (pre-period avg = 1.0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def _render_message(msg: dict, idx: int):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("control_group_df_json"):
            df = pd.read_json(io.StringIO(msg["control_group_df_json"]))
            st.download_button(
                "Download control group (Excel)",
                _excel_bytes(df),
                file_name="control_group.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"cg_{idx}",
            )

        if msg.get("parallel_trends_df_json"):
            df = pd.read_json(io.StringIO(msg["parallel_trends_df_json"]))
            _render_parallel_trends_chart(df)
            st.download_button(
                "Download parallel trends data (Excel)",
                _excel_bytes(df),
                file_name="parallel_trends.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"pt_{idx}",
            )


st.set_page_config(page_title="Coffee Analytics", page_icon="☕", layout="centered")

col1, col2 = st.columns([5, 1])
with col1:
    st.title("☕ Coffee Analytics Agent")
with col2:
    st.write("")
    if st.button("New chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.display_messages = []
        st.rerun()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

config = {"configurable": {"thread_id": st.session_state.thread_id}}

for idx, msg in enumerate(st.session_state.display_messages):
    _render_message(msg, idx)

if prompt := st.chat_input("Ask about coffee shop performance..."):
    user_msg = {"role": "user", "content": prompt}
    st.session_state.display_messages.append(user_msg)
    _render_message(user_msg, len(st.session_state.display_messages) - 1)

    with st.chat_message("assistant"):
        with st.spinner(""):
            try:
                new_texts, interrupt_text, extra_data = _run_graph(prompt, config)
            except Exception as e:
                new_texts, interrupt_text, extra_data = [], str(e), {}

        parts = [t for t in new_texts if t]
        if interrupt_text:
            parts.append(interrupt_text)

        response = "\n\n".join(parts) or "I couldn't process your request."
        st.markdown(response)

        assistant_msg = {"role": "assistant", "content": response}

        if extra_data.get("control_group_df_json"):
            df = pd.read_json(io.StringIO(extra_data["control_group_df_json"]))
            idx = len(st.session_state.display_messages)
            st.download_button(
                "Download control group (Excel)",
                _excel_bytes(df),
                file_name="control_group.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"cg_{idx}",
            )
            assistant_msg["control_group_df_json"] = extra_data["control_group_df_json"]

        if extra_data.get("parallel_trends_df_json"):
            df = pd.read_json(io.StringIO(extra_data["parallel_trends_df_json"]))
            idx = len(st.session_state.display_messages)
            _render_parallel_trends_chart(df)
            st.download_button(
                "Download parallel trends data (Excel)",
                _excel_bytes(df),
                file_name="parallel_trends.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"pt_{idx}",
            )
            assistant_msg["parallel_trends_df_json"] = extra_data["parallel_trends_df_json"]

        st.session_state.display_messages.append(assistant_msg)
