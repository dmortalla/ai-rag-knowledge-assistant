"""Streamlit user interface for the AI RAG Knowledge Assistant.

This module provides a polished chat-style frontend that sends user
queries to the FastAPI backend and displays the resulting grounded
answer, retrieval summary, grounding signal, and source-attributed
context.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
QUERY_API_URL = f"{API_BASE_URL}/query"
HEALTH_API_URL = f"{API_BASE_URL}/health"


def call_rag_api(query: str, api_url: str = QUERY_API_URL) -> Dict[str, Any]:
    """Send a query to the FastAPI backend.

    Args:
        query: User query text.
        api_url: Backend API endpoint URL.

    Returns:
        Parsed JSON response from the API.

    Raises:
        ValueError: If the query is empty.
        requests.RequestException: If the API request fails.
    """
    if not query or query.strip() == "":
        raise ValueError("Query cannot be empty.")

    response = requests.post(
        api_url,
        json={"query": query},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def check_backend_health(health_url: str = HEALTH_API_URL) -> Tuple[bool, str]:
    """Check whether the FastAPI backend is reachable and healthy.

    Args:
        health_url: Backend health endpoint URL.

    Returns:
        Boolean health state and a short status message.
    """
    try:
        response = requests.get(health_url, timeout=5)
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") == "ok":
            return True, "Connected"

        return False, "Unexpected health response"
    except requests.RequestException:
        return False, "Unavailable"


def highlight_query_terms(text: str, query: str) -> str:
    """Highlight query words inside retrieved text using Markdown bolding.

    Args:
        text: Retrieved text content.
        query: Original user query.

    Returns:
        Markdown-formatted text with matched query terms emphasized.
    """
    cleaned_words = re.findall(r"\b\w+\b", query.lower())
    unique_words = sorted(
        set(word for word in cleaned_words if len(word) > 2),
        key=len,
        reverse=True,
    )

    highlighted_text = text
    for word in unique_words:
        pattern = re.compile(rf"(?i)\b({re.escape(word)})\b")
        highlighted_text = pattern.sub(r"**\1**", highlighted_text)

    return highlighted_text


def compute_grounding_signal(sources: List[Dict[str, Any]]) -> int:
    """Compute a heuristic grounding signal from retrieval evidence.

    The signal summarizes the amount of retrieved context available to
    ground an answer. It is not a calibrated probability of correctness.

    Args:
        sources: Source records returned by the API.

    Returns:
        Heuristic grounding signal from 0 to 100.
    """
    if not sources:
        return 20

    if len(sources) == 1:
        return 90

    if len(sources) <= 3:
        return 75

    return 60


def initialize_session_state() -> None:
    """Initialize Streamlit session state values."""
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""

    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def render_sidebar() -> None:
    """Render the application sidebar."""
    backend_ok, backend_message = check_backend_health()

    with st.sidebar:
        st.header("About")
        st.write(
            "Ask questions over a custom knowledge base. The app retrieves "
            "relevant chunks, grounds the answer, and shows the sources used."
        )

        st.header("Endpoints")
        st.markdown("**Query API**")
        st.code(QUERY_API_URL)
        st.markdown("**Health API**")
        st.code(HEALTH_API_URL)

        st.header("System Status")
        if backend_ok:
            st.success(f"API: {backend_message}")
            st.success("RAG Pipeline: Ready")
            st.success("Vector Store: Available")
        else:
            st.error(f"API: {backend_message}")
            st.warning("RAG Pipeline: Check backend")
            st.warning("Vector Store: Unknown")

        st.header("What this app demonstrates")
        st.markdown(
            "- Retrieval-Augmented Generation\n"
            "- Vector search with FAISS\n"
            "- Grounded answer generation\n"
            "- Source attribution and explainability"
        )


def render_hero() -> None:
    """Render the page title and intro block."""
    st.title("AI RAG Knowledge Assistant")
    st.caption(
        "A production-style Retrieval-Augmented Generation application "
        "built with FastAPI, Streamlit, OpenAI, and FAISS."
    )


def render_example_queries() -> None:
    """Render quick example query buttons."""
    st.markdown("### Try an example")

    col1, col2, col3 = st.columns(3)

    if col1.button("What is customer churn?"):
        handle_submission("What is customer churn?")

    if col2.button("Why is churn important?"):
        handle_submission("Why is churn important?")

    if col3.button("What is customer retention?"):
        handle_submission("What is customer retention?")


def render_input_section() -> str:
    """Render the query input section.

    Returns:
        Current query text from the text area.
    """
    return st.text_area(
        "Ask a question about the knowledge base:",
        height=120,
        placeholder="Example: What is customer churn?",
    )


def render_conversation(active_query: str, answer: str) -> None:
    """Render the conversation section.

    Args:
        active_query: User query.
        answer: Final grounded answer.
    """
    st.divider()
    st.markdown("## Conversation")

    with st.chat_message("user"):
        st.write(active_query)

    with st.chat_message("assistant"):
        st.markdown("### Grounded Answer")
        st.caption("Answer generated using retrieved knowledge base context.")
        st.success(answer)

        with st.expander("Why this answer?"):
            st.write(
                "This answer was generated by retrieving relevant chunks from the "
                "knowledge base using vector similarity search, then grounding "
                "the response using those sources."
            )


def render_retrieval_summary(active_query: str, sources: List[Dict[str, Any]]) -> None:
    """Render the retrieval and grounding analysis section.

    Args:
        active_query: User query.
        sources: Source records returned by the API.
    """
    st.divider()
    st.markdown("## Query Context")
    st.write(f"**User asked:** {active_query}")

    st.divider()
    st.markdown("## Retrieval & Grounding Analysis")
    st.write(
        f"Top {len(sources)} relevant chunk(s) retrieved from the knowledge base."
    )

    grounding_signal = compute_grounding_signal(sources)

    st.markdown("### Grounding Signal")
    st.progress(grounding_signal)

    if grounding_signal >= 85:
        st.success(
            f"Grounding signal: {grounding_signal}% — Strong retrieval support"
    )
    elif grounding_signal >= 70:
        st.info(
            f"Grounding signal: {grounding_signal}% — Moderate retrieval support"
    )
    else:
        st.warning(
            f"Grounding signal: {grounding_signal}% — Limited retrieval support"
    )

    st.caption(
        "Heuristic signal based on retrieved context. "
        "This is not a calibrated probability that the answer is correct."
    )

    if len(sources) == 0:
        st.warning(
            "No relevant context was retrieved from the knowledge base."
        )
    elif len(sources) == 1:
        st.info(
            "Focused retrieval: one source chunk was used to ground the answer."
        )
    else:
        st.info(
            "Multi-source retrieval: multiple source chunks were used to "
            "ground the answer."
        )


def render_source_card(source_record: Dict[str, Any], index: int, query: str) -> None:
    """Render a single retrieved source card.

    Args:
        source_record: Source record returned by the API.
        index: Display index for the source.
        query: Original user query used for highlight matching.
    """
    source_name = str(source_record.get("source", "unknown_source"))
    chunk_id = source_record.get("chunk_id", "unknown_chunk")
    content = str(source_record.get("content", ""))

    header = f"📄 {source_name}  |  🔢 Chunk {chunk_id}"

    with st.expander(header, expanded=(index == 1)):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("**Source file**")
            st.code(source_name)

        with col2:
            st.markdown("**Chunk ID**")
            st.code(str(chunk_id))

        st.markdown("**Retrieved text**")
        highlighted = highlight_query_terms(content, query)
        st.markdown(highlighted)
        st.caption("Highlighted terms indicate relevance to your query.")


def render_sources_section(sources: List[Dict[str, Any]], active_query: str) -> None:
    """Render the sources section.

    Args:
        sources: Source records returned by the API.
        active_query: User query.
    """
    st.divider()
    st.markdown("## Retrieved Sources")
    st.caption("Chunks used to generate the answer.")

    if not sources:
        st.warning("No relevant sources were returned.")
        return

    for index, source_record in enumerate(sources, start=1):
        render_source_card(source_record, index, active_query)


def handle_submission(query: str) -> None:
    """Handle a query submission and store the result in session state.

    Args:
        query: User query.
    """
    if not query.strip():
        st.warning("Please enter a question before submitting.")
        return

    with st.spinner("Retrieving context and generating answer..."):
        try:
            result = call_rag_api(query=query)
            st.session_state.last_query = query
            st.session_state.last_result = result
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")
        except ValueError as exc:
            st.warning(str(exc))
        except KeyError as exc:
            st.error(f"Unexpected API response format: missing key {exc}")


def render_app() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="AI RAG Knowledge Assistant",
        page_icon="🤖",
        layout="wide",
    )

    initialize_session_state()
    render_sidebar()
    render_hero()
    render_example_queries()

    query = render_input_section()
    submit = st.button("Generate Answer", use_container_width=True)

    if submit:
        handle_submission(query)

    result = st.session_state.last_result
    active_query = st.session_state.last_query

    if result:
        answer = str(result.get("answer", "")).strip()
        sources = result.get("sources", [])

        if not answer:
            st.error("No valid answer generated. Try a different query.")
            return

        render_conversation(active_query, answer)
        render_retrieval_summary(active_query, sources)
        render_sources_section(sources, active_query)


if __name__ == "__main__":
    render_app()
