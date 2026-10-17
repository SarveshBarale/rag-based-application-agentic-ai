"""
app.py
======
Streamlit front-end for the Document Q&A RAG application powered by Google Gemini.

Flow:  User -> Streamlit UI -> RAGPipeline (retrieval) -> ChatGoogleGenerativeAI -> Answer

This file owns the UI and session state only; all RAG logic lives in
``rag_pipeline.py`` so the two concerns stay cleanly separated. The built vector
index is cached in ``st.session_state`` so we never re-embed on every message.
"""

from __future__ import annotations

import os
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    SUPPORTED_EXTENSIONS,
    DocumentLoadError,
    RAGPipeline,
    load_document,
)

# ---------------------------------------------------------------------------
# Environment — load API key strictly from .env (no sidebar input)
# ---------------------------------------------------------------------------

load_dotenv()

_api_key: str = (
    os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
).strip()

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DocuMind · Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme — CSS tokens for dark (default) and light-cream modes
# ---------------------------------------------------------------------------

# Initialise theme before any rendering so CSS is injected on first load too.
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

_DARK_CSS = """
    /* ════════════════  DARK THEME  ════════════════ */
    .stApp, body, html {
        background-color: #0e1117 !important;
    }
    section[data-testid="stMain"] { background-color: #0e1117 !important; }

    /* All body text */
    .stApp p, .stApp span, .stApp label,
    .stApp h1, .stApp h2, .stApp h3,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #e2e8f0 !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] { background: #161b26 !important; }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background: #161b26 !important;
        color: #e2e8f0 !important;
    }

    /* st.info / st.warning boxes */
    [data-testid="stAlert"] { background: #1e293b !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.07) !important;
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08) !important; }

    /* Hero subtitle */
    .hero-sub { color: #64748b; }

    /* Try asking label */
    .try-label { color: #94a3b8; }

    /* Stats card */
    .stats-card {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.25);
    }
    .stats-card .stat-row { color: #94a3b8; }
    .stats-card .stat-row span { color: #c7d2fe; }

    /* Source snippet */
    .source-snippet {
        color: #94a3b8;
        background: rgba(100,116,139,0.07);
        border-left: 3px solid rgba(99,102,241,0.4);
    }

    /* Expander */
    details[data-testid="stExpander"] {
        background: rgba(15,23,42,0.04) !important;
        border: 1px solid rgba(99,102,241,0.2) !important;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* Sample-question chips */
    div[data-testid="column"] button[kind="secondary"] {
        background: rgba(99,102,241,0.06) !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
        color: #6366f1 !important;
    }
    div[data-testid="column"] button[kind="secondary"]:hover {
        background: rgba(99,102,241,0.15) !important;
        border-color: rgba(99,102,241,0.6) !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.2);
    }

    /* Footer text */
    .sb-footer { color: #334155 !important; }

    /* Widgets — explicit dark styling regardless of system theme */
    [data-testid="stFileUploaderDropzone"] {
        background: #0b1220 !important;
        border: 1px dashed rgba(148,163,184,0.35) !important;
    }
    button[kind="secondary"],
    [data-testid="stBaseButton-secondary"] {
        background: #1e293b !important;
        border: 1px solid rgba(148,163,184,0.25) !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stFileChips"],
    [data-testid="stFileChips"] > div,
    [data-testid="stFileChip"],
    [data-testid="stFileChip"] > div,
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFile"] > div,
    [data-testid="stFileUploaderFileData"],
    [data-testid="stFileUploaderFileData"] > div {
        background: #1e293b !important;
        border-radius: 8px;
    }
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileChipName"],
    [data-testid="stFileChip"] *,
    [data-testid="stFileUploaderFile"] small,
    [data-testid="stFileUploaderFile"] span { color: #e2e8f0 !important; }
    [data-testid="stFileChipDeleteBtn"],
    [data-testid="stFileChipDeleteBtn"] button {
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button {
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] { background: #0e1117 !important; }
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] div[data-baseweb] { background: #161b26 !important; }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
        -webkit-text-fill-color: #64748b !important;
        opacity: 1 !important;
    }
    [data-testid="stExpander"],
    [data-testid="stExpander"] details { background: #111827 !important; }
"""

_LIGHT_CSS = """
    /* ════════════════  LIGHT CREAM THEME  ════════════════ */
    .stApp, body, html {
        background-color: #faf8f4 !important;
    }

    /* Main container */
    .main { background-color: #faf8f4 !important; }
    section[data-testid="stMain"] { background-color: #faf8f4 !important; }

    /* All body text */
    .stApp p, .stApp span, .stApp label,
    .stApp h1, .stApp h2, .stApp h3,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #2c2823 !important;
    }

    /* Sidebar — warm cream */
    section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #f0ebe0 0%, #e8e0d0 100%) !important;
        border-right: 1px solid rgba(120,100,80,0.15) !important;
    }
    section[data-testid="stSidebar"] * { color: #2c2823 !important; }
    section[data-testid="stSidebar"] .stFileUploader small { color: #78716c !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(120,100,80,0.2) !important; }

    /* Hero subtitle */
    .hero-sub { color: #78716c; }

    /* Try asking label */
    .try-label { color: #a8956e; }

    /* Stats card */
    .stats-card {
        background: rgba(99,102,241,0.06);
        border: 1px solid rgba(99,102,241,0.2);
    }
    .stats-card .stat-row { color: #78716c; }
    .stats-card .stat-row span { color: #4f46e5; }

    /* Source snippet */
    .source-snippet {
        color: #57534e;
        background: rgba(120,113,108,0.08);
        border-left: 3px solid rgba(99,102,241,0.45);
    }

    /* Expander */
    details[data-testid="stExpander"] {
        background: rgba(250,248,244,0.9) !important;
        border: 1px solid rgba(99,102,241,0.18) !important;
    }

    /* Divider */
    hr { border-color: rgba(120,100,80,0.18) !important; }

    /* Chat messages — cream background */
    [data-testid="stChatMessage"] {
        background: #f5f0e8 !important;
    }

    /* Sample-question chips */
    div[data-testid="column"] button[kind="secondary"] {
        background: rgba(99,102,241,0.07) !important;
        border: 1px solid rgba(99,102,241,0.28) !important;
        color: #4f46e5 !important;
    }
    div[data-testid="column"] button[kind="secondary"]:hover {
        background: rgba(99,102,241,0.14) !important;
        border-color: rgba(99,102,241,0.55) !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.15);
    }

    /* st.info / st.warning boxes */
    [data-testid="stAlert"] { background: #f5f0e8 !important; }

    /* chat input */
    [data-testid="stChatInput"] textarea {
        background: #f5f0e8 !important;
        color: #2c2823 !important;
    }

    /* Footer text */
    .sb-footer { color: #a8956e !important; }

    /* ── Widgets that keep Streamlit's native (dark) styling unless overridden ── */

    /* File uploader dropzone */
    [data-testid="stFileUploaderDropzone"] {
        background: #f5f0e8 !important;
        border: 1px dashed rgba(120,100,80,0.4) !important;
    }
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzoneInstructions"] * { color: #78716c !important; }
    [data-testid="stFileUploaderFile"] { background: transparent !important; }

    /* Uploaded file row (was a dark bar hiding the file name) */
    [data-testid="stFileChips"],
    [data-testid="stFileChips"] > div,
    [data-testid="stFileChip"],
    [data-testid="stFileChip"] > div,
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFile"] > div,
    [data-testid="stFileUploaderFileData"],
    [data-testid="stFileUploaderFileData"] > div {
        background: #fffdf8 !important;
        border-radius: 8px;
    }
    [data-testid="stFileUploaderFileName"],
    [data-testid="stFileChipName"],
    [data-testid="stFileChip"] *,
    [data-testid="stFileUploaderFile"] small,
    [data-testid="stFileUploaderFile"] span {
        color: #2c2823 !important;
    }
    [data-testid="stFileChipDeleteBtn"],
    [data-testid="stFileChipDeleteBtn"] button {
        background: transparent !important;
        border: none !important;
        color: #57534e !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] button {
        background: transparent !important;
        border: none !important;
        color: #57534e !important;
    }

    /* Secondary buttons (Upload, Clear Chat, sample chips) */
    button[kind="secondary"],
    [data-testid="stBaseButton-secondary"] {
        background: #fffdf8 !important;
        border: 1px solid rgba(120,100,80,0.3) !important;
        color: #2c2823 !important;
    }
    button[kind="secondary"]:hover,
    [data-testid="stBaseButton-secondary"]:hover {
        background: #f0ebe0 !important;
        border-color: rgba(99,102,241,0.55) !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"] *,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] * {
        color: #2c2823 !important;
    }
    /* Sample-question chips in the main area keep the indigo tint */
    [data-testid="stMain"] button[kind="secondary"],
    [data-testid="stMain"] [data-testid="stBaseButton-secondary"] {
        background: rgba(99,102,241,0.07) !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
    }

    /* Bottom bar behind the chat input + the input itself */
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] { background: #faf8f4 !important; }
    [data-testid="stChatInput"],
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] div[data-baseweb] {
        background: #f5f0e8 !important;
        border-color: rgba(120,100,80,0.28) !important;
    }
    [data-testid="stChatInput"] textarea {
        background: #f5f0e8 !important;
        color: #2c2823 !important;
        -webkit-text-fill-color: #2c2823 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #78716c !important;
        -webkit-text-fill-color: #78716c !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInputSubmitButton"] {
        background: rgba(99,102,241,0.12) !important;
        color: #4f46e5 !important;
    }

    /* Source expander */
    [data-testid="stExpander"],
    [data-testid="stExpander"] details {
        background: #fffdf8 !important;
        border-color: rgba(99,102,241,0.2) !important;
    }
"""

_COMMON_CSS = """
    /* ════════════════  COMMON (both themes)  ════════════════ */
    html, body, [class*="css"] { font-family: "Inter", "Segoe UI", sans-serif; }

    #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent !important; }
    footer { visibility: hidden; }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }

    /* Hero gradient title — works on both backgrounds */
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.1rem;
    }
    .hero-sub { font-size: 0.95rem; margin-bottom: 1.4rem; }

    /* Status badges */
    .badge-ready {
        display: inline-block; background: #dcfce7; color: #15803d;
        font-size: 0.78rem; font-weight: 600;
        padding: 0.25rem 0.7rem; border-radius: 999px;
        margin-left: 0.6rem; vertical-align: middle;
    }
    .badge-processing {
        display: inline-block; background: #fef3c7; color: #92400e;
        font-size: 0.78rem; font-weight: 600;
        padding: 0.25rem 0.7rem; border-radius: 999px;
        margin-left: 0.6rem; vertical-align: middle;
    }

    /* Badges sit inside the gradient-text title, which makes text transparent;
       restore a solid fill and beat the theme's generic span colour rules. */
    .stApp span.badge-ready {
        color: #15803d !important;
        -webkit-text-fill-color: #15803d !important;
        background: #dcfce7 !important;
    }
    .stApp span.badge-processing {
        color: #92400e !important;
        -webkit-text-fill-color: #92400e !important;
        background: #fef3c7 !important;
    }

    /* Stats card shell */
    .stats-card { border-radius: 12px; padding: 0.85rem 1rem; margin-top: 0.6rem; }
    .stats-card .stat-row { font-size: 0.82rem; margin-bottom: 0.2rem; }
    .stats-card .stat-row span { font-weight: 600; }

    /* Sample question chip layout */
    div[data-testid="column"] button[kind="secondary"] {
        border-radius: 10px !important;
        font-size: 0.83rem !important;
        transition: all 0.18s ease;
        white-space: normal !important;
        text-align: left !important;
        height: auto !important;
        min-height: 54px;
        padding: 0.55rem 0.75rem !important;
    }
    div[data-testid="column"] button[kind="secondary"]:hover { transform: translateY(-1px); }
    div[data-testid="column"] button[kind="secondary"]:disabled {
        opacity: 0.35 !important; transform: none !important;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 14px; padding: 0.2rem 0; margin-bottom: 0.4rem;
    }

    /* Expander */
    details[data-testid="stExpander"] { border-radius: 10px !important; margin-top: 0.5rem; }
    details[data-testid="stExpander"] summary {
        font-size: 0.85rem; color: #6366f1 !important; font-weight: 600;
    }

    /* Sidebar process button */
    section[data-testid="stSidebar"] button[kind="primary"] {
        border-radius: 10px !important; font-weight: 700 !important;
        letter-spacing: 0.02em; margin-top: 0.4rem;
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important; color: #ffffff !important;
        transition: opacity 0.15s;
    }
    section[data-testid="stSidebar"] button[kind="primary"]:hover { opacity: 0.88; }

    /* Source snippet shell */
    .source-snippet {
        font-size: 0.8rem; padding: 0.4rem 0.65rem;
        border-radius: 0 6px 6px 0; margin-top: 0.25rem; white-space: pre-wrap;
    }

    /* Theme toggle label override */
    section[data-testid="stSidebar"] .stToggle label { color: inherit !important; }
"""


def _inject_theme_css() -> None:
    theme_css = _DARK_CSS if st.session_state.theme == "dark" else _LIGHT_CSS
    st.markdown(
        f"<style>{_COMMON_CSS}{theme_css}</style>",
        unsafe_allow_html=True,
    )


def _on_theme_toggle() -> None:
    """Callback: runs before the rerun, so the new CSS is injected immediately."""
    st.session_state.theme = "dark" if st.session_state.theme_toggle else "light"


_inject_theme_css()

# ---------------------------------------------------------------------------
# Sample questions
# ---------------------------------------------------------------------------

SAMPLE_QUESTIONS = [
    "What is the main objective of this document?",
    "Explain the methodology used.",
    "What are the key findings or conclusions?",
]

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_state() -> None:
    """Create session-state keys once so reruns don't wipe history or the index."""
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pipeline", None)
    st.session_state.setdefault("indexed_files", [])
    st.session_state.setdefault("chunk_count", 0)
    st.session_state.setdefault("is_processing", False)


_init_state()

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def render_sources(sources: list) -> None:
    """Render retrieved source chunks inside a collapsible expander."""
    if not sources:
        return
    with st.expander(f"🔍 View Retrieved Context & Sources ({len(sources)} chunks)"):
        for s in sources:
            st.markdown(
                f"**[{s['index']}] {s['source']}** &nbsp;·&nbsp; page {s['page']}",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='source-snippet'>{s['content']}</div>",
                unsafe_allow_html=True,
            )
            if s["index"] < len(sources):
                st.divider()


def process_documents(uploaded_files: list, api_key: str) -> None:
    """Parse, chunk, embed and index the uploaded files into a fresh pipeline."""
    documents = []
    for uploaded in uploaded_files:
        uploaded.seek(0)
        documents.extend(load_document(uploaded))

    pipeline = RAGPipeline(api_key=api_key)
    chunk_count = pipeline.build_index(documents)

    st.session_state.pipeline = pipeline
    st.session_state.indexed_files = [f.name for f in uploaded_files]
    st.session_state.chunk_count = chunk_count
    st.session_state.messages = []
    st.session_state.is_processing = False


def answer_question(query: str) -> None:
    """Retrieve context, stream the answer, and record the turn in history."""
    pipeline: RAGPipeline = st.session_state.pipeline
    history = list(st.session_state.messages)

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        try:
            docs = pipeline.retrieve(query)
            context = RAGPipeline._format_context(docs)
            sources = RAGPipeline.format_sources(docs)
            answer = st.write_stream(
                pipeline.stream_answer(query, chat_history=history, context=context)
            )
            render_sources(sources)
        except Exception as exc:
            answer = f"⚠️ Something went wrong while answering: {exc}"
            sources = []
            st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


# ---------------------------------------------------------------------------
# Guard: API key must be present
# ---------------------------------------------------------------------------

if not _api_key:
    st.error(
        "🔑 **Missing API key.** Please add `GEMINI_API_KEY=your_key` to your `.env` file "
        "and restart the app.  \n"
        "Get a free key at [Google AI Studio →](https://aistudio.google.com/app/apikey)"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — 📂 Document Manager
# ---------------------------------------------------------------------------

with st.sidebar:
    st.toggle(
        "🌙 Dark mode" if st.session_state.theme == "dark" else "☀️ Light mode",
        value=st.session_state.theme == "dark",
        key="theme_toggle",
        on_change=_on_theme_toggle,
    )

    st.markdown(
        "<h2 style='color:#c7d2fe;font-size:1.25rem;font-weight:800;"
        "margin-bottom:0.1rem;'>📂 Document Manager</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#64748b;font-size:0.8rem;margin-bottom:1.1rem;'>"
        "Upload documents and build the knowledge index.</p>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Drop your files here",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT.",
        label_visibility="collapsed",
    )
    if uploaded_files:
        st.caption(f"📎 {len(uploaded_files)} file(s) selected")

    process_clicked = st.button(
        "⚡ Process & Index Document",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    if process_clicked:
        if not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            st.session_state.is_processing = True
            with st.spinner("Reading, chunking and embedding with Gemini…"):
                try:
                    process_documents(uploaded_files, _api_key)
                    st.success(
                        f"✅ Indexed **{st.session_state.chunk_count}** chunks from "
                        f"**{len(st.session_state.indexed_files)}** file(s)."
                    )
                except DocumentLoadError as exc:
                    st.session_state.is_processing = False
                    st.error(f"Could not read a document: {exc}")
                except Exception as exc:
                    st.session_state.is_processing = False
                    st.error(f"Failed to build the index: {exc}")

    if st.session_state.pipeline is not None:
        is_processing = st.session_state.get("is_processing", False)
        badge_html = (
            "<span class='badge-processing'>⏳ Processing…</span>"
            if is_processing
            else "<span class='badge-ready'>🟢 Ready for Questions</span>"
        )
        file_rows = "".join(
            f"<div class='stat-row'>📄 <span>{name}</span></div>"
            for name in st.session_state.indexed_files
        )
        st.markdown(
            f"""
            <div class='stats-card'>
                {file_rows}
                <div class='stat-row' style='margin-top:0.5rem;'>
                    Chunks indexed: <span>{st.session_state.chunk_count}</span>
                </div>
                <div style='margin-top:0.5rem;'>{badge_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:auto;'></div>", unsafe_allow_html=True)
    st.divider()

    if st.button("🗑️ Clear Chat / Reset Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pipeline = None
        st.session_state.indexed_files = []
        st.session_state.chunk_count = 0
        st.session_state.is_processing = False
        st.rerun()

    st.markdown(
        "<p style='font-size:0.72rem;color:#334155;text-align:center;margin-top:1rem;'>"
        "Powered by Google Gemini · LangChain · FAISS</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main — chat interface
# ---------------------------------------------------------------------------

ready = st.session_state.pipeline is not None

active_doc_badge = ""
if ready and st.session_state.indexed_files:
    doc_names = ", ".join(st.session_state.indexed_files)
    active_doc_badge = (
        f"<span class='badge-ready' title='{doc_names}'>🟢 {st.session_state.indexed_files[0]}"
        + (" +more" if len(st.session_state.indexed_files) > 1 else "")
        + "</span>"
    )

st.markdown(
    f"<div class='hero-title'>🧠 DocuMind{active_doc_badge}</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='hero-sub'>Ask questions about your documents — answers grounded in "
    "content using RAG + Google Gemini.</div>",
    unsafe_allow_html=True,
)

if not ready:
    st.info(
        "👈 **Get started** — open the **Document Manager** in the sidebar, "
        "upload a PDF / DOCX / TXT, then click **⚡ Process & Index Document**."
    )

st.markdown(
    "<p class='try-label' style='font-size:0.85rem;margin-bottom:0.3rem;font-weight:600;'>"
    "💡 Try asking</p>",
    unsafe_allow_html=True,
)
sample_query: str | None = None
cols = st.columns(len(SAMPLE_QUESTIONS), gap="small")
for col, question in zip(cols, SAMPLE_QUESTIONS):
    if col.button(question, disabled=not ready, use_container_width=True):
        sample_query = question

st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources") or [])

typed_query = st.chat_input(
    "Ask a question about your document(s)…",
    disabled=not ready,
)
query = typed_query or sample_query

if query and ready:
    answer_question(query)
elif query and not ready:
    st.warning("⚠️ Please upload and index a document first.")