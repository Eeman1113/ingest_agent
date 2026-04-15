import streamlit as st
import requests
import json
from rag import RAGEngine

# ── Config ────────────────────────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_ID = "google/gemini-2.0-flash-001"

SYSTEM_PROMPT = """You are a senior technical consultant with deep expertise in the **Red Dog Mailer** codebase — an AI-powered freight brokerage platform built for Red Dog Logistics.

You are speaking with the project owner / client. Your answers must be:
- **Precise**: Always cite specific file paths (e.g. `src/lib/ai/classify.ts`). Never say "somewhere in the code".
- **Code-backed**: When asked about functions, show the actual function signature, key logic, and explain parameters. Pull directly from the provided context.
- **Traceable**: When explaining features, trace the full call chain (e.g. "The API route `api/emails/classify/route.ts` calls `classifyEmail()` from `src/lib/ai/classify.ts`, which uses the prompt from `src/lib/learning/prompts.ts`...").
- **Honest**: If the retrieved context doesn't fully answer the question, say exactly what's missing and which files would need to be checked.
- **Business-aware**: For non-technical questions, translate code concepts into business impact.

When answering:
1. ALWAYS start with a simple, plain-English opening paragraph that any non-technical person can understand — no jargon, no file paths, no code. Just a clear human explanation of the answer.
2. Then naturally continue into the technical details, code references, file paths, and call chains as one continuous response. Do NOT separate them with headers or labels — it should read as one flowing answer that starts simple and gets deeper.

PROJECT DIRECTORY STRUCTURE:
{dir_tree}

RETRIEVED CODE CONTEXT (most relevant files for this query):
{context}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Indexing codebase (BM25 + TF-IDF hybrid)...")
def load_rag():
    return RAGEngine("data/ingest.txt")


def stream_chat(messages: list, model: str, api_key: str):
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.2,
            "max_tokens": 4096,
        },
        stream=True,
        timeout=90,
    )
    resp.raise_for_status()

    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        payload = decoded[6:]
        if payload.strip() == "[DONE]":
            break
        try:
            delta = json.loads(payload)["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                yield token
        except (json.JSONDecodeError, KeyError, IndexError):
            continue


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Red Dog Mailer — Project Agent",
    page_icon="🐕",
    layout="wide",
)

st.markdown(
    """
<style>
    .block-container { max-width: 900px; }
    .stChatMessage { border-radius: 12px; }
    div[data-testid="stExpander"] { border: 1px solid #333; border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.title("Red Dog Mailer")
    st.caption("AI Project Agent")
    st.divider()

    top_k = st.slider("Sources to retrieve", 4, 25, 15)
    st.divider()

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# API key
api_key = st.secrets.get("OPENROUTER_API_KEY", "")
if not api_key:
    st.error("Set `OPENROUTER_API_KEY` in `.streamlit/secrets.toml` or Streamlit Cloud secrets.")
    st.stop()

# Load RAG
rag = load_rag()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📂 Sources ({len(msg['sources'])} files)"):
                for s in msg["sources"]:
                    st.code(s, language=None)

# Chat input
if prompt := st.chat_input("Ask anything about the Red Dog Mailer project..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve
    with st.spinner("Searching codebase..."):
        context, sources = rag.get_context(prompt, top_k=top_k)

    # Build LLM messages — trim dir tree for token budget
    dir_tree_trimmed = rag.dir_tree[:4000]
    sys_msg = SYSTEM_PROMPT.format(dir_tree=dir_tree_trimmed, context=context)

    # Keep last 6 conversation turns for continuity
    history = st.session_state.messages[-7:-1]
    llm_messages = [{"role": "system", "content": sys_msg}]
    for m in history:
        llm_messages.append({"role": m["role"], "content": m["content"]})
    llm_messages.append({"role": "user", "content": prompt})

    # Stream response
    with st.chat_message("assistant"):
        try:
            full_response = st.write_stream(stream_chat(llm_messages, MODEL_ID, api_key))
        except requests.exceptions.HTTPError as e:
            full_response = f"API error: {e.response.status_code} — {e.response.text}"
            st.error(full_response)

        if sources:
            with st.expander(f"📂 Sources ({len(sources)} files)"):
                for s in sources:
                    st.code(s, language=None)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response, "sources": sources}
    )
