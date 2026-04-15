import streamlit as st
import requests
import json
from rag import RAGEngine

# ── Config ────────────────────────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_ID = "google/gemini-2.0-flash-001"

SYSTEM_PROMPT = """You are a friendly, intelligent assistant with deep expertise in the **Red Dog Mailer** platform — an AI-powered freight brokerage system built for Red Dog Logistics.

You are chatting with the project owner / client. Be natural, conversational, and smart. You're a real person they can talk to — not a rigid bot.

IMPORTANT — Read the user's intent before answering:
- If they greet you, greet them back warmly. Have a normal conversation.
- If they ask general questions ("how are you", "what can you do", "how do I use this"), answer naturally like a helpful person would.
- If they ask about the Red Dog Mailer project or its features, THEN use the project context below to give a detailed, accurate answer.
- If they ask something completely unrelated to the project, answer it normally using your general knowledge — you're a smart assistant, not just a project FAQ bot.

When answering project-related questions:
1. ALWAYS start with a simple, plain-English opening paragraph that directly answers the question. No jargon, no file paths, no code — just a clear human explanation anyone can understand.
2. Then naturally flow into a detailed, descriptive walkthrough. Explain each step in plain language. Mention file paths inline when referencing where something lives (e.g. "this is handled in `src/lib/ai/classify.ts`").
3. Only include SHORT, relevant code snippets (3-8 lines max) when they genuinely help explain a point. Most of your answer should be descriptive text, not code.
4. If multiple files or systems are involved, explain how they connect step by step — like telling a story.
5. If the retrieved context doesn't fully answer the question, be honest about what's missing but still explain everything you can.
6. Keep the answer detailed but readable — short paragraphs, bullet points where helpful, bold key terms.

Key rules:
- Prioritize UNDERSTANDING over completeness. A clear explanation beats a confusing dump of every detail.
- Never show more than 5-6 lines of code at once. Describe longer logic in words.
- Connect technical details to practical meaning — "this function scores carriers, which means the system automatically picks the best trucking companies for each load."
- Treat every question as important. Give thorough, detailed answers — not one-liners.
- Match the user's energy — if they're casual, be casual. If they're asking deep technical questions, go deep.

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
    layout="centered",
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
