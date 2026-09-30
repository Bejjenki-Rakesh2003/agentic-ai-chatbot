import time
import streamlit as st

from src.config import missing_credentials

try:
    from src.graph import rag_graph
    rag_graph_ready = True
except Exception as exc:
    rag_graph = None
    rag_graph_ready = False
    startup_error = exc

try:
    from src.embedding import index
    vector_store_ready = index is not None
except Exception:
    vector_store_ready = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Main app */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}

/* Header */
.main-header {
    text-align: center;
    padding: 10px 0 20px 0;
}

.main-header h1 {
    color: white;
    margin-bottom: 5px;
}

.main-header p {
    color: #8B949E;
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 12px;
    margin-bottom: 10px;
}

[data-testid="stChatMessageContent"] {
    font-size: 16px;
}

/* User Message */
[data-testid="chatAvatarIcon-user"] {
    background-color: #238636;
}

/* Assistant Message */
[data-testid="chatAvatarIcon-assistant"] {
    background-color: #1F6FEB;
}

/* Input */
.stChatInputContainer {
    border-top: 1px solid #30363D;
}

/* Source Cards */
.source-card {
    border: 1px solid #30363D;
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 10px;
    background-color: #161B22;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>🤖 Agentic AI Assistant</h1>
    <p>Ask questions about Agentic AI from your knowledge base</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 Knowledge Assistant")

    if vector_store_ready:
        st.success("Knowledge Base Connected")
    else:
        st.error("Knowledge Base Offline")

    st.divider()

    st.metric(
        "Messages",
        len(st.session_state.get("messages", []))
    )

    st.divider()

    st.markdown("""
### Features

- Semantic Search
- Pinecone Vector DB
- Groq LLM
- Source Citations
- Agentic AI Knowledge Base
""")

    st.divider()

    if st.button("🗑️ New Chat"):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================================
# CONFIGURATION CHECK
# ============================================================

missing = missing_credentials()

if not rag_graph_ready or not vector_store_ready:

    st.error(
        "Assistant startup failed."
    )

    if missing:
        st.warning(
            "Missing variables: " +
            ", ".join(missing)
        )

    if "startup_error" in globals():
        st.code(str(startup_error))

    st.stop()

# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("📚 Sources Used"):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        <div class="source-card">
                            <b>Page:</b> {source.get("page","N/A")}
                            <br>
                            <b>Relevance:</b>
                            {source.get("score",0):.3f}
                            <br><br>
                            {source.get("text","")[:500]}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# ============================================================
# WELCOME SECTION
# ============================================================

if len(st.session_state.messages) == 0:

    st.markdown("## How can I help you today?")

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "📖 What is Agentic AI?"
        ):
            st.session_state.pending_question = (
                "What is Agentic AI?"
            )
            st.rerun()

        if st.button(
            "🤝 Multi-Agent Collaboration"
        ):
            st.session_state.pending_question = (
                "What is multi-agent collaboration?"
            )
            st.rerun()

    with col2:

        if st.button(
            "🧠 Building Blocks"
        ):
            st.session_state.pending_question = (
                "What are the building blocks of Agentic AI?"
            )
            st.rerun()

        if st.button(
            "⚙ Traditional AI vs Agentic AI"
        ):
            st.session_state.pending_question = (
                "How does Agentic AI differ from traditional AI?"
            )
            st.rerun()

# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask anything about Agentic AI..."
)

if "pending_question" in st.session_state:
    question = st.session_state.pop(
        "pending_question"
    )

# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    question = question.strip()

    if not question:
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching knowledge base..."
        ):

            try:

                result = rag_graph.invoke(
                    {
                        "question": question
                    }
                )

                answer = result.get(
                    "answer",
                    ""
                )

                if not answer:
                    answer = (
                        "I could not find "
                        "an answer."
                    )

                placeholder = st.empty()

                typed_text = ""

                for word in answer.split():

                    typed_text += word + " "

                    placeholder.markdown(
                        typed_text
                    )

                    time.sleep(0.01)

                retrieved_docs = result.get(
                    "documents",
                    []
                )

                sources = []

                for doc in retrieved_docs:

                    if hasattr(
                        doc,
                        "metadata"
                    ):

                        metadata = (
                            doc.metadata or {}
                        )

                        sources.append(
                            {
                                "page":
                                metadata.get(
                                    "page",
                                    "N/A"
                                ),
                                "score":
                                metadata.get(
                                    "score",
                                    0
                                ),
                                "text":
                                doc.page_content,
                            }
                        )

                    elif isinstance(
                        doc,
                        dict
                    ):

                        sources.append(
                            {
                                "page":
                                doc.get(
                                    "page",
                                    "N/A"
                                ),
                                "score":
                                doc.get(
                                    "score",
                                    0
                                ),
                                "text":
                                doc.get(
                                    "text",
                                    ""
                                ),
                            }
                        )

                if sources:

                    with st.expander(
                        "📚 Sources Used"
                    ):

                        for source in sources:

                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <b>Page:</b>
                                    {source['page']}
                                    <br>
                                    <b>Score:</b>
                                    {source['score']:.3f}
                                    <br><br>
                                    {source['text'][:500]}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as e:

                st.error(
                    "Something went wrong."
                )

                st.exception(e)