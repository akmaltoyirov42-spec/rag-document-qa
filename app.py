"""Streamlit front-end for the RAG Document Q&A system."""

import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.document_processor import process_pdf
from src.rag_chain import ask, build_rag_chain
from src.vector_store import build_vector_store

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat — Ask Your PDF",
    page_icon="📄",
    layout="wide",
)

st.title("📄 DocChat — Ask Your PDF")
st.markdown(
    "Upload a PDF and ask questions about it. "
    "Powered by **RAG** (Retrieval-Augmented Generation) with **GPT-4o-mini** + **FAISS**."
)

# ── Sidebar: settings & upload ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Get one at platform.openai.com",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and st.button("🔍 Process Document", use_container_width=True):
        if not api_key:
            st.error("Please enter your OpenAI API key first.")
        else:
            with st.spinner("Reading and indexing your PDF…"):
                # Save upload to a temp file so PyPDFLoader can read it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    chunks = process_pdf(tmp_path)
                    vector_store = build_vector_store(chunks)
                    st.session_state.chain = build_rag_chain(vector_store)
                    st.session_state.messages = []
                    st.session_state.doc_name = uploaded_file.name
                    st.success(f"✅ Indexed **{len(chunks)}** chunks from *{uploaded_file.name}*")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

    if "doc_name" in st.session_state:
        st.info(f"Active document: **{st.session_state.doc_name}**")

    st.divider()
    st.markdown(
        "**How it works:**\n"
        "1. PDF → text chunks\n"
        "2. Chunks → embeddings (OpenAI)\n"
        "3. Embeddings stored in **FAISS**\n"
        "4. Question → retrieve top chunks → **GPT-4o-mini** answers\n\n"
        "[View source on GitHub](https://github.com/yourusername/rag-document-qa)"
    )

# ── Chat interface ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources"):
                for doc in msg["sources"]:
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Page {page + 1}:** {doc.page_content[:300]}…")

# Input box
if prompt := st.chat_input("Ask a question about the document…"):
    if "chain" not in st.session_state:
        st.warning("Please upload and process a PDF first.")
    else:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result = ask(st.session_state.chain, prompt)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)
                    if sources:
                        with st.expander("📚 Sources"):
                            for doc in sources:
                                page = doc.metadata.get("page", "?")
                                st.markdown(f"**Page {page + 1}:** {doc.page_content[:300]}…")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
