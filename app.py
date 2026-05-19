import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.document_processor import process_pdf
from src.rag_chain import ask, build_rag_chain
from src.vector_store import build_vector_store

load_dotenv()

st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")
st.title("📄 DocChat — Ask Your PDF")
st.write("Upload a PDF and ask questions about it.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and st.button("Process", use_container_width=True):
        if not api_key:
            st.error("Enter your OpenAI API key first.")
        else:
            with st.spinner("Indexing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                try:
                    chunks = process_pdf(tmp_path)
                    vector_store = build_vector_store(chunks)
                    st.session_state.chain = build_rag_chain(vector_store)
                    st.session_state.messages = []
                    st.session_state.doc_name = uploaded_file.name
                    st.success(f"Indexed {len(chunks)} chunks from {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

    if "doc_name" in st.session_state:
        st.info(f"Active: {st.session_state.doc_name}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for doc in msg["sources"]:
                    page = doc.metadata.get("page", "?")
                    st.markdown(f"**Page {page + 1}:** {doc.page_content[:300]}…")

if prompt := st.chat_input("Ask something about the document..."):
    if "chain" not in st.session_state:
        st.warning("Upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = ask(st.session_state.chain, prompt)
                    answer = result["answer"]
                    sources = result["sources"]
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for doc in sources:
                                page = doc.metadata.get("page", "?")
                                st.markdown(f"**Page {page + 1}:** {doc.page_content[:300]}…")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
