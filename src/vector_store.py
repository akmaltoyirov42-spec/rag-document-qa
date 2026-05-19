"""FAISS vector store: build, save, and search document embeddings."""

import os
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def get_embeddings() -> OpenAIEmbeddings:
    """Return OpenAI embeddings model (text-embedding-3-small is fast and cheap)."""
    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_vector_store(chunks: List[Document]) -> FAISS:
    """Create a FAISS index from document chunks."""
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def save_vector_store(vector_store: FAISS, path: str | Path) -> None:
    """Persist the FAISS index to disk."""
    vector_store.save_local(str(path))


def load_vector_store(path: str | Path) -> FAISS:
    """Load a previously saved FAISS index from disk."""
    embeddings = get_embeddings()
    return FAISS.load_local(str(path), embeddings, allow_dangerous_deserialization=True)


def similarity_search(
    vector_store: FAISS,
    query: str,
    k: int = 4,
) -> List[Document]:
    """Retrieve the top-k most relevant chunks for a query."""
    return vector_store.similarity_search(query, k=k)
