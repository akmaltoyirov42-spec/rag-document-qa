from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_vector_store(chunks: List[Document]) -> FAISS:
    return FAISS.from_documents(chunks, get_embeddings())


def save_vector_store(vector_store: FAISS, path: str | Path) -> None:
    vector_store.save_local(str(path))


def load_vector_store(path: str | Path) -> FAISS:
    return FAISS.load_local(str(path), get_embeddings(), allow_dangerous_deserialization=True)


def similarity_search(vector_store: FAISS, query: str, k: int = 4) -> List[Document]:
    return vector_store.similarity_search(query, k=k)
