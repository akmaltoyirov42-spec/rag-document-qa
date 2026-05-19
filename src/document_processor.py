"""Handles PDF loading, text extraction, and chunking."""

from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str | Path) -> List[Document]:
    """Load a PDF file and return a list of Document objects (one per page)."""
    loader = PyPDFLoader(str(file_path))
    return loader.load()


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into overlapping chunks for better retrieval.

    chunk_overlap ensures context isn't lost at chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def process_pdf(file_path: str | Path) -> List[Document]:
    """Full pipeline: load PDF → split into chunks."""
    docs = load_pdf(file_path)
    chunks = split_documents(docs)
    return chunks
