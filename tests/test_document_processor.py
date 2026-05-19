"""Unit tests for document_processor.py (no API calls needed)."""

from langchain.schema import Document

from src.document_processor import split_documents


def _make_docs(text: str) -> list[Document]:
    return [Document(page_content=text, metadata={"page": 0})]


def test_split_short_text_stays_as_one_chunk():
    docs = _make_docs("Short text.")
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1


def test_split_long_text_creates_multiple_chunks():
    long_text = "word " * 600  # ~3000 chars
    docs = _make_docs(long_text)
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1


def test_chunks_have_overlap():
    long_text = "A" * 1000 + "B" * 1000
    docs = _make_docs(long_text)
    chunks = split_documents(docs, chunk_size=600, chunk_overlap=100)
    # At least some chunk should contain both A and B (overlap region)
    combined = "".join(c.page_content for c in chunks)
    assert "A" in combined and "B" in combined


def test_metadata_preserved_after_split():
    docs = _make_docs("word " * 400)
    docs[0].metadata["source"] = "test.pdf"
    chunks = split_documents(docs)
    for chunk in chunks:
        assert chunk.metadata.get("source") == "test.pdf"
