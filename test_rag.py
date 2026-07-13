import os
from ingest import clean_chunk_text, sanitize_collection_name
from langchain.schema import Document


def test_sanitize_collection_name():
    assert sanitize_collection_name("https://example.com") == "example_com"
    assert sanitize_collection_name("http://sub.domain.io") == "sub_domain_io"


def test_clean_chunk_text():
    text = "  Hello\n\nWorld  \n"
    assert clean_chunk_text(text) == "Hello World"


def test_chunk_documents_empty():
    docs = []
    from ingest import chunk_documents
    assert chunk_documents(docs) == []


def test_chunk_documents_split():
    docs = [Document(page_content="A\n\n" * 1200, metadata={"source_url": "https://example.com"})]
    from ingest import chunk_documents
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 2
    assert all("source_url" in chunk.metadata for chunk in chunks)


if __name__ == "__main__":
    test_sanitize_collection_name()
    test_clean_chunk_text()
    test_chunk_documents_empty()
    test_chunk_documents_split()
    print("All tests passed")
