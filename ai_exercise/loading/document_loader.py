"""Document loader for the RAG example."""

import json
from typing import Any

import chromadb
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai_exercise.constants import SETTINGS
from ai_exercise.loading.chunk_json import chunk_data
from ai_exercise.models import Document


def get_json_data() -> dict[str, Any]:
    try:
        response = requests.get(SETTINGS.docs_url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Failed to fetch {SETTINGS.docs_url}: {e}")
        raise

def document_json_array(data: list[dict[str, Any]], source: str) -> list[Document]:
    """Converts an array of JSON chunks into a list of Document objects."""
    return [
        Document(page_content=json.dumps(item), metadata={"source": source})
        for item in data
    ]


def build_docs(data: dict[str, Any]) -> list[Document]:
    """Chunk (badly) and convert the JSON data into a list of Document objects."""
    docs = []
    for attribute in ["paths", "webhooks", "components"]:
        chunks = chunk_data(data, attribute)
        docs.extend(document_json_array(chunks, attribute))
    return docs


def split_docs(docs_array: list[Document]) -> list[Document]:
    """Some may still be too long, so we split them."""
    splitter = RecursiveCharacterTextSplitter(
        separators=["}],", "},", "}", "]", " ", ""], chunk_size=SETTINGS.chunk_size
    )
    return splitter.split_documents(docs_array)

def add_documents(
    collection: chromadb.Collection,
    docs: list[Document],
    batch_size: int = 100,
) -> None:
    """Add documents to the collection in batches to stay under embedding API limits."""
    total = len(docs)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = docs[start:end]
        collection.add(
            documents=[doc.page_content for doc in batch],
            metadatas=[doc.metadata or {} for doc in batch],
            ids=[f"doc_{i}" for i in range(start, end)],
        )
        print(f"  Embedded batch {start}-{end} of {total}")

def get_all_specs() -> list[tuple[str, dict[str, Any]]]:
    """Fetch all StackOne OpenAPI specs. Returns list of (spec_name, json_data)."""
    results = []
    for url in SETTINGS.spec_urls:
        # Extract spec name from URL: .../stackone.json -> stackone
        spec_name = url.split("/")[-1].replace(".json", "")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            results.append((spec_name, response.json()))
            print(f"Fetched {spec_name}.json")
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {e}")
    return results