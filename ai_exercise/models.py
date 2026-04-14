"""Types for the API."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class Document:
    """A document to be added to the vector store."""

    page_content: str
    metadata: dict[str, Any] | None = None


class HealthRouteOutput(BaseModel):
    """Model for the health route output."""

    status: str


class LoadDocumentsOutput(BaseModel):
    """Model for the load documents route output."""

    status: str


class ChatQuery(BaseModel):
    """Model for the chat input."""

    query: str


class ChatOutput(BaseModel):
    """Model for the chat route output."""

    message: str
    retrieval_strategy: str | None = None
    refused: bool = False
    citations: list[str] = []
