"""FastAPI app creation, main API routes."""

from fastapi import FastAPI

from ai_exercise.constants import SETTINGS, chroma_client, openai_client
from ai_exercise.llm.completions import create_prompt, extract_citations, get_completion
from ai_exercise.llm.embeddings import openai_ef
from ai_exercise.loading.document_loader import (
    Strategy,
    add_documents,
    build_docs_for_spec,
    get_all_specs,
)
from ai_exercise.models import (
    ChatOutput,
    ChatQuery,
    Document,
    HealthRouteOutput,
    LoadDocumentsOutput,
)
from ai_exercise.retrieval.bm25 import BM25Index
from ai_exercise.retrieval.retrieval import REFUSAL_MESSAGE, retrieve_or_refuse
from ai_exercise.retrieval.vector_store import create_collection

app = FastAPI()

collection = create_collection(
    chroma_client, openai_ef, SETTINGS.collection_name
)

# Global BM25 index, built at load time
bm25_index = BM25Index()


@app.get("/health")
def health_check_route() -> HealthRouteOutput:
    """Health check route to check that the API is up."""
    return HealthRouteOutput(status="ok")


@app.get("/load")
async def load_docs_route(
    strategy: Strategy = "structural",
) -> LoadDocumentsOutput:
    """Route to load documents from all 7 StackOne specs into vector store."""
    global collection
    collection_name = f"stackone_{strategy}"
    collection = create_collection(
        chroma_client, openai_ef, collection_name
    )

    all_documents: list[Document] = []
    for spec_name, json_data in get_all_specs():
        documents = build_docs_for_spec(
            spec_name, json_data, strategy=strategy
        )
        all_documents.extend(documents)
        print(f"  {spec_name}: {len(documents)} chunks")

    add_documents(collection, all_documents)

    # Build BM25 index from the same documents
    bm25_index.build(
        documents=[doc.page_content for doc in all_documents],
        metadatas=[doc.metadata or {} for doc in all_documents],
        doc_ids=[f"doc_{i}" for i in range(len(all_documents))],
    )

    print(
        f"Total documents in collection ({collection_name}): "
        f"{collection.count()}"
    )
    print(f"BM25 index built with {len(all_documents)} documents")
    return LoadDocumentsOutput(status="ok")


@app.post("/chat")
def chat_route(chat_query: ChatQuery) -> ChatOutput:
    """Chat route to chat with the API."""
    # Retrieve with hybrid search + refusal check
    active_bm25 = bm25_index if bm25_index._index is not None else None
    relevant_chunks, refused, retrieval_strategy = retrieve_or_refuse(
        collection=collection,
        query=chat_query.query,
        k=SETTINGS.k_neighbors,
        bm25_index=active_bm25,
    )

    if refused:
        return ChatOutput(
            message=REFUSAL_MESSAGE,
            retrieval_strategy=retrieval_strategy,
            refused=True,
        )

    # Create prompt with context
    prompt = create_prompt(
        query=chat_query.query, context=relevant_chunks
    )

    print(f"Prompt: {prompt}")

    # Get completion from LLM
    result = get_completion(
        client=openai_client,
        prompt=prompt,
        model=SETTINGS.openai_model,
    )

    return ChatOutput(
        message=result,
        retrieval_strategy=retrieval_strategy,
        refused=False,
        citations=extract_citations(result),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=80, reload=True)
