"""Set up some constants for the project."""

import chromadb
from openai import OpenAI
from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings for the demo app.

    Reads from environment variables.
    You can create the .env file from the .env_example file.

    !!! SecretStr is a pydantic type that hides the value in logs.
    If you want to use the real value, you should do:
    SETTINGS.<variable>.get_secret_value()
    """

    class Config:
        """Config for the settings."""

        env_file = ".env"

    openai_api_key: SecretStr
    openai_model: str = "gpt-4o"
    embeddings_model: str = "text-embedding-3-small"

    collection_name: str = "documents"
    chunk_size: int = 1000
    k_neighbors: int = 5

    # You will want to point towards all of the specs not just this one
    docs_url: str = "https://api.eu1.stackone.com/oas/stackone.json"
    # All 7 StackOne OpenAPI specs
    spec_urls: list[str] = [
        "https://api.eu1.stackone.com/oas/stackone.json",
        "https://api.eu1.stackone.com/oas/hris.json",
        "https://api.eu1.stackone.com/oas/ats.json",
        "https://api.eu1.stackone.com/oas/lms.json",
        "https://api.eu1.stackone.com/oas/iam.json",
        "https://api.eu1.stackone.com/oas/crm.json",
        "https://api.eu1.stackone.com/oas/marketing.json",
    ]


SETTINGS = Settings()  # type: ignore


# clients
openai_client = OpenAI(api_key=SETTINGS.openai_api_key.get_secret_value())
chroma_client = chromadb.PersistentClient(path="./.chroma_db")
