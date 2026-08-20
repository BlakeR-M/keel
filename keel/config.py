"""Configuration. Environment variables prefixed KEEL_ (see .env.example), optional .env file.

Everything has a safe default for the on-premise demo. The Azure profile reads no keys: it uses
DefaultAzureCredential (managed identity in Azure, `az login` locally).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Profile = Literal["local", "azure", "aws"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KEEL_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    profile: Profile = "local"
    data_dir: Path = Path("./data")
    airgap: bool = False

    # Retrieval
    chunk_size: int = 800  # characters
    chunk_overlap: int = 120
    top_k_bm25: int = 20
    top_k_vector: int = 20
    top_k_final: int = 6
    rerank: bool = True
    min_relevance: float = 0.15  # below this fused score, refuse rather than guess

    # Local profile
    local_llm_base_url: str = "http://127.0.0.1:8081/v1"
    local_llm_model: str = "qwen2.5-3b-instruct"
    local_llm_api_key: str = "local"  # llama-server ignores it; the OpenAI client requires a value
    local_llm_timeout: float = 120.0  # seconds per model call; raise it for a slow CPU-only server
    local_embed_model: str = "BAAI/bge-small-en-v1.5"
    local_rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Azure profile (no keys)
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_chat_deployment: str = "gpt-4o-mini"
    azure_openai_embed_deployment: str = "text-embedding-3-small"
    azure_search_endpoint: str = ""
    azure_search_index: str = "keel-chunks"

    # Generation
    max_output_tokens: int = 700
    temperature: float = 0.1

    # Judge (evals). Empty means "use the primary LLM provider".
    judge_model: str = ""
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # Web
    host: str = "127.0.0.1"
    port: int = 8400

    # Hosted demo of the fixture corpus. Never set these for a real deployment.
    demo_identity: bool = False
    """Beyond loopback, honour the chat page's self-asserted demo users (`public`, `hr-officer`) so a
    public demo can show permission filtering. Admin routes still need the admin token."""
    demo_readonly: bool = False
    """Declare the read-only posture: the web app exposes no ingest route, and every write that exists
    (approve, reject, quarantine release) sits under the admin guard. Shown on the chat page banner."""
    bootstrap_corpus: Path | None = None
    """A corpus manifest the web app ingests at startup when the store holds no documents."""

    @property
    def db_path(self) -> Path:
        return self.data_dir / "keel.db"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings


def reset_settings() -> None:
    """Tests call this after changing environment variables."""
    global _settings
    _settings = None
