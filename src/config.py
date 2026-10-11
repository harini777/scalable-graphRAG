from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_DIR: Path = BASE_DIR / "data" / "cache"
    BENCHMARK_DIR: Path = BASE_DIR / "data" / "benchmarks"
    EXPERIMENT_DIR: Path = BASE_DIR / "experiments_output"

    # LLM Settings
    LLM_PROVIDER: str = "openai"  # "openai", "gemini", "anthropic", "groq", "ollama", "mock"
    LLM_MODEL: str = "gpt-4o-mini"
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Embedding Settings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384
    DEVICE: str = "mps" if os.uname().sysname == "Darwin" and os.uname().machine == "arm64" else "cpu"

    # RAG Settings
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K: int = 5

    def ensure_directories(self) -> None:
        """Create needed directory hierarchy if not existing."""
        for path in [self.DATA_DIR, self.CACHE_DIR, self.BENCHMARK_DIR, self.EXPERIMENT_DIR]:
            path.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
