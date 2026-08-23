# ============================================================================
# FILE: config.py
# Configuration and constants for the application
# ============================================================================

from enum import Enum
import os 


class Config:
    """Application configuration constants."""
    APP_TITLE = "Evidence Engine"
    PAGE_ICON = "🧪"
    ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "researcher@example.com")
    ARXIV_API_URL = "http://export.arxiv.org/api/query"
    BIORXIV_API_URL = "https://api.biorxiv.org/details/biorxiv"
    SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    CORE_API_URL = "https://api.core.ac.uk/v3/search/works"
    BIORXIV_LOOKBACK_DAYS = 180
    BIORXIV_MAX_ATTEMPTS = 20
    BIORXIV_BATCH_SIZE = 100
    PDF_MAX_PAGES = 3
    PDF_MAX_CHARS = 3000
    DEFAULT_MODEL = "llama3"
    # Local embedding model for the semantic-similarity layer (find-similar, seed
    # anchoring, deep-scan). Runs on the same Ollama by default (nothing leaves the
    # machine); set EMBED_MODEL=text-embedding-3-small to use OpenAI instead.
    EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
    MIN_KEYWORD_LENGTH = 2
    
    # API Keys (can be set via environment variables or UI)
    SEMANTIC_SCHOLAR_KEY = os.getenv("SEMANTIC_SCHOLAR_KEY", "")
    CORE_API_KEY = os.getenv("CORE_API_KEY", "")

    # Public URL this backend is reachable at. For local development: http://localhost:8000
    APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8000")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Parallel processing configuration for Ollama
    # Ollama can handle multiple concurrent requests, but limit depends on:
    # - Model size (larger models need more VRAM, reduce workers)
    # - GPU VRAM available (8GB ~ 2-3 workers for 7B models)
    # - CPU vs GPU inference (CPU can handle more but slower)
    # Recommended: 4-8 for local Ollama, 8-16 for cloud APIs
    PARALLEL_SCREENING_WORKERS = int(os.getenv("PARALLEL_SCREENING_WORKERS", "16"))
    PARALLEL_AGENT_WORKERS = int(os.getenv("PARALLEL_AGENT_WORKERS", "16"))

    # Reproducibility. A fixed decoding seed plus temperature=0 make LOCAL (Ollama)
    # and OpenAI runs deterministic, so a review can be re-run to identical output.
    # Anthropic and Gemini expose no seed API, so they are best-effort only.
    # PROMPT_VERSION is stamped into the run manifest / audit log and bumped whenever
    # a prompt template changes, so a stored decision can be tied to the exact prompt.
    RUN_SEED = int(os.getenv("EE_SEED", "42"))
    PROMPT_VERSION = os.getenv("EE_PROMPT_VERSION", "2026.08.1")


class DataSource(Enum):
    """Available data sources for literature search."""
    PUBMED = "PubMed"
    ARXIV = "arXiv"
    BIORXIV = "bioRxiv"
    LOCAL_PDF = "Local PDFs"
    SEMANTIC_SCHOLAR = "Semantic Scholar"
    CORE = "CORE"
