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
    MIN_KEYWORD_LENGTH = 2
    
    # API Keys (can be set via environment variables or UI)
    SEMANTIC_SCHOLAR_KEY = os.getenv("SEMANTIC_SCHOLAR_KEY", "")
    CORE_API_KEY = os.getenv("CORE_API_KEY", "")

    # Elsevier (Embase + Scopus) — requires institutional subscription (e.g. UCSF)
    # API key: https://dev.elsevier.com/  (register with your @ucsf.edu email)
    # Institutional token: request from your library or via the Elsevier portal after
    # registration; it unlocks full-record access for IP ranges outside the campus network.
    ELSEVIER_API_KEY = os.getenv("ELSEVIER_API_KEY", "")
    ELSEVIER_INST_TOKEN = os.getenv("ELSEVIER_INST_TOKEN", "")

    # Elsevier OAuth 2.0 (enables the one-click "Connect via Institution" button).
    # Register your app once at https://dev.elsevier.com/ to get a client ID and secret.
    # Users then connect with a single popup — Elsevier's login page supports
    # institutional SSO (Shibboleth / SAML), so UCSF users are redirected through
    # MyAccess automatically.  The resulting access token is used instead of a
    # static API key, so individual users never need to manage credentials.
    ELSEVIER_OAUTH_CLIENT_ID = os.getenv("ELSEVIER_OAUTH_CLIENT_ID", "")
    ELSEVIER_OAUTH_CLIENT_SECRET = os.getenv("ELSEVIER_OAUTH_CLIENT_SECRET", "")
    # Public URL this backend is reachable at — used as the OAuth redirect URI.
    # For local development: http://localhost:8000
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


class DataSource(Enum):
    """Available data sources for literature search."""
    PUBMED = "PubMed"
    ARXIV = "arXiv"
    BIORXIV = "bioRxiv"
    LOCAL_PDF = "Local PDFs"
    SEMANTIC_SCHOLAR = "Semantic Scholar"
    CORE = "CORE"
    EMBASE = "Embase"
    SCOPUS = "Scopus"
