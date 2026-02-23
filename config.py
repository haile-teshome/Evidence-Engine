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
    SEMANTIC_SCHOLAR_KEY = os.getenv("SEMANTIC_SCHOLAR_KEY", "")
    CORE_API_KEY = os.getenv("CORE_API_KEY", "")


class DataSource(Enum):
    """Available data sources for literature search."""
    PUBMED = "PubMed"
    ARXIV = "arXiv"
    BIORXIV = "bioRxiv"
    LOCAL_PDF = "Local PDFs"
    SEMANTIC_SCHOLAR = "Semantic Scholar"
    CORE = "CORE"
