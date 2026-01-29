# ============================================================================
# FILE: config.py
# Configuration and constants for the application
# ============================================================================

from enum import Enum


class Config:
    """Application configuration constants."""
    APP_TITLE = "Global Epi-Agent Pro"
    PAGE_ICON = "🧪"
    ENTREZ_EMAIL = "researcher@example.com"
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
    SEMANTIC_SCHOLAR_KEY = "18DkUdgrPW3OYHmj7OYWq8M0rg8VD5iraUBg5WQP"
    CORE_API_KEY = "n9W4CIJbKULDkcjAmyMG2rPQOX5RvS8E"


class DataSource(Enum):
    """Available data sources for literature search."""
    PUBMED = "PubMed"
    BIG3_JOURNALS = "Big 3 Journals"
    ARXIV = "arXiv"
    BIORXIV = "bioRxiv"
    LOCAL_PDF = "Local PDFs"
    SEMANTIC_SCHOLAR = "Semantic Scholar"
    CORE = "CORE"
