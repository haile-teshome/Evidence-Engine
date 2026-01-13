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
    BIORXIV_LOOKBACK_DAYS = 180
    BIORXIV_MAX_ATTEMPTS = 20
    BIORXIV_BATCH_SIZE = 100
    PDF_MAX_PAGES = 3
    PDF_MAX_CHARS = 3000
    DEFAULT_MODEL = "llama3"
    MIN_KEYWORD_LENGTH = 2


class DataSource(Enum):
    """Available data sources for literature search."""
    PUBMED = "PubMed"
    BIG3_JOURNALS = "Big 3 Journals"
    ARXIV = "arXiv"
    BIORXIV = "bioRxiv"
    LOCAL_PDF = "Local PDFs"
