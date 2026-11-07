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
    # Qwen is the house default for local work: it follows the structured-JSON
    # screening prompts more reliably than the llama tags and matches
    # NON_SCREENING_FALLBACK_MODEL, so one model serves both paths and Ollama
    # never has to swap weights mid-run.
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:7b")
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

    # Ollama serves a small number of inference slots per instance, so fanning
    # 16 workers at a local model just queues them: you pay the scheduling cost
    # without gaining throughput. Cloud APIs have no such limit, hence two knobs.
    PARALLEL_SCREENING_WORKERS_LOCAL = int(os.getenv("PARALLEL_SCREENING_WORKERS_LOCAL", "4"))

    # Ollama's default context window (2048 on most models) silently TRUNCATES
    # prompts that exceed it — the request still succeeds, it just loses the tail
    # of the abstract. The screening prompt (criteria + protocol + a 6000-char
    # abstract) routinely overflows 2048, so set this explicitly.
    OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))

    # Data extraction reads a whole paper rather than an abstract, and the answer
    # (one quoted snippet per field) is long too. num_ctx is the budget for BOTH,
    # so at 8192 a 30k-character paper plus a 60-field form leaves no room: the
    # prompt is truncated before the results section and the JSON is cut off
    # mid-answer. Measured on one paper, 7 fields answered only in the results:
    # 1/7 correct at 12k of text in 8192, 4/7 at 30k in 8192, 7/7 at 30k in 32768.
    # Local models in use here (qwen2.5, mistral) support 32k natively.
    # How many fields one extraction request may ask for.
    #
    # Measured on a 41,000-character paper with the default local model
    # (qwen2.5:7b), asking for values whose answers are known:
    #
    #     16 fields  ->  12 of 12 right, every quote verified, 15s
    #     20 fields  ->  no usable JSON at all, twice, 80s
    #     183 fields ->  no usable JSON at all
    #
    # The answer is what runs out of room, not the question: every field costs a
    # value, a source quote and a confidence in the reply. Twelve leaves a
    # margin under the limit measured above. A model that holds more can be
    # given more through the environment variable.
    EXTRACTION_FIELDS_PER_CALL = int(os.getenv("EXTRACTION_FIELDS_PER_CALL", "12"))

    EXTRACTION_NUM_CTX = int(os.getenv("EXTRACTION_NUM_CTX", "32768"))

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
