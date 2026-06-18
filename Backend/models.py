# ============================================================================
# FILE: models.py
# Data models and structures
# ============================================================================

import html
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Inline formatting tags that sources (Europe PMC, PubMed, CrossRef JATS, …)
# leave in titles/abstracts, e.g. "<i>T. gondii</i>". We strip ONLY these known
# tags (not arbitrary <...>) so math like "p < 0.05" is preserved.
_FMT_TAG_RE = re.compile(
    r"</?(?:i|b|u|em|strong|sub|sup|inf|sc|span|bold|italic|underline|small|p|br|"
    r"title|sec|xref|ext-link|jats:[\w-]+)\b[^>]*>",
    re.IGNORECASE,
)


def clean_markup(s: Optional[str]) -> str:
    """Decode HTML entities and strip inline markup tags from source text.

    Handles double-encoded entities (e.g. "&amp;lt;i&amp;gt;") and the raw-tag
    case ("<i>…</i>"), collapsing whitespace afterwards. Returns "" for falsy
    input.
    """
    if not s:
        return s or ""
    out = str(s)
    # Decode entities, twice if the source double-encoded them.
    for _ in range(2):
        decoded = html.unescape(out)
        if decoded == out:
            break
        out = decoded
    out = _FMT_TAG_RE.sub("", out)
    return re.sub(r"\s+", " ", out).strip()


@dataclass
class PICOCriteria:
    """PICO framework for systematic review including I/E criteria."""
    population: str = ""
    intervention: str = ""
    comparator: str = ""
    outcome: str = ""
    inclusion_criteria: str = ""  
    exclusion_criteria: str = ""  
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for AI prompting and state management."""
        return {
            'p': self.population,
            'i': self.intervention,
            'c': self.comparator,
            'o': self.outcome,
            'inclusion': self.inclusion_criteria,
            'exclusion': self.exclusion_criteria
        }


@dataclass
class Paper:
    """Represents a research paper."""
    source: str
    id: str
    title: str
    abstract: str
    score: Optional[int] = None
    url: str = ""

    def __post_init__(self):
        # Normalise source markup once, so every downstream consumer (display,
        # title matching for full-text retrieval, screening, extraction) sees
        # clean text instead of "<i>…</i>" / "&lt;i&gt;".
        self.title = clean_markup(self.title)
        self.abstract = clean_markup(self.abstract)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "Source": self.source,
            "ID": self.id,
            "Title": self.title,
            "Abstract": self.abstract,
            "URL": self.url
        }
        if self.score is not None:
            result["Score"] = self.score
        return result


@dataclass
class ScreeningResult:
    """AI screening result for a paper based on I/E criteria."""
    decision: str = "ERROR"
    reason: str = "Failed"
    design: str = "N/A"
    sample_size: str = "N/A"
    risk_of_bias: str = "N/A"
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "design": self.design,
            "sample_size": self.sample_size,
            "risk_of_bias": self.risk_of_bias
        }