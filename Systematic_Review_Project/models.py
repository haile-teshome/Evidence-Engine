# ============================================================================
# FILE: models.py
# Data models and structures
# ============================================================================

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class PICOCriteria:
    """PICO framework for systematic review."""
    population: str = ""
    intervention: str = ""
    comparator: str = ""
    outcome: str = ""
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'p': self.population,
            'i': self.intervention,
            'c': self.comparator,
            'o': self.outcome
        }


@dataclass
class Paper:
    """Represents a research paper."""
    source: str
    id: str
    title: str
    abstract: str
    score: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "Source": self.source,
            "ID": self.id,
            "Title": self.title,
            "Abstract": self.abstract
        }
        if self.score is not None:
            result["Score"] = self.score
        return result


@dataclass
class ScreeningResult:
    """AI screening result for a paper."""
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


