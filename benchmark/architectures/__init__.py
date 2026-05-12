from .base import ScreeningArchitecture, ScreeningResult, ScreeningContext
from .single_combined import SingleCombined
from .cascade_triage import CascadeTriage
from .per_criterion_cot import PerCriterionCoT
from .decompose_match import DecomposeMatch
from .self_consistency import SelfConsistency
from .single_agent_tools import SingleAgentTools
from .multi_agent import MultiAgent
from .leads_native import LeadsNative
from .leads_multi_persona import LeadsMultiPersona
from .cascade_leads_strict import CascadeLeadsStrict
from .leads_native_fewshot import LeadsNativeFewshot

REGISTRY = {
    "single_combined": SingleCombined,
    "cascade_triage": CascadeTriage,
    "per_criterion_cot": PerCriterionCoT,
    "decompose_match": DecomposeMatch,
    "self_consistency": SelfConsistency,
    "single_agent_tools": SingleAgentTools,
    "multi_agent": MultiAgent,
    "leads_native": LeadsNative,
    "leads_multi_persona": LeadsMultiPersona,
    "cascade_leads_strict": CascadeLeadsStrict,
    "leads_native_fewshot": LeadsNativeFewshot,
}

__all__ = ["ScreeningArchitecture", "ScreeningResult", "ScreeningContext", "REGISTRY"]
