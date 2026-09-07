import re
import os
import json
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage 

from config import Config
from models import Paper, PICOCriteria, ScreeningResult

# utils.py

class AIService:
    @staticmethod
    def screen_single_paper(paper: Paper, pico: PICOCriteria, model_name: str, api_keys: dict) -> Tuple[str, ScreeningResult]:
        """
        Worker Agent: Screens one paper against PICO criteria.
        Returns a tuple of (paper_id, ScreeningResult).
        """
        # Pass API keys explicitly for thread safety in parallel execution
        model = AIService.get_model_with_keys(model_name, api_keys)
        if not model:
            return paper.id, ScreeningResult(decision="ERROR", reason="Model initialization failed")

        system_msg = SystemMessage(content="You are a clinical research screener. Evaluate if the study meets the Inclusion/Exclusion criteria.")
        
        prompt = f"""
        Screen the following paper based on the PICO framework and criteria.
        
        PICO: {pico.to_dict()}
        
        Paper Title: {paper.title}
        Abstract: {paper.abstract}
        
        Return a JSON object:
        {{
            "decision": "INCLUDE" or "EXCLUDE",
            "reason": "Brief explanation",
            "design": "Study design type",
            "sample_size": "N=...",
            "risk_of_bias": "High/Low/Unclear"
        }}
        """
        try:
            response = model.invoke([system_msg, HumanMessage(content=prompt)])
            data = AIService._extract_json(response.content)
            if data:
                return paper.id, ScreeningResult(**data)
        except Exception as e:
            return paper.id, ScreeningResult(decision="ERROR", reason=f"Worker error: {str(e)}")
        
        return paper.id, ScreeningResult(decision="EXCLUDE", reason="Failed to parse AI response")

    @staticmethod
    def _cloud_key(provider: str, explicit: str = "") -> str:
        """Resolve an LLM provider key, most specific first:
          1. explicit arg (thread-passed keys),
          2. per-request header (browser passphrase-encrypted mode, unlocked),
          3. OS keychain (device-keychain mode),
          4. backend environment variable.
        Keys are used only to construct the model; this backend never persists
        them beyond the OS keychain the user explicitly opted into."""
        from request_creds import get_cred
        import keychain
        env = {
            "openai": Config.OPENAI_API_KEY,
            "anthropic": Config.ANTHROPIC_API_KEY,
            "google": Config.GEMINI_API_KEY,
        }.get(provider, "")
        return (explicit or "").strip() or get_cred(provider) or keychain.get_key(provider) or env

    @staticmethod
    def _build_model(model_name: str, keys: dict | None = None):
        """Construct a chat model for `model_name`. Cloud providers (gpt/claude/
        gemini) require a key from the request or environment; anything else runs
        locally via Ollama and needs no key. Returns None if a required cloud key
        is missing, so callers surface an 'unavailable' message."""
        keys = keys or {}
        name_lower = (model_name or "").lower()
        ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            if "gpt" in name_lower:
                key = AIService._cloud_key("openai", keys.get("openai", ""))
                if not key:
                    print("[get_model] OpenAI API key not provided for", model_name)
                    return None
                return ChatOpenAI(model=model_name, api_key=key, temperature=0, seed=Config.RUN_SEED)
            if "claude" in name_lower:
                key = AIService._cloud_key("anthropic", keys.get("anthropic", ""))
                if not key:
                    print("[get_model] Anthropic API key not provided for", model_name)
                    return None
                return ChatAnthropic(model=model_name, api_key=key, temperature=0)
            if "gemini" in name_lower:
                key = AIService._cloud_key("google", keys.get("google", ""))
                if not key:
                    print("[get_model] Google API key not provided for", model_name)
                    return None
                return ChatGoogleGenerativeAI(model=model_name, api_key=key, temperature=0)
            # DEFAULT / LOCAL: Ollama (llama3, mistral, qwen, LEADS, …) — no key.
            # A fixed seed + temperature=0 make local runs deterministic/reproducible.
            # num_predict is a safety ceiling: high enough for any legitimate output
            # (PICO JSON, criteria, summaries, extraction) but low enough that a model
            # that fails to stop can't run away for tens of thousands of tokens and
            # wedge Ollama's single inference slot.
            return ChatOllama(model=model_name, temperature=0, base_url=ollama_base,
                              seed=Config.RUN_SEED, num_predict=4096)
        except Exception as e:
            print(f"[get_model] AI connection error for {model_name}: {e}")
            return None

    @staticmethod
    def _build_embedder(model_name: str | None = None, keys: dict | None = None):
        """Construct an embeddings model for the semantic-similarity layer.

        Mirrors _build_model: defaults to a LOCAL Ollama embedder
        (nomic-embed-text) so semantic search needs no key and nothing leaves the
        machine; an OpenAI embedding model (text-embedding-*) is used instead when
        named and a key is available. Returns None on failure so callers can fall
        back gracefully."""
        keys = keys or {}
        name = model_name or os.getenv("EMBED_MODEL", "nomic-embed-text")
        name_lower = name.lower()
        ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            if name_lower.startswith("text-embedding") or name_lower.startswith("openai:"):
                key = AIService._cloud_key("openai", keys.get("openai", ""))
                if not key:
                    print("[get_embedder] OpenAI key not provided for", name)
                    return None
                return OpenAIEmbeddings(model=name.split(":", 1)[-1], api_key=key)
            # DEFAULT / LOCAL: Ollama embeddings (nomic-embed-text, mxbai-embed-large, …)
            return OllamaEmbeddings(model=name, base_url=ollama_base)
        except Exception as e:
            print(f"[get_embedder] embedder error for {name}: {e}")
            return None

    @staticmethod
    def get_embedder(model_name: str | None = None):
        """Public embedder accessor (mirrors get_model)."""
        return AIService._build_embedder(model_name, None)

    @staticmethod
    def get_embedder_with_keys(model_name: str | None, keys: dict):
        """Thread-safe embedder initializer preferring an explicit keys dict."""
        return AIService._build_embedder(model_name, keys)

    @staticmethod
    def get_model_with_keys(model_name: str, keys: dict):
        """Thread-safe model initializer. Prefers the explicit `keys` dict, then
        per-request / environment keys. Safe to call from worker threads."""
        return AIService._build_model(model_name, keys)

    @staticmethod
    def get_model(model_name: str):
        """Initialize a model. Cloud keys come from the user's per-request headers
        (stored only in their browser) or backend environment variables."""
        return AIService._build_model(model_name, None)

    # @staticmethod
    # def _extract_json(text: str) -> Optional[Any]:
    #     """Robust JSON extraction to prevent 'AI Processing Errors'."""
    #     try:
    #         # Remove markdown blocks if present
    #         clean_text = text.replace("```json", "").replace("```", "").strip()
            
    #         # Find the actual JSON boundaries to ignore conversational 'chatter'
    #         start_idx = min(clean_text.find('{'), clean_text.find('['))
    #         end_idx = max(clean_text.rfind('}'), clean_text.rfind(']'))
            
    #         if start_idx != -1 and end_idx != -1:
    #             json_str = clean_text[start_idx:end_idx+1]
    #             return json.loads(json_str)
            
    #         return json.loads(clean_text)
    #     except Exception as e:
    #         st.warning(f"⚠️ JSON Parsing Error: AI returned invalid format.")
    #         return None
            
    @staticmethod
    def infer_pico_and_query(goal: str, model_name: str, previous_goal: str = "", prior: Optional[Dict[str, Any]] = None, framework: str = "pico") -> Dict[str, Any]:
        """Extract the question frame + criteria from the research goal.

        framework="pico" -> Population/Intervention/Comparator/Outcome (default).
        framework="pcc"  -> Population/Concept/Context (JBI scoping review).

        Core rule: never CONTRADICT what the user actually wrote. Preserve their
        literal phrasing for elements they specified. For elements they did NOT
        specify, infer a BROAD, INCLUSIVE default that does not conflict with the
        stated topic — never a narrow stereotype.
        """
        model = AIService.get_model(model_name)

        if (framework or "pico").lower() == "pcc":
            return AIService._infer_pcc(goal, model, prior)

        system_msg = SystemMessage(content=(
            "You are a clinical research methodologist helping a researcher scaffold "
            "a PICO for systematic review. Stay loyal to the topic the user wrote — "
            "preserve their literal phrasing for any element they specified, and "
            "never replace it with a different concept. For elements they did NOT "
            "specify, infer a BROAD, INCLUSIVE default that supports the stated topic. "
            "Never invent narrow specifics (age ranges, dosages, comorbidities, "
            "specific durations) the user did not state — those belong in optional "
            "clarifying questions, not in the inferred PICO."
        ))

        # When a previous strategy is supplied, this is a REFINEMENT — modify the
        # existing PICO/criteria per the user's request, preserving all other
        # operationalised detail instead of rebuilding (and losing) it.
        _has_prior = isinstance(prior, dict) and any(
            str(prior.get(k) or "").strip() for k in ("p", "i", "c", "o")
        )
        if _has_prior:
            _inc = "\n".join(f"    - {x}" for x in (prior.get("inclusion") or [])) or "    (none)"
            _exc = "\n".join(f"    - {x}" for x in (prior.get("exclusion") or [])) or "    (none)"
            _intro = f"""
EXISTING STRATEGY from the previous turn — REFINE it, do not rebuild it:
  Population:   {prior.get('p', '')}
  Intervention: {prior.get('i', '')}
  Comparator:   {prior.get('c', '')}
  Outcome:      {prior.get('o', '')}
  Inclusion criteria:
{_inc}
  Exclusion criteria:
{_exc}

The researcher now asks to MODIFY this strategy: "{goal}"

Apply ONLY the change the researcher asked for and KEEP EVERYTHING ELSE INTACT:
  • Preserve every operationalised detail from the existing strategy verbatim
    (age ranges, durations, validated scales, follow-up windows, comparator type,
    and the inclusion/exclusion criteria) UNLESS the requested change makes a
    specific element no longer valid.
  • Change only the element(s) the user named, plus any element that logically
    must change as a direct consequence. Leave all others exactly as written.
  • The refined PICO must be AT LEAST as detailed as the existing one — never
    drop specificity or regenerate generically.
  • Example: "cat ownership instead of pet ownership" → change the intervention's
    pet/companion-animal wording to the cat equivalent while keeping the same
    duration/operationalisation, and keep population, comparator, outcome, and
    all criteria unchanged.
"""
        else:
            _intro = f'Current Research Goal: "{goal}"'

        prompt = f"""
{_intro}

Extract these elements into a JSON object. RULES:

1. STATED elements — preserve the user's literal phrasing. Never paraphrase
   "Mediterranean diet" as "dietary intervention", or "longevity" as "BMI", or
   "cancer" as "neoplasm". The exact words the user wrote must appear.

2. UNSTATED elements — supply a BROAD, INCLUSIVE default that does not conflict
   with the stated topic. Prefer breadth over specificity.

3. NEVER override what the user did state. If the user wrote "Mediterranean diet",
   the intervention is "Mediterranean diet" — never "low-carb diet" or
   "diet intervention".

4. EVERY PICO field must be a SPECIFIC, OPERATIONALISED phrase (5–18 words),
   not a single bare word or a rambling list. Push for concreteness: name the
   subgroup, the dose / duration / adherence index, the comparator type, the
   validated outcome measure. The user's clarifying answers (if any) are folded
   into the goal text — use them. Examples of strong phrasings:
     • Population: "Community-dwelling adults aged 40+ with cardiovascular risk factors"
     • Population: "Older adults (65+ years) without established cardiovascular disease"
     • Intervention: "Adherence to a Mediterranean diet measured by validated index (e.g. MedDiet Score)"
     • Intervention: "Metformin monotherapy at standard daily doses (≥ 1000 mg/day)"
     • Comparator: "Standard Western diet or no specific dietary intervention"
     • Comparator: "Placebo or active control (sulfonylurea or DPP-4 inhibitor)"
     • Outcome: "All-cause mortality and lifespan, with follow-up ≥ 5 years"
     • Outcome: "Change in HbA1c (%) from baseline at 6 months or later"

5. Generate 5–7 inclusion criteria and 5–7 exclusion criteria. Each criterion
   should be a SPECIFIC, OPERATIONALISED sentence (8–22 words) — concrete
   enough that two reviewers screening the same paper would reach the same
   decision. Include numeric thresholds, validated scales, and design
   restrictions where reasonable. Cover, between the two lists, the standard
   methodological dimensions:
     • study design (RCT preference, prospective observational floor, follow-up duration)
     • population restriction (age, comorbidity, exposure window)
     • intervention / comparator operationalisation (dose, duration, adherence measure)
     • outcome measurement (validated scale, timing, hard vs surrogate endpoint)
     • methodological quality (sample size, attrition, blinding, language, full-text)
   Avoid circular boilerplate ("studies relevant to this research question"),
   and avoid generic one-word criteria ("English") — embed them in a sentence.

JSON shape:
{{
    "p": "Population — 5-18 word operationalised phrase",
    "i": "Intervention — verbatim if stated, plus operationalisation (5-18 words)",
    "c": "Comparator — 5-18 word operationalised phrase",
    "o": "Outcome — verbatim if stated, plus operationalisation (5-18 words)",
    "inclusion": ["5-7 specific inclusion criteria, each 8-22 words, operationalised"],
    "exclusion": ["5-7 specific exclusion criteria, each 8-22 words, operationalised"]
}}
"""
        try:
            response = model.invoke([system_msg, HumanMessage(content=prompt)])
            data = AIService._extract_json(response.content)
            
            if data:
                # Clean up any dictionary formatting and ensure strings
                def clean_item(item):
                    if isinstance(item, dict):
                        return str(list(item.values())[0]) if item.values() else str(item)
                    elif isinstance(item, str):
                        cleaned = re.sub(r"^[^:]*:\s*", "", item)  # Remove "key: " prefix
                        cleaned = re.sub(r"[{}]", "", cleaned)  # Remove braces
                        cleaned = re.sub(r"['\"]", "", cleaned)  # Remove quotes
                        return cleaned.strip()
                    return str(item).strip()
                
                inclusion = [clean_item(item) for item in data.get("inclusion", []) if item]
                exclusion = [clean_item(item) for item in data.get("exclusion", []) if item]

                def _clean_pico_value(v: Any) -> str:
                    """Normalise a PICO value. Strip pure placeholder strings — but
                    keep '(inferred) ...' markers so the UI can style them.
                    """
                    s = "" if v is None else str(v).strip()
                    if s.lower() in {"n/a", "na", "none", "not stated", "unspecified",
                                     "not specified", "any", "empty", "null"}:
                        return ""
                    return s

                return {
                    "p": _clean_pico_value(data.get("p") or data.get("population")),
                    "i": _clean_pico_value(data.get("i") or data.get("intervention")),
                    "c": _clean_pico_value(data.get("c") or data.get("comparator")),
                    "o": _clean_pico_value(data.get("o") or data.get("outcome")),
                    "inclusion": inclusion[:8],
                    "exclusion": exclusion[:8],
                }
        except Exception as e:
            print(f"PICO analysis error: {e}")
            pass

        # Fallback when the model failed entirely. Return blanks so the refinement
        # popup can prompt the user to fill in PICO explicitly, rather than
        # keyword-matching the goal into a stereotyped PICO value.
        return {
            "p": "",
            "i": "",
            "c": "",
            "o": "",
            "inclusion": [],
            "exclusion": [],
        }

    @staticmethod
    def _infer_pcc(goal: str, model, prior: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """PCC (JBI scoping review): extract Population / Concept / Context plus
        eligibility criteria. Returns keys population/concept/context/inclusion/
        exclusion (the caller maps Concept onto the search anchor)."""
        system_msg = SystemMessage(content=(
            "You are a research methodologist scaffolding a JBI scoping review using the "
            "PCC frame (Population, Concept, Context). Scoping reviews MAP the extent, range "
            "and nature of evidence rather than test an intervention's effect — so there is "
            "no comparator or outcome. Stay loyal to the user's wording for any element they "
            "stated; for unstated elements infer a BROAD, INCLUSIVE default. Never invent "
            "narrow specifics the user did not state."
        ))
        _has_prior = isinstance(prior, dict) and any(
            str(prior.get(k) or "").strip() for k in ("population", "concept", "context", "p", "i", "c")
        )
        if _has_prior:
            _inc = "\n".join(f"    - {x}" for x in (prior.get("inclusion") or [])) or "    (none)"
            _exc = "\n".join(f"    - {x}" for x in (prior.get("exclusion") or [])) or "    (none)"
            intro = f"""EXISTING SCOPING FRAME from the previous turn — REFINE it, do not rebuild it:
  Population: {prior.get('population') or prior.get('p', '')}
  Concept:    {prior.get('concept') or prior.get('i', '')}
  Context:    {prior.get('context') or prior.get('c', '')}
  Inclusion criteria:
{_inc}
  Exclusion criteria:
{_exc}

The researcher now asks to MODIFY this frame: "{goal}"
Apply ONLY the requested change and keep every other operationalised detail intact."""
        else:
            intro = f'Current Research Goal: "{goal}"'

        prompt = f"""
{intro}

Extract these elements for a scoping review into a JSON object. RULES:
1. STATED elements — preserve the user's literal phrasing verbatim.
2. UNSTATED elements — supply a BROAD, INCLUSIVE default (scoping reviews are wide by design).
3. Each element is a SPECIFIC phrase (5–18 words). Examples:
     • Population: "Community-dwelling older adults (65+), any comorbidity"
     • Concept: "Use of telehealth for chronic disease self-management"
     • Context: "Primary-care and community settings in high-income countries, 2010–present"
4. Generate 5–7 inclusion and 5–7 exclusion criteria (each 8–22 words, operationalised),
   covering study/source types eligible, population scope, concept scope, context limits,
   date/language, and evidence-source restrictions. Scoping reviews INCLUDE diverse source
   types (empirical studies, reviews, grey literature) — reflect that.

JSON shape:
{{
    "population": "Population — 5-18 word phrase",
    "concept": "Concept — the core idea/phenomenon/intervention being mapped (5-18 words)",
    "context": "Context — setting, geography, timeframe (5-18 words)",
    "inclusion": ["5-7 specific inclusion criteria"],
    "exclusion": ["5-7 specific exclusion criteria"]
}}
"""
        try:
            response = model.invoke([system_msg, HumanMessage(content=prompt)])
            data = AIService._extract_json(response.content)
            if data:
                def clean_item(item):
                    if isinstance(item, dict):
                        return str(list(item.values())[0]) if item.values() else str(item)
                    if isinstance(item, str):
                        c = re.sub(r"^[^:]*:\s*", "", item)
                        c = re.sub(r"[{}]", "", c)
                        return re.sub(r"['\"]", "", c).strip()
                    return str(item).strip()

                def _clean(v):
                    s = "" if v is None else str(v).strip()
                    if s.lower() in {"n/a", "na", "none", "not stated", "unspecified",
                                     "not specified", "any", "empty", "null"}:
                        return ""
                    return s
                return {
                    "population": _clean(data.get("population") or data.get("p")),
                    "concept": _clean(data.get("concept") or data.get("i")),
                    "context": _clean(data.get("context") or data.get("c")),
                    "inclusion": [clean_item(x) for x in data.get("inclusion", []) if x][:8],
                    "exclusion": [clean_item(x) for x in data.get("exclusion", []) if x][:8],
                }
        except Exception as e:
            print(f"PCC analysis error: {e}")
        return {"population": "", "concept": "", "context": "", "inclusion": [], "exclusion": []}

    @staticmethod
    def _pico_to_search_anchors(text: str) -> List[str]:
        """Extract clean, search-friendly noun phrases from a PICO field.

        The PICO inference prompt now asks for operationalised descriptions
        like "Adherence to a Mediterranean diet, measured by validated index
        (e.g., MedDiet Score)" — great for the user-facing card, terrible as a
        verbatim search anchor. This helper strips parentheticals, operationalisation
        suffixes, leading qualifiers, and criterion-like fragments, leaving only
        the core noun phrases ("Mediterranean diet").

        Returns up to 3 cleaned anchor phrases per field.
        """
        if not text:
            return []

        s = re.sub(r"^\s*\(inferred\)\s*", "", text, flags=re.IGNORECASE)

        # Strip parenthetical content like "(e.g., MedDiet Score)" or "(65+)".
        s = re.sub(r"\s*\([^)]*\)", "", s)

        # Strip operationalisation suffixes that turn a noun phrase into a sentence.
        for cut in [
            r"\s*[,;]?\s*measured by.*$",
            r"\s*[,;]?\s*defined (as|by).*$",
            r"\s*[,;]?\s*assessed (by|via).*$",
            r"\s*[,;]?\s*using.*$",
            r"\s*[,;]?\s*with follow.?up.*$",
            r"\s*[,;]?\s*at \d+\s*(months?|years?).*$",
            r"\s*[,;]?\s*over \d+\s*(months?|years?).*$",
            r"\s*[,;]?\s*from baseline.*$",
            r"\s*[,;]?\s*at (least|standard).*$",
        ]:
            s = re.sub(cut, "", s, flags=re.IGNORECASE)

        # Split compound concepts joined by " and " / " or " / commas / semicolons.
        pieces = re.split(r"\s+(?:and|or)\s+|,|;", s, flags=re.IGNORECASE)

        out: List[str] = []
        seen: set[str] = set()
        for p in pieces:
            p = re.sub(r"\s+", " ", p).strip(" ,.;:")
            # Strip leading "Adherence to a/the", "Use of the", etc.
            p = re.sub(
                r"^(adherence to|the use of|exposure to|presence of|use of)\s+(a |an |the )?",
                "", p, flags=re.IGNORECASE,
            )
            p = re.sub(r"^(a |an |the )", "", p, flags=re.IGNORECASE)
            if not p:
                continue
            # Reject criterion-like fragments (numeric thresholds, follow-up text).
            if re.search(r"≥|>=|≤|<=|\bfollow.?up\b|\d+\s*\+?\s*(years?|months?)", p, flags=re.IGNORECASE):
                continue
            # Reject overly long or overly short / generic items.
            if len(p) > 60 or len(p) < 3:
                continue
            if p.lower() in {"adults", "humans", "patients", "people", "subjects",
                              "participants", "general population", "outcome",
                              "outcomes", "intervention", "comparator"}:
                continue
            # Reject items that still contain stray parens or unbalanced quotes.
            if "(" in p or ")" in p or p.count('"') % 2 != 0:
                continue
            key = p.lower()
            if key not in seen:
                seen.add(key)
                out.append(p)
            if len(out) >= 3:
                break
        return out

    @staticmethod
    def generate_mesh_query(pico: PICOCriteria, model_name: str, goal: str = "",
                            ground_mesh: bool = False, return_concepts: bool = False,
                            seeds: Optional[List[Dict[str, Any]]] = None):
        """Generate a high-sensitivity PubMed search by decomposing the question
        into orthogonal concept blocks and AND-ing only the ones that narrow it.

        Strategy (validated on CLEF-TAR): a search is an AND of concept blocks,
        each an OR of synonyms + grounded MeSH for ONE concept. Precision comes
        from AND-ing the *discriminating* facets; broad population facets
        ("adults", "patients", "humans"), study-design facets ("RCT",
        "observational"), and pure context facets (setting, country, year) are
        dropped, since requiring them mostly deletes true hits without narrowing
        scope. Compound facets ("records with BOTH medical AND dental data") are
        split into separate blocks so the discriminating part (dental) stays
        required instead of being merged into one over-broad OR list.

        When ``ground_mesh`` is set, each kept concept's MeSH headings are
        validated / expanded against real NCBI MeSH. When ``return_concepts`` is
        set, returns ``(query, concepts)`` where concepts is the list of kept
        blocks as ``[{"name", "tiab", "mesh"}]``; otherwise returns just the
        query string. Every literal phrase the user wrote is kept verbatim.
        """
        model = AIService.get_model(model_name)

        def _strip_inferred_prefix(s: str) -> str:
            return re.sub(r"^\s*\(inferred\)\s*", "", s or "", flags=re.IGNORECASE)

        # ---- 1. Gather source facets from every populated PICO / PCC field ---
        # Works for both PICO (intervention/comparator/outcome) and PCC
        # (concept/context); population is shared. Each populated field is raw
        # material the LLM decomposes into orthogonal concepts below.
        raw_fields = [
            pico.intervention, getattr(pico, "concept", ""), pico.outcome,
            getattr(pico, "context", ""), pico.population,
        ]
        field_desc: List[str] = []
        for val in raw_fields:
            v = _strip_inferred_prefix(val or "").strip()
            if v:
                field_desc.append(v)
        source_lines = "\n".join(f"  - {d}" for d in field_desc) or "  - (none)"
        goal_line = f'\nReview question: "{goal.strip()}"' if (goal or "").strip() else ""

        # ---- 2. Decompose into orthogonal named concepts (plain synonyms) ----
        # The LLM only produces synonyms + MeSH labels; grounding and PubMed
        # syntax are assembled deterministically below.
        prompt = f"""You are an expert clinical search librarian building a HIGH-SENSITIVITY PubMed search.
Break the review below into its CORE SEARCH CONCEPTS and, for each concept, give a
comprehensive list of synonyms and the relevant MeSH headings.

Review elements:
{source_lines}{goal_line}

Rules:
- Identify 2 to 5 ORTHOGONAL concepts (distinct ideas that do not overlap).
- Split any compound facet into SEPARATE concepts. For example "records containing
  both medical and dental data" is TWO concepts: an "electronic health records"
  concept AND a "dental" concept. Never merge two distinct ideas into one list.
- Give each SPECIFIC / discriminating idea (a named domain, condition, test, data
  type, or intervention) its own concept. Include a broad population concept
  (e.g. "adults", "patients") only if the review truly restricts to it, and keep it
  as its own concept. Do NOT invent concepts for setting, country, or year.
- For each concept list 6 to 15 terms: the user's exact phrase(s) verbatim,
  singular/plural and hyphenation variants, British/American spellings, common
  abbreviations and their expansions, and closely related wording for the SAME
  idea. Stay on-concept; do not drift or over-broaden.
- Also give 1 to 4 MeSH controlled-vocabulary headings per concept when they exist.
- Use plain terms only: NO field tags, NO Boolean operators, NO quotation marks.

Return ONLY a JSON object of exactly this shape:
{{"concepts": [{{"name": "<short concept label>", "terms": ["...", "..."], "mesh": ["...", "..."]}}, "..."]}}

EXAMPLE for "AI/ML prediction models using electronic health records with both medical and dental data":
{{"concepts": [
  {{"name": "Artificial intelligence", "terms": ["artificial intelligence", "AI", "machine learning", "deep learning", "predictive model", "prediction model", "clinical decision support", "decision support system", "neural network"], "mesh": ["Artificial Intelligence", "Machine Learning", "Decision Support Systems, Clinical"]}},
  {{"name": "Electronic health records", "terms": ["electronic health record", "electronic health records", "EHR", "electronic medical record", "EMR", "electronic dental record", "health record"], "mesh": ["Electronic Health Records"]}},
  {{"name": "Dental", "terms": ["dental", "dentistry", "oral health", "odontology", "dental record", "periodontal", "oral medicine"], "mesh": ["Dentistry", "Oral Health", "Dental Records"]}}
]}}

Output ONLY the JSON object. No explanation, no markdown, no code fences."""

        def _clean_term_list(raw: Any) -> List[str]:
            out: List[str] = []
            if isinstance(raw, list):
                for x in raw:
                    if isinstance(x, str):
                        t = x.strip().strip('"').strip("'")
                        # Strip stray PubMed syntax fragments the model may leak.
                        t = re.sub(r"\[[A-Za-z]+\]", "", t)
                        t = re.sub(r"[\[\]]", "", t)
                        t = re.sub(r"\s+", " ", t).strip(" ,.;:")
                        if "(" in t or ")" in t:
                            continue
                        if re.search(r"≥|>=|≤|<=|\bfollow.?up\b|\d+\s*\+?\s*(years?|months?)",
                                      t, flags=re.IGNORECASE):
                            continue
                        if re.search(r"\b(measured by|defined by|assessed by|with follow|at baseline)\b",
                                      t, flags=re.IGNORECASE):
                            continue
                        if 2 <= len(t) <= 45:
                            out.append(t)
            seen: set = set()
            dedup: List[str] = []
            for t in out:
                k = t.lower()
                if k not in seen:
                    seen.add(k)
                    dedup.append(t)
            return dedup

        concepts: List[Dict[str, Any]] = []
        try:
            r = model.invoke([HumanMessage(content=prompt)])
            data = AIService._extract_json(r.content) or {}
            for c in (data.get("concepts") or []):
                if not isinstance(c, dict):
                    continue
                name = str(c.get("name", "")).strip()
                tiab = _clean_term_list(c.get("terms"))
                mesh = _clean_term_list(c.get("mesh"))
                if name or tiab or mesh:
                    concepts.append({"name": name or (tiab[0] if tiab else "Concept"),
                                     "tiab": tiab, "mesh": mesh})
        except Exception as e:
            print(f"Query concept decomposition error: {e}")

        # ---- 2b. Ground MeSH against real NCBI controlled vocabulary ---------
        # Validate the model's headings (drop hallucinations via a token-overlap
        # guard) and fold official entry terms into the free-text list.
        if ground_mesh:
            try:
                from data_services import PubMedService
            except Exception:
                PubMedService = None
            if PubMedService is not None:
                _stop = {"with", "and", "the", "for", "type", "adult", "adults", "care", "disease", "disorder"}
                def _toks(x: str) -> set:
                    return {t for t in re.findall(r"[a-z]{4,}", (x or "").lower())} - _stop
                for c in concepts:
                    candidates = list(c["mesh"])
                    if not candidates and c["tiab"] and len(c["tiab"][0].split()) <= 3:
                        candidates = [c["tiab"][0]]
                    if not candidates and c["name"] and len(c["name"].split()) <= 3:
                        candidates = [c["name"]]
                    if not candidates:
                        continue
                    headings: List[str] = []
                    entries: List[str] = []
                    seen_h: set = set()
                    for cand in candidates[:3]:
                        h, ent = PubMedService.mesh_lookup(cand)
                        if h and h.lower() not in seen_h and (_toks(cand) & _toks(h)):
                            seen_h.add(h.lower())
                            headings.append(h)
                            entries.extend(ent[:6])
                    if headings:
                        c["mesh"] = headings
                    have = {t.lower() for t in c["tiab"]}
                    for e in entries:
                        if e.lower() not in have and 2 <= len(e) <= 45 and "(" not in e:
                            c["tiab"].append(e)
                            have.add(e.lower())

        # ---- 2c. Seed-study feedback: broaden the concept a seed overlaps ----
        if seeds:
            _sw = {"with", "and", "the", "for", "type", "study", "studies", "trial", "trials",
                   "review", "using", "versus", "effect", "effects", "among", "between", "patients", "disease"}
            def _ct(terms: List[str]) -> set:
                s: set = set()
                for t in terms:
                    s |= {w for w in re.findall(r"[a-z]{4,}", (t or "").lower())} - _sw
                return s
            ctoks = {i: _ct(c["tiab"] + c["mesh"] + [c["name"]]) for i, c in enumerate(concepts)}
            for sd in seeds:
                for m in (sd.get("mesh") or []):
                    m = str(m).strip()
                    if not m:
                        continue
                    mt = {w for w in re.findall(r"[a-z]{4,}", m.lower())} - _sw
                    for i, c in enumerate(concepts):
                        if mt & ctoks[i]:
                            if m not in c["mesh"] and len(c["mesh"]) < 8:
                                c["mesh"].append(m)
                                ctoks[i] |= mt
                            break

        # ---- 3. Keep only discriminating concepts, then assemble -------------
        BROAD = re.compile(r"\b(population|participants?|patients?|subjects?|people|persons?|"
                           r"humans?|adults?|children|child|adolescents?|infants?|neonates?|"
                           r"elderly|aged|men|women|male|female|individuals?|cases?|controls?)\b",
                           re.IGNORECASE)
        DESIGN = re.compile(r"\b(study\s*type|study\s*design|design|methodolog\w*|trial|trials|"
                            r"rct|randomi\w*|observational|cohort|case[\s-]?control|"
                            r"cross[\s-]?sectional|prospective|retrospective|publication\s*type)\b",
                            re.IGNORECASE)
        CONTEXT = re.compile(r"\b(setting|settings|country|countries|region|regions|geograph\w*|"
                             r"year|years|date|dates|time\s*period|worldwide|global|nationwide)\b",
                             re.IGNORECASE)
        POP_TERM = re.compile(r"^(patients?|adults?|children|people|persons?|subjects?|"
                              r"individuals?|humans?|men|women|males?|females?|participants?)$",
                              re.IGNORECASE)

        def _discriminating(c: Dict[str, Any]) -> bool:
            name = c["name"] or ""
            if BROAD.search(name) or DESIGN.search(name) or CONTEXT.search(name):
                return False
            terms = c["tiab"]
            if terms:
                generic = sum(1 for t in terms if POP_TERM.match(t.strip()))
                if generic >= max(2, int(0.8 * len(terms))):
                    return False
            return True

        kept = [c for c in concepts if _discriminating(c)]
        if not kept:
            kept = list(concepts)          # never drop every concept
        kept = kept[:4]                    # cap the number of AND blocks

        def _build_block(c: Dict[str, Any]) -> str:
            terms: List[str] = []
            seen: set = set()
            for m in c["mesh"]:
                tag = f'"{m}"[Mesh]'
                if tag.lower() not in seen:
                    seen.add(tag.lower())
                    terms.append(tag)
            for t in c["tiab"]:
                tag = f'"{t}"[tiab]' if re.search(r"[\s\-/]", t) else f"{t}[tiab]"
                if tag.lower() not in seen:
                    seen.add(tag.lower())
                    terms.append(tag)
            return "(" + " OR ".join(terms) + ")" if terms else ""

        blocks = [b for b in (_build_block(c) for c in kept) if b]

        def _ret(q: str):
            if return_concepts:
                return q, [{"name": c["name"], "tiab": c["tiab"], "mesh": c["mesh"]} for c in kept]
            return q

        if blocks:
            return _ret(" AND ".join(blocks))

        # ---- 4. Last-resort fallback -----------------------------------------
        pop = (_strip_inferred_prefix(pico.population or "") or "adults").split()[:2]
        intv = (_strip_inferred_prefix(pico.intervention or getattr(pico, "concept", "") or "") or "treatment").split()[:2]
        pq = " OR ".join(f'"{t}"[tiab]' for t in pop)
        iq = " OR ".join(f'"{t}"[tiab]' for t in intv)
        return _ret(f"({pq}) AND ({iq})")

    @staticmethod
    def generate_adversarial_query(pico: PICOCriteria, model_name: str) -> str:
        """Generate an adversarial search query (deterministic assembly).

        Strategy: combine the user's intervention concept with a fixed set of
        "negative-finding" terms. The structure is always
            (intervention synonyms) AND (negative-finding terms) AND (outcome synonyms)
        which guarantees well-formed PubMed syntax regardless of model quality.
        """
        def _strip_inferred_prefix(s: str) -> str:
            return re.sub(r"^\s*\(inferred\)\s*", "", s or "", flags=re.IGNORECASE)

        intv = _strip_inferred_prefix(pico.intervention or "").strip()
        outc = _strip_inferred_prefix(pico.outcome or "").strip()

        # Fixed adversarial vocabulary — terms a methodologist would look for to
        # surface null, harmful, or contradictory findings.
        adverse_terms = [
            "no effect", "null finding", "ineffective", "no benefit", "no association",
            "harmful", "harm", "adverse", "increased risk", "worse outcome",
            "negative finding", "contradict", "no difference", "non-significant",
        ]
        adverse_block = "(" + " OR ".join(f'"{t}"[tiab]' for t in adverse_terms) + ")"

        def _wildcard_variant(token: str) -> Optional[str]:
            # Strip a common suffix (length 2-3) before appending * so the
            # wildcard actually broadens retrieval — "longevity" → "longev*",
            # "patients" → "patient*", "diabetes" → "diabet*".
            t = token.strip()
            if " " in t or len(t) < 6:
                return None
            for suf in ("ity", "ies", "ous", "ing", "ed", "es", "s", "e"):
                if t.lower().endswith(suf) and len(t) - len(suf) >= 4:
                    return t[: -len(suf)] + "*"
            return None

        blocks: List[str] = []
        if intv:
            intv_terms = [intv]
            wv = _wildcard_variant(intv)
            if wv:
                intv_terms.append(wv)
            blocks.append("(" + " OR ".join(f'"{t}"[tiab]' for t in intv_terms) + ")")
        blocks.append(adverse_block)
        if outc:
            outc_terms = [outc]
            wv = _wildcard_variant(outc)
            if wv:
                outc_terms.append(wv)
            blocks.append("(" + " OR ".join(f'"{t}"[tiab]' for t in outc_terms) + ")")

        if not blocks:
            return ""
        return " AND ".join(blocks)

    @staticmethod
    def propose_supplementary_query(
        pico: PICOCriteria,
        current_query: str,
        inclusion: List[str],
        exclusion: List[str],
        kept_titles: List[str],
        model_name: str,
    ) -> Dict[str, Any]:
        """Given the current search and the relevant studies already found,
        propose ONE supplementary boolean query that would surface additional
        eligible studies the first query likely missed. Powers the single
        adaptive round of the Home "deep scan": it does not replace the primary
        documented search, it complements it (logged as a supplementary search)."""
        model = AIService.get_model(model_name)
        titles_text = "\n".join(f"- {t}" for t in (kept_titles or [])[:15]) or "(none found yet)"
        incl = "; ".join(inclusion or []) or "(none specified)"
        excl = "; ".join(exclusion or []) or "(none specified)"
        prompt = f"""You are a systematic-review search specialist. A first search has already run and returned the relevant studies listed below. Propose ONE supplementary PubMed-style boolean query that would surface ADDITIONAL eligible studies the first query most likely MISSED — for example synonyms or spelling variants, a related intervention, an under-covered comparator, an adjacent population, or a study design not yet represented. Do not simply restate the original query.

PICO:
- Population: {pico.population or 'any'}
- Intervention: {pico.intervention or 'any'}
- Comparator: {pico.comparator or 'any'}
- Outcome: {pico.outcome or 'any'}

Eligibility (include): {incl}
Eligibility (exclude): {excl}

Original query:
{current_query}

Relevant studies already found:
{titles_text}

Return ONLY JSON in this shape:
{{"query": "<a well-formed boolean query>", "rationale": "<one sentence naming the coverage gap it targets>", "tactic": "<2-4 word label, e.g. 'synonym expansion'>"}}"""
        try:
            response = model.invoke([HumanMessage(content=prompt)])
            data = AIService._extract_json(response.content)
            if isinstance(data, dict) and str(data.get("query", "")).strip():
                return {
                    "query": str(data.get("query", "")).strip(),
                    "rationale": str(data.get("rationale", "")).strip(),
                    "tactic": str(data.get("tactic", "")).strip(),
                }
        except Exception as e:
            print(f"Supplementary query error: {e}")
        return {"query": "", "rationale": "", "tactic": ""}

    @staticmethod
    def _adapt_query_for_source(query: str, source: str) -> str:
        """Adapt a PubMed-style query to a target database's syntax.

        Three flavours:
          • PubMed: native syntax with [Mesh] / [tiab] / [ti] tags kept.
          • Europe PMC, OpenAlex, CrossRef, arXiv, bioRxiv, medRxiv, CORE,
            DOAJ: accept quoted phrases with AND/OR — strip the field tags.
          • Semantic Scholar: does NOT honour Boolean operators or quoted
            phrases meaningfully. We flatten to a plain space-separated
            keyword string: take the first synonym from each AND-joined
            concept block and join with spaces.
        """
        if not query:
            return query
        if source == "PubMed":
            return query

        if source == "Semantic Scholar":
            # Pick the first quoted phrase from each top-level concept block
            # and join with spaces. SS does best with 2-3 keyword anchors.
            blocks = AIService._split_concept_blocks(query)
            picks: List[str] = []
            for b in blocks:
                m = re.search(r'"([^"]+)"', b)
                if m:
                    picks.append(m.group(1))
            if picks:
                return " ".join(picks)
            # Fallback: strip all syntax markers and AND/OR keywords.
            text = re.sub(r"\[[A-Za-z]+\]", "", query)
            text = re.sub(r"\b(AND|OR|NOT)\b", " ", text)
            text = re.sub(r"[\"()]", "", text)
            return re.sub(r"\s+", " ", text).strip()

        # Everything else (Europe PMC, OpenAlex, CrossRef, arXiv, …):
        # strip [Mesh]/[tiab]/[ti] field tags but preserve the quoted phrases
        # and the AND/OR structure.
        stripped = re.sub(r"\[[A-Za-z]+\]", "", query)
        return re.sub(r"\s+", " ", stripped).strip()

    @staticmethod
    def _split_concept_blocks(query: str) -> List[str]:
        """Split a query of shape `(A) AND (B) AND (C)` into the top-level
        AND-joined concept blocks (preserving their outer parentheses)."""
        if not query:
            return []
        parts = re.split(r"\)\s*AND\s*\(", query.strip())
        # Re-add the outer parens that were eaten by the split.
        blocks: List[str] = []
        for i, p in enumerate(parts):
            if not p.startswith("(") and i == 0:
                p = "(" + p
            if not p.endswith(")") and i == len(parts) - 1:
                p = p + ")"
            if not p.startswith("("):
                p = "(" + p
            if not p.endswith(")"):
                p = p + ")"
            blocks.append(p)
        return blocks

    @staticmethod
    def _strip_mesh_only_terms(query: str) -> str:
        """Remove `"X"[Mesh] OR ` fragments so each concept block falls back
        to its `[tiab]` synonyms. Leaves [tiab] terms untouched."""
        if not query or "[Mesh]" not in query:
            return query
        # Cases: `"X"[Mesh] OR ` (at start of a block) — strip including OR.
        out = re.sub(r'"[^"]+"\[Mesh\]\s*OR\s+', "", query)
        # `OR "X"[Mesh]` (mid/end) — strip leading OR.
        out = re.sub(r'\s*OR\s+"[^"]+"\[Mesh\]', "", out)
        # Bare `"X"[Mesh]` left alone — that's the only term in its block.
        out = re.sub(r"\s+", " ", out).strip()
        return out

    @staticmethod
    def _retag(query: str, src_tag: str, dst_tag: str) -> str:
        """Replace one PubMed field tag with another (e.g. `[tiab]` → `[ti]`)."""
        return query.replace(src_tag, dst_tag)

    @staticmethod
    def _query_diff(prev: str, curr: str) -> Dict[str, List[str]]:
        """Compute the term-level diff between two queries.

        Returns a dict with `added` and `removed` lists, each containing the
        quoted terms (with their PubMed field tag if any) that appear in only
        one of the two queries.
        """
        if not prev or not curr or prev == curr:
            return {"added": [], "removed": []}
        # Capture `"term"[tag]` and bare `"term"` patterns separately so the
        # diff includes which field tag a term was searched under.
        def _terms(q: str) -> set[str]:
            tagged = set(re.findall(r'"[^"]+"\[[A-Za-z]+\]', q))
            return tagged
        prev_terms = _terms(prev)
        curr_terms = _terms(curr)
        return {
            "added": sorted(curr_terms - prev_terms),
            "removed": sorted(prev_terms - curr_terms),
        }

    @staticmethod
    def _tactic_variant(base_query: str, tactic_idx: int) -> Tuple[str, str]:
        """Apply tactic N to the base query, returning (variant_query, tactic_name).

        Tactic ladder — each is a deterministic transformation of the base
        query. Designed to span both broadening (more retrieval) and narrowing
        (more relevance) directions so the optimiser can discover the best
        operating point for each source.
        """
        if tactic_idx == 0:
            return base_query, "base query (multi-concept AND, full synonyms + MeSH)"

        blocks = AIService._split_concept_blocks(base_query)

        if tactic_idx == 1:
            # Drop the last AND-joined concept (usually population).
            if len(blocks) >= 3:
                return " AND ".join(blocks[:-1]), "drop population block"
            return base_query, "base query (only 2 concepts present)"

        if tactic_idx == 2:
            # Strip MeSH-only fragments so retrieval relies on [tiab].
            stripped = AIService._strip_mesh_only_terms(base_query)
            return stripped, "strip MeSH-only filters, keep [tiab] synonyms"

        if tactic_idx == 3:
            # Title-only matching for the FIRST concept block (intervention).
            if blocks:
                blocks[0] = AIService._retag(blocks[0], "[tiab]", "[ti]")
                blocks[0] = AIService._retag(blocks[0], "[Mesh]", "[ti]")
                return " AND ".join(blocks), "title-only matching on intervention"
            return base_query, "base query (no blocks to retag)"

        if tactic_idx == 4:
            # Drop both population and MeSH — narrowest broadening, keeps tiab.
            if len(blocks) >= 3:
                shrunk = " AND ".join(blocks[:-1])
            else:
                shrunk = base_query
            return AIService._strip_mesh_only_terms(shrunk), "drop population + strip MeSH"

        if tactic_idx == 5:
            # Title-only for ALL concepts (most specific variant).
            q = AIService._retag(base_query, "[tiab]", "[ti]")
            q = AIService._retag(q, "[Mesh]", "[ti]")
            return q, "title-only matching on all concepts"

        if tactic_idx == 6:
            # Keep only the FIRST synonym in each concept block. That's
            # usually the user's literal phrase, so this collapses to
            # `(literal-intervention) AND (literal-outcome) [AND (literal-pop)]`.
            new_blocks: List[str] = []
            for b in blocks:
                m = re.search(r'\(\s*("[^"]+"\[[A-Za-z]+\])', b)
                if m:
                    new_blocks.append(f"({m.group(1)})")
                else:
                    new_blocks.append(b)
            return " AND ".join(new_blocks), "keep only the user's literal phrase per concept"

        if tactic_idx == 7:
            # Drop the SECOND concept (outcome) — most aggressive broadening.
            # Useful when outcome terms (mortality, longevity) are too rare.
            if len(blocks) >= 2:
                kept = [blocks[0]] + (blocks[2:] if len(blocks) >= 3 else [])
                if kept:
                    return " AND ".join(kept), "drop outcome block, keep intervention"
            return base_query, "base query (only one concept present)"

        if tactic_idx == 8:
            # Title-only on intervention, [tiab] on outcome (asymmetric narrowing).
            if len(blocks) >= 2:
                int_block = AIService._retag(blocks[0], "[tiab]", "[ti]")
                int_block = AIService._retag(int_block, "[Mesh]", "[ti]")
                return " AND ".join([int_block] + blocks[1:]), "title-only intervention + [tiab] outcome"
            return base_query, "base query (no second concept)"

        # tactic_idx >= 9
        # Last tactic: drop synonyms entirely, just AND the user's literal
        # phrases together (most precise variant).
        new_blocks2: List[str] = []
        for b in blocks:
            m = re.search(r'"([^"]+)"\[[A-Za-z]+\]', b)
            if m:
                new_blocks2.append(f'("{m.group(1)}"[ti])')
            else:
                new_blocks2.append(b)
        if new_blocks2:
            return " AND ".join(new_blocks2), "literal phrases only, title-only matching"
        return base_query, "base query (last-resort fallback)"

    @staticmethod
    def agentic_optimize_per_source(
        current_query: str,
        pico: PICOCriteria,
        model_name: str,
        active_sources: List[str],
        research_goal: str = "",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Per-source query optimisation, refactored to mirror the Home page
        retrieval quality.

        Pipeline:
          1. Build ONE well-formed PubMed-style base query via
             `generate_mesh_query` (structured-JSON synonym expansion +
             deterministic syntax assembly).
          2. Per source, adapt the base query to that source's syntax
             (PubMed and Europe PMC keep tags; everything else strips them).
          3. Iterate up to N times per source: fetch up to 50 papers, score
             title relevance, if yield is low, deterministically broaden
             (drop the last AND-joined concept; strip MeSH-only filters).
          4. Track the best query per source by relevance, then by count.

        Returns the same shape the streaming endpoint and SimulationPage UI
        expect: `final_query`, `per_source_queries`, `trace[..].sources[..]`
        with `count`, `relevance_score`, `quality_rating`, `query`, `titles`,
        `iteration_reasoning`.
        """
        from data_services import DataAggregator

        model = AIService.get_model(model_name)
        if not model:
            return {
                "final_query": current_query,
                "trace": [],
                "per_source_queries": {source: current_query for source in active_sources},
                "error": "Model initialization failed",
            }

        # ---- 1. Build the well-formed base query ONCE -----------------------
        try:
            base_query = AIService.generate_mesh_query(pico, model_name, goal=research_goal)
        except Exception as e:
            print(f"[agentic] base query generation failed: {e}")
            base_query = current_query or ""

        # ---- 2. Per-source iteration with tactic ladder ---------------------
        # The optimiser walks a deterministic ladder of distinct tactics, each
        # producing a meaningfully different variant of the base query (drop
        # population, strip MeSH, title-only matching, literal-only, …).
        #
        # There is no fixed iteration cap. Each source is stopped individually
        # when EITHER:
        #   • 3 consecutive iterations failed to improve its best relevance, OR
        #   • every tactic in the ladder has been tried.
        # The loop terminates when every active source has stopped.
        # A safety ceiling prevents an infinite loop if the stop conditions
        # somehow fail to fire.
        tactic_count = 10      # number of distinct tactics in the ladder (see _tactic_variant)
        early_stop_after = 3   # stop a source after this many no-improvement iterations
        safety_max = 50        # hard ceiling against infinite loops
        # No client-side cap on retrieval — pass a very high ceiling and let
        # each source's natural API limit (PubMed retmax, OpenAlex per_page,
        # CrossRef rows, etc.) bind instead.
        per_source_max = 10000
        epsilon = 0.005        # minimum relevance delta that counts as an improvement

        # Per-source state
        best_queries: Dict[str, str] = {
            source: AIService._adapt_query_for_source(base_query, source)
            for source in active_sources
        }
        best_relevance: Dict[str, float] = {s: 0.0 for s in active_sources}
        best_count: Dict[str, int] = {s: 0 for s in active_sources}
        best_iter: Dict[str, int] = {s: 0 for s in active_sources}
        best_tactic: Dict[str, str] = {s: "" for s in active_sources}
        no_improve: Dict[str, int] = {s: 0 for s in active_sources}
        stopped: Dict[str, bool] = {s: False for s in active_sources}
        prev_iter_query: Dict[str, str] = {s: "" for s in active_sources}

        trace: List[Dict[str, Any]] = []
        # Per-source reason for stopping, surfaced in the trace.
        stop_reason: Dict[str, str] = {s: "" for s in active_sources}

        iteration = 0
        while iteration < safety_max:
            if all(stopped[s] for s in active_sources):
                break
            iter_data: Dict[str, Any] = {
                "iteration": iteration + 1,
                "sources": {},
                "total_papers": 0,
                "avg_relevance": 0,
                "status": "checking",
            }

            for source in active_sources:
                if stopped[source]:
                    # Carry forward the running best so the UI keeps showing
                    # this source instead of dropping it from the trace.
                    iter_data["sources"][source] = {
                        "query": best_queries[source],
                        "count": best_count[source],
                        "titles": [],
                        "relevance_score": round(best_relevance[source], 2),
                        "quality_rating": AIService._score_to_rating(best_relevance[source]),
                        "iteration_reasoning": (
                            stop_reason[source] or
                            f"stopped — reverted to best (iter {best_iter[source]})"
                        ),
                        "tactic": "(stopped)",
                        "query_diff": {"added": [], "removed": []},
                        "action": "stopped",
                        "stopped": True,
                        "best_so_far": {
                            "iteration": best_iter[source],
                            "tactic": best_tactic[source],
                            "query": best_queries[source],
                            "relevance_score": round(best_relevance[source], 2),
                            "count": best_count[source],
                        },
                    }
                    continue

                # If the ladder is exhausted, this source has nothing new left
                # to try — stop it and emit a final summary entry.
                if iteration >= tactic_count:
                    stopped[source] = True
                    stop_reason[source] = (
                        f"all {tactic_count} tactics exhausted; best at iter {best_iter[source]} "
                        f"(relevance {best_relevance[source]:.2f}, {best_count[source]} papers)"
                    )
                    iter_data["sources"][source] = {
                        "query": best_queries[source],
                        "count": best_count[source],
                        "titles": [],
                        "relevance_score": round(best_relevance[source], 2),
                        "quality_rating": AIService._score_to_rating(best_relevance[source]),
                        "iteration_reasoning": stop_reason[source],
                        "tactic": "(exhausted)",
                        "query_diff": {"added": [], "removed": []},
                        "action": "stopped",
                        "stopped": True,
                        "best_so_far": {
                            "iteration": best_iter[source],
                            "tactic": best_tactic[source],
                            "query": best_queries[source],
                            "relevance_score": round(best_relevance[source], 2),
                            "count": best_count[source],
                        },
                    }
                    continue

                # Apply the tactic for this iteration to get the next variant,
                # then adapt to the source's syntax.
                raw_variant, tactic_name = AIService._tactic_variant(base_query, iteration)
                query = AIService._adapt_query_for_source(raw_variant, source)

                source_result: Dict[str, Any] = {
                    "query": query,
                    "tactic": tactic_name,
                    "count": 0,
                    "titles": [],
                    "relevance_score": 0,
                    "quality_rating": "Poor",
                    "iteration_reasoning": "",
                    "query_diff": AIService._query_diff(prev_iter_query[source], query),
                    "action": "tested",
                    "stopped": False,
                }

                try:
                    papers, _ = DataAggregator.fetch_all(
                        query, [source], max_per_source=per_source_max, limit=per_source_max
                    )
                except Exception as e:
                    print(f"[agentic] {source} fetch error iter {iteration + 1}: {e}")
                    papers = []

                count = len(papers) if papers else 0
                titles = [p.title for p in papers] if papers else []
                # Score relevance against ALL retrieved titles (not a fixed
                # sample of 20). The cap was a remnant from when retrieval was
                # always 10–50 papers per source; with no client-side cap a
                # 20-title sample is too small to reflect the actual mix.
                relevance = (
                    AIService._analyze_title_relevance(titles, research_goal, pico)
                    if titles else 0.0
                )

                source_result.update({
                    "count": count,
                    "titles": titles[:10],
                    "relevance_score": round(relevance, 2),
                    "quality_rating": AIService._score_to_rating(relevance),
                })

                # Decide accept vs backtrack. Relevance is the optimisation
                # target; count is a tiebreaker at equal relevance.
                action: str
                if relevance > best_relevance[source] + epsilon:
                    action = "new_best"
                    best_relevance[source] = relevance
                    best_count[source] = count
                    best_queries[source] = query
                    best_iter[source] = iteration + 1
                    best_tactic[source] = tactic_name
                    no_improve[source] = 0
                elif (relevance >= best_relevance[source] - epsilon and
                      count > best_count[source]):
                    action = "tied_better_yield"
                    best_count[source] = count
                    best_queries[source] = query
                    best_iter[source] = iteration + 1
                    best_tactic[source] = tactic_name
                    # Tie on relevance doesn't reset the no-improve counter —
                    # we're still searching for a relevance improvement.
                    no_improve[source] += 1
                else:
                    action = "backtrack"
                    no_improve[source] += 1

                source_result["action"] = action

                # Snapshot of the running best AFTER this iteration's decision.
                source_result["best_so_far"] = {
                    "iteration": best_iter[source],
                    "tactic": best_tactic[source],
                    "query": best_queries[source],
                    "relevance_score": round(best_relevance[source], 2),
                    "count": best_count[source],
                }

                reasoning_bits = [
                    f"tactic: {tactic_name}",
                    f"count {count}",
                    f"relevance {relevance:.2f}",
                ]
                if action == "new_best":
                    reasoning_bits.append("↑ new best — adopted")
                elif action == "tied_better_yield":
                    reasoning_bits.append(
                        f"= relevance, more papers ({count}) → kept query, still searching"
                    )
                else:
                    reasoning_bits.append(
                        f"↓ below best ({best_relevance[source]:.2f} at iter {best_iter[source]}) "
                        f"→ backtrack; {no_improve[source]}/{early_stop_after} non-improvements"
                    )
                if no_improve[source] >= early_stop_after:
                    stopped[source] = True
                    stop_reason[source] = (
                        f"stopped after {early_stop_after} iterations without improvement; "
                        f"reverted to best (iter {best_iter[source]})"
                    )
                    reasoning_bits.append("stopping source — reverting to best")
                source_result["iteration_reasoning"] = "; ".join(reasoning_bits)

                iter_data["sources"][source] = source_result
                prev_iter_query[source] = query

                if progress_callback:
                    try:
                        progress_callback(
                            iteration=iteration + 1,
                            total=tactic_count,
                            source=source,
                            count=count,
                            relevance=relevance,
                            reasoning=source_result["iteration_reasoning"],
                        )
                    except Exception as e:
                        print(f"[agentic] progress callback error: {e}")

            # Aggregate metrics across sources for this iteration.
            active_results = [
                r for r in iter_data["sources"].values() if not r.get("stopped")
            ] or list(iter_data["sources"].values())
            scores = [r["relevance_score"] for r in active_results]
            counts = [r["count"] for r in active_results]
            iter_data["avg_relevance"] = round(sum(scores) / len(scores), 2) if scores else 0
            iter_data["total_papers"] = sum(counts)

            trace.append(iter_data)

            # Loop control: a `while` driven by per-source `stopped` flags.
            # Terminate when every source has stopped (either via the 3-no-
            # improvement rule above or the tactic-exhaustion check at the
            # top of the per-source block).
            if all(stopped[s] for s in active_sources):
                iter_data["status"] = "all sources converged or stopped"
                break
            iteration += 1
        else:
            # Loop exited because we hit the safety ceiling, not because
            # every source stopped naturally. Flag it on the last trace entry
            # so the UI can surface it as an anomaly worth investigating.
            if trace:
                trace[-1]["status"] = f"safety ceiling reached ({safety_max} iterations)"

        return {
            "final_query": best_queries.get(active_sources[0], base_query) if active_sources else base_query,
            "per_source_queries": best_queries,
            "trace": trace,
            "iterations_run": len(trace),
            "best_relevance": max([t.get("avg_relevance", 0) for t in trace]) if trace else 0,
            "total_papers_found": trace[-1].get("total_papers", 0) if trace else 0,
        }
    
    @staticmethod
    def _analyze_title_relevance(titles: List[str], research_goal: str, pico: PICOCriteria) -> float:
        """Analyze how relevant returned titles are to the research goal."""
        if not titles or not research_goal:
            return 0.0
        
        # Extract key terms from PICO and goal
        key_terms = []
        for field in [pico.population, pico.intervention, pico.outcome, research_goal]:
            if field:
                key_terms.extend([t.lower() for t in field.split() if len(t) > 2])
        
        # Remove stop words
        stop_words = {'the', 'and', 'or', 'of', 'in', 'to', 'for', 'with', 'a', 'an', 'is', 'are', 'was', 'were', 'on', 'at', 'by'}
        key_terms = [t for t in key_terms if t not in stop_words]
        
        if not key_terms:
            return 0.5
        
        # Score each title
        scores = []
        for title in titles:
            title_lower = title.lower()
            matches = sum(1 for term in key_terms if term in title_lower)
            score = min((matches / len(key_terms)) * 2, 1.0)  # Scale and cap
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0
    
    @staticmethod
    def _score_to_rating(score: float) -> str:
        """Convert relevance score to quality rating."""
        if score >= 0.8: return "Excellent"
        elif score >= 0.6: return "Good"
        elif score >= 0.4: return "Fair"
        elif score >= 0.2: return "Poor"
        else: return "Very Poor"

    @staticmethod
    def optimize_search_string_per_source(
        current_query: str,
        pico: PICOCriteria,
        model_name: str,
        active_sources: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Agentic AI approach to optimize search string individually for each database.
        Returns optimized queries and predicted yields per source.
        """
        model = AIService.get_model(model_name)
        if not model:
            return {source: {"query": current_query, "yield": 0} for source in active_sources}
        
        # Database-specific optimization strategies
        db_strategies = {
            "PubMed": "Use MeSH terms, [tiab] for title/abstract, proper PubMed boolean operators",
            "arXiv": "Simplify to keyword-focused, use OR for synonyms, minimal field tags",
            "Semantic Scholar": "Academic-focused, use phrase quotes, broader terms acceptable",
            "Europe PMC": "Similar to PubMed but more flexible with free text",
            "Google Scholar": "Simple keyword combinations, minimal syntax, natural language friendly"
        }
        
        results = {}
        
        for source in active_sources:
            strategy = db_strategies.get(source, "General academic database optimization")
            
            prompt = f"""
            You are an expert Information Specialist optimizing for {source}.
            
            CURRENT SEARCH STRING:
            {current_query}
            
            PICO CONTEXT:
            - Population: {pico.population}
            - Intervention: {pico.intervention}
            - Comparator: {pico.comparator}
            - Outcome: {pico.outcome}
            
            DATABASE-SPECIFIC STRATEGY for {source}:
            {strategy}
            
            OPTIMIZATION TASK:
            1. Adapt the query syntax specifically for {source}'s search engine
            2. Use appropriate field tags and operators for this database
            3. Maximize sensitivity (catch more papers) while maintaining relevance
            4. Expand with database-appropriate synonyms and variants
            
            CRITICAL: Return ONLY a JSON object in this exact format:
            {{"optimized_query": "the adapted search string", "estimated_yield": 123}}
            
            - optimized_query: The database-specific optimized search string
            - estimated_yield: Estimated number of papers this query would return (integer)
            """
            
            try:
                messages = [HumanMessage(content=prompt)]
                response = model.invoke(messages)
                content = response.content.strip()
                
                # Try multiple JSON extraction methods
                import re
                json_data = None
                
                # Method 1: Look for JSON object
                json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
                
                # Method 2: Try to parse the entire content as JSON
                if not json_data:
                    try:
                        json_data = json.loads(content)
                    except json.JSONDecodeError:
                        pass
                
                # Method 3: Try to extract key-value pairs manually
                if not json_data:
                    try:
                        # Look for optimized_query and estimated_yield patterns
                        query_match = re.search(r'["\']?optimized_query["\']?\s*[:=]\s*["\']([^"\']+)["\']', content, re.IGNORECASE)
                        yield_match = re.search(r'["\']?estimated_yield["\']?\s*[:=]\s*(\d+)', content, re.IGNORECASE)
                        
                        if query_match:
                            json_data = {
                                "optimized_query": query_match.group(1),
                                "estimated_yield": int(yield_match.group(1)) if yield_match else 0
                            }
                    except Exception:
                        pass
                
                if json_data:
                    results[source] = {
                        "query": json_data.get("optimized_query", current_query),
                        "yield": json_data.get("estimated_yield", 0)
                    }
                else:
                    print(f"Could not extract JSON from AI response for {source}")
                    print(f"AI response content: {content[:200]}...")
                    results[source] = {"query": current_query, "yield": 0}
                    
            except Exception as e:
                print(f"Optimization error for {source}: {e}")
                results[source] = {"query": current_query, "yield": 0}
        
        return results

    @staticmethod
    def optimize_search_string(
        current_query: str,
        pico: PICOCriteria,
        model_name: str,
        active_sources: List[str]
    ) -> str:
        """
        Agentic AI approach to optimize search string for maximum relevant papers.
        Uses iterative refinement to balance sensitivity and specificity.
        """
        model = AIService.get_model(model_name)
        if not model:
            return current_query
        
        prompt = f"""
        You are an expert Information Specialist performing an agentic search optimization.
        
        CURRENT SEARCH STRING:
        {current_query}
        
        PICO CONTEXT:
        - Population: {pico.population}
        - Intervention: {pico.intervention}
        - Comparator: {pico.comparator}
        - Outcome: {pico.outcome}
        
        TARGET DATABASES: {', '.join(active_sources)}
        
        OPTIMIZATION GOAL: Maximize the number of RELEVANT papers while minimizing noise.
        
        AGENTIC REFINEMENT STRATEGY:
        1. Analyze the current query for:
           - Missing synonyms for key terms
           - Overly restrictive operators
           - Missing MeSH terms
           - Incorrect field tags
        
        2. Apply these optimization rules:
           - Expand population terms with synonyms (e.g., "adults" → "adult" OR "adults" OR "aged")
           - Add alternative spellings and British/American variants
           - Include broader MeSH terms where appropriate
           - Add wildcard (*) for root words with multiple endings
           - Balance AND/OR operators to avoid too restrictive combinations
        
        3. ENSURE the query is HIGH-SENSITIVITY (broad) for screening:
           - Better to catch more and filter later than miss relevant papers
           - Use OR liberally within concept groups
           - Keep AND only between major concept groups (Pop/Int)
        
        CRITICAL: Return ONLY the optimized PubMed search string.
        NO explanations, NO markdown, NO commentary.
        Just the clean, executable search string.
        
        EXAMPLE OUTPUT FORMAT:
        ("Diabetes Mellitus"[Mesh] OR "diabetes"[tiab] OR "diabetic"[tiab] OR "T2DM"[tiab]) AND ("Metformin"[Mesh] OR "metformin"[tiab] OR "Glucophage"[tiab])
        """
        
        try:
            messages = [HumanMessage(content=prompt)]
            response = model.invoke(messages)
            optimized = response.content.strip()
            
            # Clean up the response
            clean_query = optimized.replace("```sql", "").replace("```", "").replace("`", "")
            clean_query = clean_query.strip()
            
            # Ensure it looks like a valid PubMed query
            if "[" in clean_query and ("(" in clean_query or "OR" in clean_query.upper()):
                return clean_query
            else:
                return current_query
                
        except Exception as e:
            print(f"Optimization error: {e}")
            return current_query

    @staticmethod
    def generate_comprehensive_summary(goal: str, papers: List[Paper], model_name: str) -> str:
        """
        Generates a comprehensive summary with:
        - Brief summaries of key references
        - Arguments supporting the research question
        - Arguments against the research question
        - Expanded reference list (10 papers instead of 5)
        """
        if not papers:
            return "⚠️ No papers found. Please adjust your research goal."

        model = AIService.get_model(model_name)
        
        # Use up to 10 papers for more comprehensive coverage
        subset = papers[:10]
        
        # Build detailed paper context with summaries
        paper_context = ""
        for idx, p in enumerate(subset):
            paper_context += f"""
Paper [{idx+1}]: {p.title}
Source: {p.source}
Abstract: {p.abstract[:400]}...
---
"""
        
        prompt = f"""
You are an expert systematic review analyst. Provide a comprehensive analysis of the research landscape.

RESEARCH GOAL: {goal}

LITERATURE CONTEXT ({len(subset)} papers analyzed):
{paper_context}

TASK: Create a structured analysis with the following sections:

1. RESEARCH LANDSCAPE OVERVIEW (2-3 sentences)
   - Brief synthesis of the current state of research on this topic
   - Identify gaps or consensus areas

2. ARGUMENTS SUPPORTING THE RESEARCH QUESTION
   - List 3-5 key points from the literature that SUPPORT the research goal
   - Cite papers using [1], [2], etc.
   - Focus on positive findings, beneficial outcomes, or established relationships

3. ARGUMENTS AGAINST/CHALLENGING THE RESEARCH QUESTION  
   - List 2-4 key points that CONTRADICT or CHALLENGE the research goal
   - Cite papers using [1], [2], etc.
   - Include null findings, conflicting results, or methodological concerns

OUTPUT FORMAT:
- Use plain text headers like "HEADER NAME:" (NO markdown ** or code blocks)
- Structure your response with clear section headers
- Do not include a reference list - that will be added separately.
"""
        
        try:
            response = model.invoke([HumanMessage(content=prompt)])
            ai_analysis = response.content.strip()

            # Convert markdown formatting to HTML so it renders correctly inside HTML divs.
            # Streamlit does NOT process markdown inside HTML blocks, so we must pre-convert.
            def md_to_html(text: str) -> str:
                # Strip any accidental triple-backtick code fences the LLM may produce
                text = re.sub(r'```[a-z]*\n?', '', text)
                # Bold: **text** or __text__
                text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
                text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
                # Italic: *text* or _text_
                text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
                text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
                # Numbered list items (1. text)
                text = re.sub(r'(?m)^\d+\.\s+(.+)', r'<li>\1</li>', text)
                # Bullet list items (- text or * text)
                text = re.sub(r'(?m)^[-*]\s+(.+)', r'<li>\1</li>', text)
                # Wrap consecutive <li> blocks in <ol>/<ul>
                text = re.sub(r'(<li>.*?</li>\n*)+', lambda m: '<ul style="margin:8px 0 8px 20px;">' + m.group(0) + '</ul>', text, flags=re.DOTALL)
                # Plain section headers like "HEADER NAME:"
                text = re.sub(r'(?m)^([A-Z][A-Z /&-]{3,}):[ \t]*$', r'<p style="font-weight:600;margin-top:14px;margin-bottom:4px;">\1:</p>', text)
                # Newlines to <br> for remaining plain paragraphs
                text = re.sub(r'\n{2,}', '</p><p style="margin:8px 0;">', text)
                text = text.replace('\n', '<br>')
                return f'<p style="margin:8px 0;">{text}</p>'

            ai_analysis_html = md_to_html(ai_analysis)

            # Build expanded reference list (10 papers)
            ref_list = ""
            for idx, p in enumerate(subset):
                ref_list += f'<div><a href="{p.url}" target="_blank"><strong>[{idx+1}]</strong> {p.title}</a></div>\n'

            full_content = f"""
            <div class="summary-box">
                <div class="summary-title">Comprehensive Literature Analysis</div>
                <div style="font-size:0.97em; color:#222; line-height:1.7;">
                    {ai_analysis_html}
                </div>
                <hr style="margin: 20px 0; border: 0; border-top: 1px solid #ddd;">
                <div style="font-size: 0.9em; color: #333; font-weight: 600; margin-bottom: 10px;">
                    Key References
                </div>
                <div style="font-size: 0.85em; color: #555; line-height: 1.6;">
                    {ref_list}
                </div>
            </div>
            """
            return re.sub(r'(?m)^\s+', '', full_content).strip()
            
        except Exception as e:
            print(f"Comprehensive summary error: {e}")
            # Fallback to simple summary
            return AIService.generate_brainstorm_summary(goal, papers, model_name)

    @staticmethod
    def generate_brainstorm_summary(goal: str, papers: List[Paper], model_name: str) -> str:
        """Refines research goal and provides a list of references in a clean box."""
        if not papers:
            return "⚠️ No papers found. Please adjust your research goal."

        model = AIService.get_model(model_name)
        subset = papers[:5]
        
        paper_context = ""
        for idx, p in enumerate(subset):
            paper_context += f"Source [{idx+1}]: {p.title}\nAbstract: {p.abstract[:300]}...\n\n"
        
        prompt = f"""
        Research Goal: {goal}
        Literature Context:
        {paper_context}
        
        TASK: Write an detailed evaluation in 5-6 sentences of how these papers support or refine the goal.
        - Cite them using numerical markers like [1] or [2]. 
        - Dont include the references or create a reference section in your summary, it will be added at the end.
        """
        
        try:
            response = model.invoke([HumanMessage(content=prompt)])
            ai_prose = response.content.strip()

            ref_list = ""
            for idx, p in enumerate(subset):
                ref_list += f"\n* **[{idx+1}]** [{p.title}]({p.url})"

            full_content = f"""
            <div class="summary-box">
                <div class="summary-title">Literature Context</div>
                {ai_prose}
                <hr style="margin: 15px 0; border: 0; border-top: 1px solid #eee;">
                <div style="font-size: 0.85em; color: #555; font-weight: bold; margin-bottom: 5px;">Key References:</div>
                {ref_list}
            </div>
            """
            return full_content.strip()
            
        except Exception as e:
            return f"Literature synthesis unavailable: {str(e)}"
            

    @staticmethod
    def get_refinement_suggestions(goal: str, papers: List[Paper], model_name: str) -> List[str]:
        """Generates specific, intelligent refinement options based on found literature."""
        model = AIService.get_model(model_name)
        
        # Analyze the found papers to understand current search scope
        paper_analysis = []
        for p in papers[:10]:
            paper_analysis.append({
                'title': p.title,
                'abstract': p.abstract[:300] if p.abstract else '',
                'source': p.source
            })
        
        analysis_text = "\n".join([
            f"Paper {i+1}: {pa['title']}" 
            f"Source: {pa['source']}" 
            f"Abstract: {pa['abstract']}"
            for i, pa in enumerate(paper_analysis)
        ])
        
        prompt = f"""
        You are an expert systematic reviewer and research methodologist. Analyze the research goal and current literature to provide intelligent refinement suggestions.
        
        RESEARCH GOAL:
        {goal}
        
        CURRENT LITERATURE LANDSCAPE:
        {analysis_text}
        
        TASK: Generate 4-5 specific, actionable refinement suggestions that will improve the precision and relevance of this systematic review.
        
        ANALYSIS CONSIDERATIONS:
        1. Population scope: Are we too broad/narrow? Consider age groups, conditions, settings
        2. Intervention specificity: Should we focus on specific drugs, therapies, or approaches?
        3. Outcome precision: Can we target more specific clinical endpoints or measurements?
        4. Study design: Should we limit to RCTs, meta-analyses, or observational studies?
        5. Timeframe: Should we specify publication date ranges or follow-up periods?
        6. Geographic scope: Should we focus on specific regions or healthcare settings?
        7. Comparison clarity: Can we better define control or alternative treatments?
        
        SUGGESTION CRITERIA:
        - Be specific and actionable (not generic)
        - Target common systematic review refinement strategies
        - Consider PICO framework improvements
        - Address potential gaps in current search
        - Suggest methodological filters
        - Include clinical or demographic specifics
        - Consider intervention delivery mechanisms
        - Target outcome measurement specificity
        
        OUTPUT FORMAT:
        Return ONLY a JSON list of strings, each being a specific refinement suggestion.
        
        EXAMPLES OF GOOD SUGGESTIONS:
        - "Focus on randomized controlled trials published after 2015"
        - "Limit to adult patients with type 2 diabetes (>18 years)"
        - "Specify metformin dosage ranges (500-2000mg daily)"
        - "Include studies with minimum 12-month follow-up period"
        - "Target HbA1c reduction as primary outcome"
        
        Generate 4-5 specific refinement suggestions for this research goal:
        """
        
        try:
            response = model.invoke([HumanMessage(content=prompt)])
            suggestions = AIService._extract_json(response.content)
            return suggestions if isinstance(suggestions, list) else []
        except Exception as e:
            print(f"Refinement suggestion error: {e}")
            # Fallback to more intelligent defaults
            return [
                "Focus on specific study designs (e.g., randomized controlled trials only)",
                "Narrow population characteristics (e.g., specific age groups or conditions)",
                "Specify intervention details (e.g., dosage, duration, administration)",
                "Define precise outcome measures (e.g., specific clinical endpoints)",
                "Add timeframe constraints (e.g., publication date or follow-up period)"
            ]

    @staticmethod
    def _extract_json(text: str) -> Optional[Any]:
        """
        Single unified helper to find JSON lists or objects.
        Handles 'chatter' before/after JSON and markdown blocks.
        """
        if not text or not isinstance(text, str):
            return None
            
        clean_text = text.replace("```json", "").replace("```", "").strip()
        # First structural character of a JSON object/array.
        starts = [i for i in (clean_text.find("{"), clean_text.find("[")) if i != -1]
        if not starts:
            try:
                return json.loads(clean_text)
            except Exception as e:
                print(f"❌ JSON Parsing Error: {e} | Raw Text: {text[:100]}...")
                return None
        start_idx = min(starts)
        # 1) Parse the FIRST complete JSON value and ignore any trailing chatter
        #    or extra objects the model appended (raw_decode stops at the end of
        #    the first value, so 'Extra data' after a valid object no longer fails).
        try:
            obj, _end = json.JSONDecoder().raw_decode(clean_text, start_idx)
            return obj
        except Exception:
            pass
        # 2) Fallback: first structural char to the last matching close bracket.
        end_idx = max(clean_text.rfind("}"), clean_text.rfind("]"))
        if end_idx > start_idx:
            try:
                return json.loads(clean_text[start_idx:end_idx + 1])
            except Exception as e:
                print(f"❌ JSON Parsing Error: {e} | Raw Text: {text[:100]}...")
                return None
        return None
    # @staticmethod
    # def _extract_json(text: str) -> Optional[Any]:
    #     """Enhanced helper to find JSON lists or objects."""
    #     try:
    #         clean_text = text.replace("```json", "").replace("```", "").strip()
    #         start_idx = min(clean_text.find('{'), clean_text.find('['))
    #         end_idx = max(clean_text.rfind('}'), clean_text.rfind(']'))
            
    #         if start_idx != -1 and end_idx != -1:
    #             json_str = clean_text[start_idx:end_idx+1]
    #             return json.loads(json_str)
    #         return json.loads(clean_text)
    #     except Exception:
    #         return None

    # @staticmethod
    # def screen_paper(paper: Paper, pico: PICOCriteria, model_name: str, inclusion: List[str], exclusion: List[str]) -> dict:
    #     """Forcefully extracts Design, Sample Size, and Decision Reason."""
    #     model = AIService.get_model(model_name)
        
    #     prompt = f"""
    #     Strict Systematic Review Screening Task.
        
    #     GOAL: Determine if this paper fits the PICO and extract study metadata.
        
    #     PICO:
    #     - Pop: {pico.population} | Int: {pico.intervention} | Comp: {pico.comparator} | Out: {pico.outcome}
        
    #     RULES:
    #     - Inclusion: {inclusion}
    #     - Exclusion: {exclusion}
        
    #     PAPER:
    #     - Title: {paper.title}
    #     - Abstract: {paper.abstract}
        
    #     YOU MUST PROVIDE THESE 4 FIELDS:
    #     1. "Decision": Either "Include" or "Exclude".
    #     2. "Design": Specific study type (e.g. RCT, Cohort, Case-Control).
    #     3. "Sample_size": The number of subjects (e.g. N=200).
    #     4. "Reason": Why it was included or the specific criteria it failed.

    #     Return ONLY a JSON object. No intro, no outro.
    #     {{
    #         "Decision": "",
    #         "Reason": ""
    #     }}
    #     """
    #     try:
    #         response = model.invoke([HumanMessage(content=prompt)])
    #         data = AIService._extract_json(response.content)
            
    #         if data:
    #             return {
    #                 "decision": data.get('Decision', 'Exclude'),
    #                 "reason": data.get('Reason', 'Check criteria')
    #             }
    #     except Exception as e:
    #         print(f"Screening error: {e}")
            
    #     return {
    #         "decision": "Exclude", 
    #         "reason": "Error parsing response", 
    #     }
    @staticmethod
    def generate_search_query(pico: PICOCriteria, model_name: str) -> str:
        """
        Generates an optimized, clean Boolean search string.
        Ensures the output is ready for API consumption without hallucinated formatting.
        """
        model = AIService.get_model(model_name)
        
        prompt = f"""
        Target: Medical Literature Database (PubMed/ArXiv)
        Task: Convert the following PICO criteria into a professional Boolean search string.
        
        PICO Data:
        - Population: {pico.population}
        - Intervention: {pico.intervention}
        - Comparator: {pico.comparator}
        - Outcome: {pico.outcome}
        
        Formatting Rules:
        1. Use [Mesh] tags for recognized medical terms if applicable.
        2. Combine concepts with AND, synonyms with OR.
        3. Use parentheses for grouping logic.
        4. RETURN ONLY THE STRING. No backticks, no "Search Query:", no explanations.
        """
        
        try:
            response = model.invoke([HumanMessage(content=prompt)])
            raw_content = response.content.strip()
            clean = re.sub(r'^(Query|Search Query|PubMed Search String):\s*', '', raw_content, flags=re.IGNORECASE)
            clean = clean.replace("```", "").replace("`", "").replace('"', '').strip()
            
            if len(clean) < 5:
                return f"({pico.population}) AND ({pico.intervention})"
                
            return clean

        except Exception as e:
            return f"({pico.population}) AND ({pico.intervention})"


    @staticmethod
    def get_pico_suggestion(goal: str, element: str) -> List[str]:
        """Generates 3 REAL clinical refinements based on the specific research goal."""
        model = AIService.get_model(Config.DEFAULT_MODEL)
        if not model: return ["Model Error", "Check", "Config"]

        # Determine clinical context for the element
        context_hints = {
            "population": "subgroups, age ranges, or specific comorbidities",
            "intervention": "dosages, specific drug classes, or delivery methods",
            "comparator": "standard of care, specific placebos, or active controls",
            "outcome": "validated scales, mortality metrics, or specific biomarkers"
        }
        hint = context_hints.get(element.lower(), "clinical specifics")

        prompt = f"""
        You are a Clinical Research Methodologist. 
        Research Goal: "{goal}"
        
        TASK: Suggest 3 actual clinical ways to narrow the "{element}" for a systematic review.
        
        STRICT FORMATTING RULES:
        1. Return ONLY a JSON list of strings. Example: ["Term 1", "Term 2", "Term 3"]
        2. NO conversational text. NO introductory remarks.
        3. Each suggestion must be 2-5 words.
        
        STRICT CONTENT RULES:
        - DO NOT use the word "{element}" in your suggestions.
        - DO NOT use generic words like "Specific", "Targeted", or "Refined".
        - Provide ACTUAL clinical {hint} relevant to the Goal.
        
        Example for Goal 'Diabetes treatment': 
        ["HbA1c reduction > 1%", "Type 2 Adults (BMI > 30)", "Metformin Monotherapy"]
        """
        
        try:
            response = model.invoke([HumanMessage(content=prompt)])
            data = AIService._extract_json(response.content)
            
            if isinstance(data, list) and len(data) > 0:
                return [str(s).strip() for s in data[:3]]
            
            if isinstance(data, dict):
                vals = list(data.values())
                return [str(v).strip() for v in vals[:3]]

        except Exception as e:
            print(f"Suggestion Error: {e}")
            
        fallbacks = {
            "population": ["Adults aged 18-65", "Chronic patients", "Acute settings"],
            "intervention": ["Combined therapy", "Monotherapy", "Standard dosage"],
            "comparator": ["Placebo control", "Standard of care", "Active comparator"],
            "outcome": ["Primary clinical endpoint", "Quality of life", "Adverse events"]
        }
        return fallbacks.get(element.lower(), ["Option A", "Option B", "Option C"])

    # @staticmethod
    # def generate_summary(goal_text: str) -> str:
    #     """Generates a short title for the sidebar history."""
    #     # If no LLM is ready, just return a snippet
    #     if not goal_text: return "New Session"
        
    #     # Simple prompt to your model
    #     prompt = f"Summarize this research goal in 3-5 words: {goal_text}"
    #     # Example call (pseudo-code):
    #     # response = model.invoke(prompt)
    #     # return response.content
    #     return goal_text[:30] + "..." # Fallback

    @staticmethod
    def generate_summary(text: str) -> str:
        """
        Creates a short label for the sidebar history.
        Trims the research goal to a readable length.
        """
        if not text:
            return "New Investigation"
        
        clean_text = text.replace("\n", " ").strip()
        if len(clean_text) > 30:
            return clean_text[:30] + "..."
        return clean_text



    @staticmethod
    def generate_formal_question(pico: PICOCriteria, model_name: str, history: list, goal: str = "") -> str:
        """Compose the formal research question/objective FAITHFULLY from the frame
        elements and the researcher's stated goal. Framework-aware: a PICO effect
        question, or a PCC scoping objective. Deliberately anti-drift — it must not
        introduce concepts the elements/goal don't contain."""
        model = AIService.get_model(model_name)

        past_questions = [h['formal_question'] for h in history if 'formal_question' in h]
        history_context = "\n".join([f"- Iteration {i+1}: {q}" for i, q in enumerate(past_questions)])

        import frameworks
        fw = frameworks.normalize(getattr(pico, "framework", "pico"))
        frame_meta = frameworks.framework_of(fw)
        elems = "\n".join(f"        - {label}: {val or '(not specified)'}" for _id, label, val in pico.element_items())
        if fw == "pcc":
            structure = ('Phrase it as a scoping-review OBJECTIVE that maps the concept, e.g. '
                         '"This scoping review maps [Concept] among [Population] in [Context]." '
                         'A scoping review does NOT test an effect — never use "does/compared to/result in".')
        else:
            structure = ('Use the structure "In [Population], does [Intervention] compared to [Comparator] '
                         'affect [Outcome]?" (adapt the verb naturally, but keep all four elements).')

        prompt = f"""
        You are an expert research librarian composing the formal {"objective" if fw == "pcc" else "question"} for a review.
        It MUST be faithful to the frame elements and the researcher's stated goal below — this is the
        single most important rule.

        RESEARCHER'S STATED GOAL (source of truth for topic and wording):
        "{goal or '(not provided — rely on the frame elements)'}"

        {frame_meta['label']} ELEMENTS (use these values; do not substitute or paraphrase the core concepts):
{elems}

        PREVIOUS ITERATIONS (keep consistent with these; do not drift the topic):
        {history_context if history_context else "None (this is the first draft)"}

        RULES:
        - Build the {"objective" if fw == "pcc" else "question"} DIRECTLY from the element values above; every
          specified element must appear, using the element's own wording (light grammatical smoothing only).
        - Do NOT introduce any population, intervention, comparator, outcome, concept, or context that is not
          present in the elements or the stated goal. Do NOT invent ages, doses, timeframes, settings, or
          subgroups that were not given.
        - Stay on the researcher's exact topic — never swap a concept for a related-but-different one.
        - {structure}
        - One sentence, no preamble or filler.

        Return ONLY the {"objective" if fw == "pcc" else "question"}.
        """

        try:
            from langchain_core.messages import HumanMessage
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content.strip().strip('"')
        except Exception:
            if fw == "pcc":
                return f"This scoping review maps {pico.concept} among {pico.population} in {pico.context}."
            return f"In {pico.population}, what is the effect of {pico.intervention} vs {pico.comparator} on {pico.outcome}?"


    # @staticmethod
    # def screen_paper(paper: Paper, pico: PICOCriteria, model_name: str) -> Dict[str, Any]:
    #     model = AIService.get_model(model_name)
        
    #     criteria_options = f"Inclusion: {pico.inclusion_criteria}\nExclusion: {pico.exclusion_criteria}"
        
    #     prompt = f"""
    #     Strictly screen this paper based on PICO and I/E Criteria.
        
    #     CRITERIA:
    #     {criteria_options}

    #     PAPER:
    #     Title: {paper.title}
    #     Abstract: {paper.abstract}

    #     TASK:
    #     1. Decision: "Include" or "Exclude".
    #     2. Bucket: Select 3-5 words from the criteria above that best explains the decision (e.g., "Wrong Population", "Study Design", "No Comparator").
    #     3. Reason: Brief explanation.

    #     RETURN ONLY JSON:
    #     {{"decision": "Exclude", "bucket": "Wrong Population", "reason": "Focuses on children, not adults."}}
    #     """
        
    #     try:
    #         from langchain_core.messages import HumanMessage
    #         response = model.invoke([HumanMessage(content=prompt)])
            
    #         raw_content = response.content
    #         json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
    #         if json_match:
    #             data = json.loads(json_match.group())
    #             return {
    #                 "decision": data.get('decision', 'Exclude'),
    #                 "reason": data.get('reason', 'Criteria mismatch'),
    #                 "citation": data.get('citation', 'Reference found in text')
    #             }
    #     except Exception as e:
    #         print(f"DEBUG: AI Processing error: {e}")
            
    #     return {"decision": "Exclude", "reason": "AI Processing Timeout", "citation": "Check manually"}

    @staticmethod
    def screen_paper(paper: Paper, pico: PICOCriteria, model_name: str, inclusion: List[str] = None, exclusion: List[str] = None, protocol: str = "") -> Dict[str, Any]:
        """Strictly screen this paper based on the review plan, PICO, and Inclusion/Exclusion Criteria."""
        model = AIService.get_model(model_name)
        if not model:
            # Use keyword matching as primary fallback when model fails
            title_lower = paper.title.lower()
            abstract_lower = paper.abstract.lower()
            pico_terms = []
            
            if pico.population:
                pico_terms.extend(pico.population.lower().split())
            if pico.intervention:
                pico_terms.extend(pico.intervention.lower().split())
            
            # Check if any PICO terms are present (more robust matching)
            matches = 0
            for term in pico_terms:
                if len(term) > 2:  # Skip very short terms
                    if term in title_lower or term in abstract_lower:
                        matches += 1
            
            if matches >= 2:  # If at least 2 terms match, include
                return {"decision": "Include", "bucket": "Keyword match", "reason": f"Found {matches} matching PICO terms"}
            else:
                return {"decision": "Exclude", "bucket": "No match", "reason": f"Only {matches} PICO terms found"}
        
        # Ensure we have criteria text to send to the AI
        inc_text = inclusion if inclusion else getattr(pico, 'inclusion_criteria', "None specified")
        excl_text = exclusion if exclusion else getattr(pico, 'exclusion_criteria', "None specified")
        # The review plan/protocol from the planning stage, when present, grounds the
        # decision beyond the bulleted criteria. Bounded so a long plan never dominates.
        proto_text = (protocol or "").strip()
        proto_block = (
            f"\n        REVIEW PLAN / PROTOCOL (screen consistently with this plan):\n        {proto_text[:4000]}\n"
            if proto_text else ""
        )

        prompt = f"""
        STRICTLY screen this paper based on the review plan, PICO, and Inclusion/Exclusion Criteria.

        PICO:
        - Pop: {pico.population} | Int: {pico.intervention} | Comp: {pico.comparator} | Out: {pico.outcome}

        CRITERIA:
        Inclusion: {inc_text}
        Exclusion: {excl_text}
        {proto_block}
        PAPER:
        Title: {paper.title}
        Abstract: {paper.abstract}

        SCREENING RULES:
        1. Check ALL inclusion criteria - paper must meet EVERY inclusion criterion to be included
        2. Check exclusion criteria - paper fails if it matches ANY exclusion criterion
        3. Be systematic but CAUTIOUS - evaluate each criterion individually
        4. If abstract is ambiguous or lacks detail, DEFAULT TO 'INCLUDE' to avoid false exclusions
        5. Only exclude if CLEARLY irrelevant or strong exclusion criteria are met

        TASK:
        1. Decision: Error on the side of INCLUSION. Only exclude if clearly irrelevant or strong exclusion criteria met.
        2. Bucket: Select 3-5 words from the criteria above that best explains the decision
        3. Reason: Explain which specific criteria were met/not met. If ambiguous, state "Abstract lacks sufficient detail"
        4. Criteria Evaluation: For EACH criterion in the inclusion and exclusion lists above, evaluate whether the paper meets it. Return "INCLUDE" if the paper meets the criterion, "EXCLUDE" if it does not.

        CRITICAL: Return ONLY a valid JSON object. No conversational text, no explanations, no markdown formatting.
        
        REQUIRED JSON FORMAT:
        {{
            "decision": "Include", 
            "bucket": "All criteria met", 
            "reason": "Meets all inclusion criteria and no exclusion criteria",
            "criteria_evaluations": {{
                "Inclusion Criterion 1": "INCLUDE" or "EXCLUDE",
                "Inclusion Criterion 2": "INCLUDE" or "EXCLUDE",
                ...
                "Exclusion Criterion 1": "INCLUDE" or "EXCLUDE",
                "Exclusion Criterion 2": "INCLUDE" or "EXCLUDE"
            }}
        }}
        
        You MUST provide: decision, bucket, reason, and criteria_evaluations for ALL criteria.
        """
        
        try:
            # Explicitly using HumanMessage from langchain_core
            from langchain_core.messages import HumanMessage
            response = model.invoke([HumanMessage(content=prompt)])
            
            # Robust JSON extraction (Old logic that worked)
            raw_content = response.content
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group())
                result = {
                    "decision": data.get('decision', 'Exclude'),
                    "bucket": data.get('bucket', 'Criteria mismatch'),
                    "reason": data.get('reason', 'N/A')
                }
                
                # Add criteria evaluations if present
                criteria_evals = data.get('criteria_evaluations', {})
                if criteria_evals:
                    result.update(criteria_evals)
                
                return result
        except Exception as e:
            # Silently handle screening errors with fallback logic
            # Try simple keyword matching as fallback
            title_lower = paper.title.lower()
            abstract_lower = paper.abstract.lower()
            pico_terms = []
            
            if pico.population:
                pico_terms.extend(pico.population.lower().split())
            if pico.intervention:
                pico_terms.extend(pico.intervention.lower().split())
            
            # Check if any PICO terms are present
            matches = sum(1 for term in pico_terms if term in title_lower or term in abstract_lower)
            
            # More permissive matching - lower threshold for inclusion
            if matches >= 1:  # If at least 1 term matches, include (more permissive)
                result = {"decision": "Include", "bucket": "Keyword match", "reason": f"Found {matches} matching PICO terms"}
            else:
                result = {"decision": "Include", "bucket": "Cautionary include", "reason": "Abstract lacks sufficient detail - defaulting to include"}
            
            # Add criteria evaluations with ERROR values for fallback
            try:
                import streamlit as st
                inclusion_criteria = st.session_state.get('inclusion_list', [])
                exclusion_criteria = st.session_state.get('exclusion_list', [])
                
                # Check if paper has basic content
                paper_text = f"{paper.title} {paper.abstract}".lower()
                if not paper_text or paper_text == 'n/a':
                    result = {
                        "decision": "Exclude", 
                        "bucket": "Missing content",
                        "reason": "Missing title or abstract",
                        **{criterion: "EXCLUDE" for criterion in inclusion_criteria + exclusion_criteria}
                    }
                else:
                    # Add criteria evaluations with safe defaults
                    for criterion in inclusion_criteria + exclusion_criteria:
                        result[criterion] = "ERROR"
            except:
                pass
            
            return result
            
    @staticmethod
    def screen_full_text(paper: Dict[str, Any], pico: PICOCriteria, model_name: str) -> Dict[str, Any]:
        """Performs deeper eligibility screening on full-text or detailed abstracts using all paper information."""
        model = AIService.get_model(model_name)
        
        # Extract paper content - combine all available fields
        paper_text = f"""
        TITLE: {paper.get('Title', 'N/A')}
        
        ABSTRACT: {paper.get('Abstract', 'N/A')}
        
        SOURCE: {paper.get('Source', 'N/A')}
        """
        
        if not model:
            # Use keyword matching as fallback when model fails
            text_lower = paper_text.lower()
            pico_terms = []
            
            if pico.population:
                pico_terms.extend(pico.population.lower().split())
            if pico.intervention:
                pico_terms.extend(pico.intervention.lower().split())
            
            # Check if any PICO terms are present
            matches = sum(1 for term in pico_terms if term in text_lower)
            
            if matches >= 3:  # Higher threshold for full-text
                return {"decision": "Include", "reason": f"Found {matches} matching PICO terms in full text", "citation": "Keyword match"}
            else:
                return {"decision": "Exclude", "reason": f"Only {matches} PICO terms found in full text", "citation": "N/A"}

        # Get current criteria from session state
        try:
            import streamlit as st
            inclusion_criteria = st.session_state.get('inclusion_list', [])
            exclusion_criteria = st.session_state.get('exclusion_list', [])
        except:
            inclusion_criteria = []
            exclusion_criteria = []

        inc_text = "\n".join(f"        - {c}" for c in inclusion_criteria) if inclusion_criteria else "        (none specified)"
        excl_text = "\n".join(f"        - {c}" for c in exclusion_criteria) if exclusion_criteria else "        (none specified)"

        prompt = f"""
        You are performing the Full-Text Eligibility phase of a Systematic Review.

        INCLUSION CRITERIA:
{inc_text}

        EXCLUSION CRITERIA:
{excl_text}

        FULL TEXT TO ANALYZE:
        {paper_text[:4000]}

        SCREENING RULES:
        1. Paper must meet ALL inclusion criteria to be included
        2. Paper must NOT match ANY exclusion criteria to be included
        3. Be thorough - this is full-text screening, so be more comprehensive
        4. Look for specific details in methods, results, and discussion sections

        CRITERIA EVALUATION:
        Evaluate the paper against EACH criterion listed above. For an INCLUSION criterion,
        return "INCLUDE" if the paper meets it, "EXCLUDE" if it does not. For an EXCLUSION
        criterion, return "INCLUDE" if the paper does NOT match it, "EXCLUDE" if the paper
        matches/violates it. Use the criterion's EXACT wording as the JSON key.

        AI REASONING SUMMARY:
        Provide a 1-2 sentence summary explaining the main reason for inclusion or exclusion.
        Focus on which specific criteria were met or violated.

        REQUIRED JSON FORMAT (use the exact criterion text as each key, one entry per criterion):
        {{
            "decision": "Include" or "Exclude",
            "reason": "Brief 1-2 sentence summary of why included/excluded",
            "citation": "Direct quote from text (max 100 words)...",
            "criteria_evaluations": {{
                "<exact text of an inclusion or exclusion criterion>": "INCLUDE" or "EXCLUDE"
            }}
        }}

        You MUST include an entry in "criteria_evaluations" for EVERY inclusion and exclusion
        criterion, keyed by the criterion's exact text. Only use "INCLUDE" or "EXCLUDE" values.
        """

        try:
            from langchain_core.messages import HumanMessage
            import re
            import json
            
            response = model.invoke([HumanMessage(content=prompt)])
            raw_content = response.content
            
            # More robust JSON extraction
            result = {}
            try:
                # First try: direct JSON parsing
                result = json.loads(raw_content)
            except json.JSONDecodeError:
                # Second try: extract JSON from markdown blocks
                import re
                # Look for JSON in markdown code blocks
                md_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_content, re.DOTALL)
                if md_match:
                    try:
                        result = json.loads(md_match.group(1))
                    except:
                        pass
                
                # Third try: find any JSON-like structure
                if not result:
                    # Find the outermost braces
                    start = raw_content.find('{')
                    end = raw_content.rfind('}')
                    if start != -1 and end != -1 and end > start:
                        try:
                            result = json.loads(raw_content[start:end+1])
                        except:
                            pass
                
                # Fourth try: manual extraction of key fields using regex
                if not result:
                    result = {}
                    # Extract decision
                    decision_match = re.search(r'"decision"\s*:\s*"([^"]+)"', raw_content, re.IGNORECASE)
                    if decision_match:
                        result['decision'] = decision_match.group(1)
                    
                    # Extract reason
                    reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', raw_content, re.IGNORECASE)
                    if reason_match:
                        result['reason'] = reason_match.group(1)
                    
                    # Extract citation
                    citation_match = re.search(r'"citation"\s*:\s*"([^"]+)"', raw_content, re.IGNORECASE)
                    if citation_match:
                        result['citation'] = citation_match.group(1)
                    
                    # Extract criteria evaluations - handle both flat and nested formats
                    for criterion in inclusion_criteria + exclusion_criteria:
                        # Escape special regex characters in criterion
                        escaped_criterion = re.escape(criterion)
                        
                        # Try flat format first: "criterion": "INCLUDE"
                        criterion_match = re.search(rf'"{escaped_criterion}"\s*:\s*"(INCLUDE|EXCLUDE|ERROR)"', raw_content, re.IGNORECASE)
                        if criterion_match:
                            result[criterion] = criterion_match.group(1).upper()
                        else:
                            # Try nested format: 'Criterion': 'Value' (single quotes)
                            nested_match = re.search(rf"'{escaped_criterion}'\s*:\s*'?(INCLUDE|EXCLUDE|ERROR)'?", raw_content, re.IGNORECASE)
                            if nested_match:
                                result[criterion] = nested_match.group(1).upper()
                            else:
                                # Try to find it in any JSON-like structure
                                loose_match = re.search(rf'{escaped_criterion}["\']?\s*[:=]\s*["\']?(INCLUDE|EXCLUDE|ERROR)["\']?', raw_content, re.IGNORECASE)
                                if loose_match:
                                    result[criterion] = loose_match.group(1).upper()
            
            # Ensure we have at least the basic fields
            if not result.get('decision'):
                # Try to infer from content
                content_lower = raw_content.lower()
                if 'include' in content_lower and 'exclude' not in content_lower.split('include')[0]:
                    result['decision'] = 'Include'
                else:
                    result['decision'] = 'Exclude'
            
            # Clean decision
            decision = result.get('decision', 'Exclude').replace('✅', '').replace('❌', '').strip()
            
            # Build full result with all criteria
            full_result = {
                "decision": decision,
                "reason": result.get('reason', 'N/A'),
                "citation": result.get('citation', 'N/A')
            }
            
            # Extract criteria_evaluations if present (new structured format)
            criteria_evals = result.get('criteria_evaluations', {})
            
            def safe_evaluate_criterion(criterion, paper_text_lower, is_inclusion=True):
                """Safely evaluate a criterion with multiple fallback strategies."""
                # Try nested structure first, then flat
                eval_value = criteria_evals.get(criterion, result.get(criterion, None))
                
                if eval_value and isinstance(eval_value, str):
                    val = eval_value.upper().strip()
                    if val in ['INCLUDE', 'EXCLUDE', 'YES', 'NO', 'TRUE', 'FALSE', 'PASS', 'FAIL']:
                        # Normalize to INCLUDE/EXCLUDE
                        if val in ['INCLUDE', 'YES', 'TRUE', 'PASS']:
                            return 'INCLUDE'
                        else:
                            return 'EXCLUDE'
                
                # Fallback 1: Try keyword matching in the paper text
                criterion_words = criterion.lower().split()
                # Extract key terms (words > 2 chars, exclude common words)
                stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'this', 'that', 'these', 'those'}
                key_terms = [w for w in criterion_words if len(w) > 2 and w not in stop_words]
                
                if key_terms:
                    matches = sum(1 for term in key_terms if term in paper_text_lower)
                    match_ratio = matches / len(key_terms)
                    
                    # For inclusion: if we match any terms, INCLUDE (more permissive)
                    if is_inclusion:
                        return 'INCLUDE' if match_ratio > 0.2 else 'EXCLUDE'
                    else:
                        # For exclusion: if we match most terms, EXCLUDE (violated)
                        return 'EXCLUDE' if match_ratio > 0.6 else 'INCLUDE'
                
                # Fallback 2: Default to INCLUDE for safety (avoid false exclusions)
                return 'INCLUDE'
            
            # Prepare paper text for keyword matching
            paper_text_lower = paper_text.lower()
            
            # Check if paper has basic content - if not, return safe defaults
            if not paper_text_lower or paper_text_lower == 'n/a':
                return {
                    "decision": "Exclude",
                    "reason": "Missing title or abstract",
                    "citation": "N/A",
                    **{criterion: "EXCLUDE" for criterion in inclusion_criteria + exclusion_criteria}
                }
            
            # Add each criterion evaluation with robust fallbacks
            for criterion in inclusion_criteria:
                full_result[criterion] = safe_evaluate_criterion(criterion, paper_text_lower, is_inclusion=True)
            
            for criterion in exclusion_criteria:
                full_result[criterion] = safe_evaluate_criterion(criterion, paper_text_lower, is_inclusion=False)
            
            return full_result
        except Exception as e:
            # Try to extract decision from response even if parsing fails
            try:
                decision = "Include" if "include" in str(e).lower() else "Exclude"
                reason = f"Processing fallback: {str(e)[:50]}"
            except:
                decision = "Include"  # Default to include for safety
                reason = "Processing completed with fallback"
            
            # Safely evaluate criteria even in error case using keyword matching
            paper_text_lower = paper_text.lower() if 'paper_text' in locals() else ""
            
            def fallback_evaluate(criterion, is_inclusion=True):
                if not paper_text_lower:
                    return 'INCLUDE'  # Default safe
                criterion_words = criterion.lower().split()
                key_terms = [w for w in criterion_words if len(w) > 3]
                if key_terms:
                    matches = sum(1 for term in key_terms if term in paper_text_lower)
                    match_ratio = matches / len(key_terms)
                    if is_inclusion:
                        return 'INCLUDE' if match_ratio > 0.5 else 'EXCLUDE'
                    else:
                        return 'EXCLUDE' if match_ratio > 0.5 else 'INCLUDE'
                return 'INCLUDE'
            
            return {
                "decision": decision,
                "reason": reason,
                "citation": "N/A",
                **{criterion: fallback_evaluate(criterion, True) for criterion in inclusion_criteria},
                **{criterion: fallback_evaluate(criterion, False) for criterion in exclusion_criteria}
            }

    @staticmethod
    def optimize_query_agentic(initial_query: str, model_name: str, selected_sources: List[str] = None) -> Dict[str, Any]:
        from data_services import DataAggregator  # Local import to prevent circularity
        
        model = AIService.get_model(model_name)
        if not model:
            return {
                "final_query": initial_query,
                "log": [{"iteration": 1, "query": initial_query, "total_yield": 0, "error": "Model initialization failed"}],
                "final_yield": 0
            }
        
        current_query = initial_query
        optimization_log = []
        
        # Target range for a "good" systematic review search
        MIN_TARGET = 100
        MAX_TARGET = 500
        
        for i in range(3):  # Max 3 iterations
            try:
                # 1. Get the current count across SELECTED APIs only
                counts = DataAggregator.get_all_counts(current_query, selected_sources)
                total = sum(counts.values())
                
                # 2. Add detailed trace information
                log_entry = {
                    "iteration": i + 1,
                    "query": current_query,
                    "total_yield": total,
                    "source_breakdown": counts,
                    "status": "checking"
                }
                optimization_log.append(log_entry)
                
                # 3. If within range, stop here
                if MIN_TARGET <= total <= MAX_TARGET:
                    log_entry["status"] = "success"
                    break
                    
                # 4. Determine if we need to broaden or narrow
                direction = "broaden" if total < MIN_TARGET else "narrow"
                
                # 5. Create a more sophisticated refinement prompt
                refinement_prompt = f"""
                You are a medical information specialist optimizing a systematic review search query.

                Current query: "{current_query}"
                Current results: {total} total (target: {MIN_TARGET}-{MAX_TARGET})
                Source breakdown: {counts}
                Selected databases: {selected_sources if selected_sources else 'All available'}

                This query needs to be {direction}ed to reach the target range.

                RULES:
                - If BROADENING: Add synonyms with OR, remove specific filters, use broader MeSH terms
                - If NARROWING: Add specific study designs (e.g., "randomized controlled trial"), use more specific MeSH terms, add population restrictions
                - Always maintain valid Boolean syntax
                - Keep PubMed [Mesh] tags when present
                - Return ONLY the new query string, no explanations

                Examples:
                - Broadening: ("Diabetes Mellitus"[Mesh] OR "diabetes"[tiab]) → ("Diabetes Mellitus"[Mesh] OR "diabetes"[tiab] OR "high blood sugar"[tiab])
                - Narrowing: ("treatment"[tiab]) → ("treatment"[tiab] AND "randomized controlled trial"[pt])
                """
                
                response = model.invoke([HumanMessage(content=refinement_prompt)])
                new_query = response.content.strip().replace('"', '').replace('```', '').strip()
                
                # Validate the new query is not empty
                if new_query and len(new_query) > 5:
                    current_query = new_query
                    log_entry["status"] = f"refined_{direction}"
                else:
                    log_entry["status"] = "refinement_failed"
                    break
                    
            except Exception as e:
                error_entry = {
                    "iteration": i + 1,
                    "query": current_query,
                    "total_yield": total if 'total' in locals() else 0,
                    "error": str(e),
                    "status": "error"
                }
                optimization_log.append(error_entry)
                break
        
        return {
            "final_query": current_query,
            "log": optimization_log,
            "final_yield": optimization_log[-1]["total_yield"] if optimization_log else 0
        }

    @staticmethod
    def optimize_query_multi_agent(initial_query: str, model_name: str, selected_sources: List[str] = None) -> Dict[str, Any]:
        """
        Multi-agent approach where each database agent debates the optimal query.
        Each agent specializes in their database's query syntax and content.
        """
        from data_services import DataAggregator  # Local import to prevent circularity
        
        model = AIService.get_model(model_name)
        if not model:
            return {
                "final_query": initial_query,
                "log": [{"iteration": 1, "query": initial_query, "total_yield": 0, "error": "Model initialization failed"}],
                "final_yield": 0
            }
        
        current_query = initial_query
        optimization_log = []
        
        # Target range for a "good" systematic review search
        MIN_TARGET = 100
        MAX_TARGET = 500
        
        # Database-specific agents
        database_agents = selected_sources if selected_sources else ["PubMed", "arXiv", "Semantic Scholar"]
        
        for i in range(3):  # Max 3 iterations
            try:
                # 1. Get current counts for all selected databases
                counts = DataAggregator.get_all_counts(current_query, selected_sources)
                total = sum(counts.values())
                
                # 2. Add detailed trace information
                log_entry = {
                    "iteration": i + 1,
                    "query": current_query,
                    "total_yield": total,
                    "source_breakdown": counts,
                    "status": "checking"
                }
                optimization_log.append(log_entry)
                
                # 3. If within range, stop here
                if MIN_TARGET <= total <= MAX_TARGET:
                    log_entry["status"] = "success"
                    break
                
                # 4. Multi-agent debate phase
                direction = "broaden" if total < MIN_TARGET else "narrow"
                
                # Create database-specific agents
                agent_suggestions = {}
                
                for database in database_agents:
                    if database not in counts:
                        continue
                        
                    # Database-specific prompt
                    agent_prompt = f"""
                    You are a {database} search specialist agent optimizing a systematic review query.

                    Current query: "{current_query}"
                    Current {database} results: {counts[database]} (target: {MIN_TARGET}-{MAX_TARGET} total)
                    Overall results: {total} across all databases

                    This query needs to be {direction}ed to reach the target range.

                    {database}-SPECIFIC RULES:
                    - PubMed: Use MeSH terms, Boolean operators, field tags [tiab], [Mesh], [pt]
                    - arXiv: Use general search terms, avoid PubMed-specific syntax, focus on technical terms
                    - Semantic Scholar: Use natural language, focus on AI/ML terminology, avoid complex Boolean

                    Suggest a {direction}ed query optimized for {database}.
                    Return ONLY the new query string, no explanations.
                    """
                    
                    try:
                        response = model.invoke([HumanMessage(content=agent_prompt)])
                        suggested_query = response.content.strip().replace('"', '').replace('```', '').strip()
                        agent_suggestions[database] = suggested_query
                    except Exception as e:
                        agent_suggestions[database] = current_query
                
                # 5. Agent debate and consensus
                debate_prompt = f"""
                You are coordinating multiple database search agents to optimize a systematic review query.

                Current query: "{current_query}"
                Current results: {total} total (target: {MIN_TARGET}-{MAX_TARGET})
                Source breakdown: {counts}

                Agent suggestions:
                {chr(10).join([f"- {agent}: {suggestion}" for agent, suggestion in agent_suggestions.items()])}

                Analyze all suggestions and create the best consensus query that:
                1. Works across all selected databases
                2. Achieves the {direction}ing goal
                3. Maintains valid syntax for all databases
                4. Preserves the core search intent

                Return ONLY the final consensus query string, no explanations.
                """
                
                response = model.invoke([HumanMessage(content=debate_prompt)])
                new_query = response.content.strip().replace('"', '').replace('```', '').strip()
                
                # Validate the new query is not empty
                if new_query and len(new_query) > 5:
                    current_query = new_query
                    log_entry["status"] = f"multi_agent_{direction}"
                    log_entry["agent_suggestions"] = agent_suggestions
                else:
                    log_entry["status"] = "consensus_failed"
                    break
                    
            except Exception as e:
                error_entry = {
                    "iteration": i + 1,
                    "query": current_query,
                    "total_yield": total if 'total' in locals() else 0,
                    "error": str(e),
                    "status": "error"
                }
                optimization_log.append(error_entry)
                break
        
        return {
            "final_query": current_query,
            "log": optimization_log,
            "final_yield": optimization_log[-1]["total_yield"] if optimization_log else 0
        }

    # Add to utils.py inside AIService class

    @staticmethod
    def run_agentic_search(initial_query: str, model_name: str, target_range=(20, 100)):
        """
        An agentic loop that adjusts search queries based on live API feedback.
        """
        from data_services import DataAggregator
        
        model = AIService.get_model(model_name)
        current_query = initial_query
        min_results, max_results = target_range
        search_log = []
        
        # Limit to 3 iterations to prevent API cost/time spiraling
        for iteration in range(3):
            # 1. Probe the APIs for counts only (fast)
            counts = DataAggregator.get_all_counts(current_query)
            total_found = sum(counts.values())
            
            log_entry = {"iteration": iteration + 1, "query": current_query, "yield": total_found}
            search_log.append(log_entry)
            
            # 2. Check if we are within the "Goldilocks" zone
            if min_results <= total_found <= max_results:
                return current_query, search_log, "success"
            
            # 3. Reflection & Adjustment
            adjustment_type = "broaden" if total_found < min_results else "narrow"
            
            reflection_prompt = f"""
            Your previous search query '{current_query}' yielded {total_found} results.
            This is {'too few' if adjustment_type == 'broaden' else 'too many'}.
            
            Goal: Adjust the query to {adjustment_type} the results to land between {min_results} and {max_results}.
            - If narrowing: Add specific study designs (e.g., 'RCT') or more specific PICO terms.
            - If broadening: Remove restrictive keywords or use OR synonyms (e.g., 'hypertension OR high blood pressure').
            
            Return ONLY the new query string. No chat.
            """
            
            response = model.invoke([HumanMessage(content=reflection_prompt)])
            current_query = response.content.strip().replace('"', '')

        return current_query, search_log, "exhausted_attempts"
        
    @staticmethod
    def fetch_citations(paper_id: str, source: str, title: str, snowball_type: str, max_results: int, active_sources: list) -> List[Dict]:
        """
        Fetch citations for a paper (backward references or forward citations).
        
        Args:
            paper_id: The ID of the paper (PMID, DOI, etc.)
            source: The source database (PubMed, Europe PMC, etc.)
            title: Paper title for fallback search
            snowball_type: "Both", "Backward (References)", or "Forward (Cited by)"
            max_results: Maximum citations to fetch per paper
            active_sources: List of active data sources to query
        
        Returns:
            List of citation dictionaries
        """
        import requests
        from Bio import Entrez
        from config import Config
        
        citations = []
        
        # Normalize paper_id
        if not paper_id and title:
            # Try to find paper by title if ID is missing
            try:
                Entrez.email = Config.ENTREZ_EMAIL
                handle = Entrez.esearch(db="pubmed", term=title, retmax=1)
                record = Entrez.read(handle)
                if record['IdList']:
                    paper_id = record['IdList'][0]
            except:
                pass
        
        if not paper_id:
            return citations
        
        # Try Europe PMC for references
        if "Europe PMC" in active_sources or "PubMed" in active_sources:
            try:
                if snowball_type in ["Both", "Backward (References)"]:
                    # Fetch references (backward)
                    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{paper_id}/references"
                    response = requests.get(url, params={"pageSize": max_results, "format": "json"}, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        refs = data.get('referenceList', {}).get('reference', [])
                        for ref in refs:
                            citations.append({
                                'id': ref.get('id', ''),
                                'title': ref.get('title', ref.get('source', 'Unknown')),
                                'abstract': ref.get('abstractText', ''),
                                'url': f"https://pubmed.ncbi.nlm.nih.gov/{ref.get('id', '')}" if ref.get('id') else '',
                                'source': 'Europe PMC (Reference)',
                                'citation_type': 'backward'
                            })
                
                if snowball_type in ["Both", "Forward (Cited by)"]:
                    # Fetch citing papers (forward)
                    url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{paper_id}/citations"
                    print(f"[Europe PMC] Forward citations URL: {url}")
                    response = requests.get(url, params={"pageSize": max_results, "format": "json"}, timeout=10)
                    print(f"[Europe PMC] Response status: {response.status_code}")
                    if response.status_code == 200:
                        data = response.json()
                        cites = data.get('citationList', {}).get('citation', [])
                        print(f"[Europe PMC] Found {len(cites)} forward citations")
                        for cite in cites:
                            citations.append({
                                'id': cite.get('id', ''),
                                'title': cite.get('title', cite.get('source', 'Unknown')),
                                'abstract': cite.get('abstractText', ''),
                                'url': f"https://pubmed.ncbi.nlm.nih.gov/{cite.get('id', '')}" if cite.get('id') else '',
                                'source': 'Europe PMC (Cited by)',
                                'citation_type': 'forward'
                            })
                    else:
                        print(f"[Europe PMC] Error response: {response.text[:200]}")
            except Exception as e:
                print(f"Europe PMC citation error: {e}")
        
        # Try Semantic Scholar
        if "Semantic Scholar" in active_sources:
            try:
                # First, we need to find the Semantic Scholar paper ID
                ss_url = "https://api.semanticscholar.org/graph/v1/paper/search"
                params = {
                    "query": title,
                    "fields": "paperId,title,abstract,url",
                    "limit": 1
                }
                response = requests.get(ss_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    papers = data.get('data', [])
                    if papers:
                        ss_paper_id = papers[0].get('paperId')
                        
                        if ss_paper_id:
                            if snowball_type in ["Both", "Backward (References)"]:
                                # Fetch references
                                refs_url = f"https://api.semanticscholar.org/graph/v1/paper/{ss_paper_id}/references"
                                refs_params = {"fields": "paperId,title,abstract,url", "limit": max_results}
                                refs_response = requests.get(refs_url, params=refs_params, timeout=10)
                                if refs_response.status_code == 200:
                                    refs_data = refs_response.json()
                                    for ref in refs_data.get('data', []):
                                        cited = ref.get('citedPaper', {})
                                        citations.append({
                                            'id': cited.get('paperId', ''),
                                            'title': cited.get('title', 'Unknown'),
                                            'abstract': cited.get('abstract', ''),
                                            'url': cited.get('url', ''),
                                            'source': 'Semantic Scholar (Reference)',
                                            'citation_type': 'backward'
                                        })
                            
                            if snowball_type in ["Both", "Forward (Cited by)"]:
                                # Fetch citations
                                cites_url = f"https://api.semanticscholar.org/graph/v1/paper/{ss_paper_id}/citations"
                                cites_params = {"fields": "paperId,title,abstract,url", "limit": max_results}
                                print(f"[Semantic Scholar] Forward citations URL: {cites_url}")
                                cites_response = requests.get(cites_url, params=cites_params, timeout=10)
                                print(f"[Semantic Scholar] Response status: {cites_response.status_code}")
                                if cites_response.status_code == 200:
                                    cites_data = cites_response.json()
                                    print(f"[Semantic Scholar] Found {len(cites_data.get('data', []))} forward citations")
                                    for cite in cites_data.get('data', []):
                                        citing = cite.get('citingPaper', {})
                                        citations.append({
                                            'id': citing.get('paperId', ''),
                                            'title': citing.get('title', 'Unknown'),
                                            'abstract': citing.get('abstract', ''),
                                            'url': citing.get('url', ''),
                                            'source': 'Semantic Scholar (Cited by)',
                                            'citation_type': 'forward'
                                        })
                                else:
                                    print(f"[Semantic Scholar] Error response: {cites_response.text[:200]}")
            except Exception as e:
                print(f"Semantic Scholar citation error: {e}")
        
        # Try OpenAlex (no API key needed, good coverage)
        try:
            # Search for paper by title or DOI
            if paper_id.startswith('10.'):  # DOI
                openalex_url = f"https://api.openalex.org/works/doi:{paper_id}"
            else:
                # Search by title
                search_url = "https://api.openalex.org/works"
                params = {"search": title, "per_page": 1}
                search_response = requests.get(search_url, params=params, timeout=10)
                if search_response.status_code == 200:
                    search_data = search_response.json()
                    results = search_data.get('results', [])
                    if results:
                        openalex_url = f"https://api.openalex.org/works/{results[0].get('id')}"
                    else:
                        openalex_url = None
                else:
                    openalex_url = None
            
            if openalex_url:
                response = requests.get(openalex_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    if snowball_type in ["Both", "Backward (References)"]:
                        # Get referenced works
                        refs = data.get('referenced_works', [])
                        for ref_id in refs[:max_results]:
                            try:
                                ref_response = requests.get(f"https://api.openalex.org/works/{ref_id}", timeout=5)
                                if ref_response.status_code == 200:
                                    ref_data = ref_response.json()
                                    citations.append({
                                        'id': ref_data.get('doi', ref_id.split('/')[-1]),
                                        'title': ref_data.get('display_name', 'Unknown'),
                                        'abstract': ref_data.get('abstract', ''),
                                        'url': ref_data.get('open_access', {}).get('oa_url', f"https://doi.org/{ref_data.get('doi')}" if ref_data.get('doi') else ''),
                                        'source': 'OpenAlex (Reference)',
                                        'citation_type': 'backward'
                                    })
                            except:
                                pass
                    
                    if snowball_type in ["Both", "Forward (Cited by)"]:
                        # Get citing works
                        cited_by_url = data.get('cited_by_api_url')
                        print(f"[OpenAlex] Forward citations URL: {cited_by_url}")
                        if cited_by_url:
                            try:
                                cites_response = requests.get(cited_by_url, params={"per_page": max_results}, timeout=10)
                                print(f"[OpenAlex] Response status: {cites_response.status_code}")
                                if cites_response.status_code == 200:
                                    cites_data = cites_response.json()
                                    print(f"[OpenAlex] Found {len(cites_data.get('results', []))} forward citations")
                                    for cite in cites_data.get('results', []):
                                        citations.append({
                                            'id': cite.get('doi', cite.get('id', '').split('/')[-1]),
                                            'title': cite.get('display_name', 'Unknown'),
                                            'abstract': cite.get('abstract', ''),
                                            'url': cite.get('open_access', {}).get('oa_url', f"https://doi.org/{cite.get('doi')}" if cite.get('doi') else ''),
                                            'source': 'OpenAlex (Cited by)',
                                            'citation_type': 'forward'
                                        })
                            except Exception as e:
                                print(f"[OpenAlex] Exception fetching forward citations: {e}")
        except Exception as e:
            print(f"OpenAlex citation error: {e}")
        
        return citations
        
class Deduplicator:
    @staticmethod
    def normalize_text(text: str) -> str:
        """Removes special characters and lowercases for fuzzy matching."""
        if not text: return ""
        return re.sub(r'[^a-z0-9]', '', text.lower())

    @staticmethod
    def run(papers: List[Paper]) -> Tuple[List[Paper], List[Paper]]:
        """Identifies unique papers and tracks duplicates."""
        unique, dups = [], []
        seen_dois, seen_slugs = set(), set()
        for p in papers:
            doi = str(p.id).strip().lower()
            slug = Deduplicator.normalize_text(p.title)
            is_dup = (doi and doi != "n/a" and doi in seen_dois) or (slug in seen_slugs)
            if not is_dup:
                unique.append(p)
                if doi and doi != "n/a": seen_dois.add(doi)
                if slug: seen_slugs.add(slug)
            else:
                dups.append(p)
        return unique, dups

class AITableExtractor:
    """
    Uses an LLM to identify and structure any tabular data that appears in a
    paper's abstract or extracted text.  This is a best-effort complement to
    direct PDF/PMC table extraction: abstracts often report key result tables
    inline (e.g. "Group A: 45% vs Group B: 23%, p<0.001").
    """

    @staticmethod
    def extract_from_text(text: str, model_name: str) -> List[Dict]:
        """
        Ask the LLM to find tabular data in *text* and return it as a list of
        structured table dicts matching the same schema used by PDFService.

        Returns [] if no tables are found or extraction fails.
        """
        if not text or len(text.strip()) < 50:
            return []

        model = AIService.get_model(model_name)
        if not model:
            return []

        prompt = f"""You are a scientific data extractor specialising in medical literature.

TASK:
Examine the text below and extract EVERY table or structured numeric comparison you can find.
This includes result tables, demographic tables, outcome tables, and any structured list of values
comparing groups.

TEXT:
\"\"\"
{text[:3000]}
\"\"\"

OUTPUT RULES:
- Return ONLY a valid JSON array.  No preamble, no markdown fences.
- If NO tabular data exists, return an empty array: []
- Each table must follow this exact schema:
  {{
    "label":   "Table N – short description",
    "caption": "One-sentence description of what the table shows",
    "headers": ["Column1", "Column2", ...],
    "rows":    [["val", "val", ...], ...]
  }}
- Use empty string "" for any missing cell value.
- Keep all numeric values exactly as written in the source.

JSON array:"""

        try:
            response = model.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip()
            # Strip markdown fences if the model added them anyway
            raw = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(raw)
            if isinstance(data, list):
                # Basic validation: every entry must have headers and rows
                valid = [
                    t for t in data
                    if isinstance(t, dict)
                    and isinstance(t.get("headers"), list)
                    and isinstance(t.get("rows"), list)
                    and t["rows"]
                ]
                return valid
        except Exception as e:
            print(f"AITableExtractor error: {e}")

        return []

    @staticmethod
    def enrich_papers(papers: List[Paper], model_name: str) -> List[Paper]:
        """
        For each paper without tables, attempt AI-based extraction from its
        abstract.  Updates paper.tables in place.
        """
        for paper in papers:
            if paper.tables:
                continue  # Already has tables from PDF/PMC – skip
            tables = AITableExtractor.extract_from_text(paper.abstract, model_name)
            if tables:
                paper.tables = tables
        return papers



    @staticmethod
    def clean_for_general_search(query: str) -> str:
        """Removes MeSH tags and simplifies queries for better search engine compatibility."""
        # Remove MeSH tags
        cleaned = re.sub(r'\[.*?\]', '', query)
        
        # Remove extra quotes that might be causing issues
        cleaned = re.sub(r'"{2,2}', '"', cleaned)  # Remove double quotes
        cleaned = re.sub(r"'{2,2}", "'", cleaned)  # Remove single quotes
        
        # For very complex queries with nested parentheses, simplify them for general search
        # This helps with arXiv which doesn't handle complex Boolean well
        if '%28' in query or '%29' in query:  # If already URL encoded
            # For arXiv, use a simpler approach
            cleaned = re.sub(r'[()]', ' ', cleaned)  # Replace parentheses with spaces
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize spaces
            cleaned = cleaned.strip()
        else:
            # Normal cleaning for non-encoded queries
            cleaned = cleaned.replace('AND', ' ').replace('OR', ' ')
            # Remove extra quotes that might be causing arXiv issues
            cleaned = re.sub(r'"{2,2}', '"', cleaned)
            cleaned = re.sub(r"'{2,2}", "'", cleaned)
        
        # Return a simplified version for better arXiv compatibility
        # Split by spaces and rejoin to remove extra spaces
        words = cleaned.split()
        return " ".join(words)

class QueryCleaner:
    """
    Utility class for cleaning and optimizing search queries for different databases.
    """
    
    @staticmethod
    def clean_for_general_search(query: str) -> str:
        """Removes MeSH tags and simplifies queries for better search engine compatibility."""
        # Remove MeSH tags
        cleaned = re.sub(r'\[.*?\]', '', query)
        
        # Remove extra quotes that might be causing issues
        cleaned = re.sub(r'"{2,2}', '"', cleaned)  # Remove double quotes
        cleaned = re.sub(r"'{2,2}", "'", cleaned)  # Remove single quotes
        
        # For very complex queries with nested parentheses, simplify them for general search
        # This helps with arXiv which doesn't handle complex Boolean well
        if '%28' in query or '%29' in query:  # If already URL encoded
            # For arXiv, use a simpler approach
            cleaned = re.sub(r'[()]', ' ', cleaned)  # Replace parentheses with spaces
            cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize spaces
            cleaned = cleaned.strip()
        else:
            # Normal cleaning for non-encoded queries
            cleaned = cleaned.replace('AND', ' ').replace('OR', ' ')
            # Remove extra quotes that might be causing arXiv issues
            cleaned = re.sub(r'"{2,2}', '"', cleaned)
            cleaned = re.sub(r"'{2,2}", "'", cleaned)
        
        # Return a simplified version for better arXiv compatibility
        # Split by spaces and rejoin to remove extra spaces
        words = cleaned.split()
        return " ".join(words)

# In utils.py
class SearchAgent:
    def __init__(self, goal, target_n=(10, 50)):
        self.goal = goal
        self.target_min, self.target_max = target_n
        self.trace = []  # Initialize the trace list

    def run_optimized_search(self, model_name):
        current_query = self.goal
        for attempt in range(3):
            self.trace.append(f"**Attempt {attempt + 1}:** Testing query `{current_query}`")
            
            # Use DataAggregator to get actual counts
            counts = DataAggregator.get_all_counts(current_query)
            total = sum(counts.values())
            self.trace.append(f"Found {total} results across sources.")

            if self.target_min <= total <= self.target_max:
                self.trace.append("✅ Result count is within target range.")
                break
            
            # Logic to refine query
            direction = "broaden" if total < self.target_min else "narrow"
            self.trace.append(f"🔍 Result count ({total}) is outside {self.target_min}-{self.target_max}. Requesting AI to {direction}...")
            
            # Call LLM to refine (simplified for brevity)
            # ... update current_query based on LLM output ...
            
        return current_query, self.trace


# =============================================================================
# MULTI-AGENT SCREENING ARCHITECTURE
# =============================================================================
# 
# 1. ORCHESTRATOR: Decomposes PICO into 4 specialist agents (P, I, C, O)
# 2. WORKER TIER: Each agent is stateless, evaluates ALL papers for ONE PICO element
# 3. AGGREGATOR: Consensus engine with PICO AND logic (all 4 must pass)
# 4. TRACEABILITY: Per-agent votes for audit trail
#
# Used for title/abstract screening only - full-text screening uses separate logic
#
# =============================================================================

@dataclass
class AgentVote:
    """Standardized output from each criterion agent."""
    agent_name: str
    agent_type: str  # 'PICO_P', 'PICO_I', 'PICO_C', 'PICO_O', 'INCLUSION', 'EXCLUSION'
    criterion: str
    paper_id: str
    met: bool  # True if paper meets this criterion
    confidence: float  # 0.0-1.0
    evidence: str  # Quote from abstract justifying the vote
    reasoning: str


class CriterionAgent:
    """
    WORKER TIER: Stateless specialist agent.
    
    Each agent knows ONE criterion only. It evaluates ALL papers against that 
    single criterion and returns standardized AgentVote objects.
    
    Domain isolation prevents LLM fatigue - the Population Agent doesn't get
    distracted by intervention or study design details.
    """
    
    def __init__(self, name: str, agent_type: str, criterion: str, model_name: str, api_keys: dict = None):
        self.name = name
        self.agent_type = agent_type
        self.criterion = criterion
        self.model_name = model_name
        # Use thread-safe model initialization with explicit keys
        if api_keys:
            self.model = AIService.get_model_with_keys(model_name, api_keys)
        else:
            self.model = AIService.get_model(model_name)
    
    def evaluate_all_papers(self, papers: List[Paper]) -> List[AgentVote]:
        """
        Worker: Evaluate ALL papers against this agent's ONE criterion.
        Returns list of AgentVote objects, one per paper.
        """
        if not self.model:
            return [self._error_vote(p) for p in papers]
        
        # Build batch prompt for all papers
        papers_text = "\n\n".join([
            f"PAPER {i+1}:\nID: {p.id}\nTitle: {p.title}\nAbstract: {p.abstract[:500] if p.abstract else 'N/A'}" 
            for i, p in enumerate(papers)
        ])
        
        prompt = f"""You are a specialist screening agent. Your ONLY job is to evaluate papers against ONE specific criterion.

YOUR IDENTITY: {self.name}
YOUR CRITERION ({self.agent_type}): {self.criterion}

INSTRUCTIONS - BE EXTREMELY PERMISSIVE (BORDERLINE PAPERS KEEP):
1. Focus ONLY on your assigned criterion - ignore all other criteria
2. Look for BROAD CONCEPTUAL ALIGNMENT - related terms, synonyms, similar concepts
3. If ANY hint of relevance in title/abstract → PASS (even if borderline)
4. Only FAIL if the abstract has CLEAR, UNAMBIGUOUS evidence it's completely unrelated
5. Abstracts are short - absence of specific terms doesn't mean irrelevance
6. When uncertain, ALWAYS lean toward PASS (keep for full-text review)
7. Extract evidence but don't over-interpret missing information
8. Rate confidence low-moderate for borderline cases (0.3-0.6)

EXTREMELY PERMISSIVE LOGIC:
- PASS = Paper could potentially be relevant, mentions related concepts, or is borderline
- FAIL = Only when abstract CLEARLY shows zero relevance to the research topic
- DEFAULT TO PASS for all uncertain or borderline cases

PAPERS TO EVALUATE ({len(papers)} total):
{papers_text}

OUTPUT FORMAT - Return a JSON array with one object per paper:
[
    {{
        "paper_id": "paper_id_here",
        "met": true or false,
        "confidence": 0.0-1.0,
        "evidence": "Exact quote from abstract supporting your decision",
        "reasoning": "Brief explanation - note if criterion not explicitly mentioned in abstract"
    }},
    ... (for ALL papers)
]

IMPORTANT: DEFAULT TO PASS (met=true) for ALL uncertain or borderline cases. Only use met=false when abstract CLEARLY shows zero relevance.

JSON ARRAY:"""
        
        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            return self._parse_response(response.content, papers)
        except Exception as e:
            print(f"Agent {self.name} error: {e}")
            return [self._error_vote(p) for p in papers]
    
    def evaluate_single_paper(self, paper: Paper) -> AgentVote:
        """
        Evaluate a single paper against this agent's criterion.
        Returns a single AgentVote for the paper.
        """
        if not self.model:
            return self._error_vote(paper)
        
        prompt = f"""You are a specialist screening agent. Your ONLY job is to evaluate ONE paper against ONE specific criterion.

YOUR IDENTITY: {self.name}
YOUR CRITERION ({self.agent_type}): {self.criterion}

PAPER TO EVALUATE:
ID: {paper.id}
Title: {paper.title}
Abstract: {paper.abstract[:800] if paper.abstract else 'N/A'}

INSTRUCTIONS - BE EXTREMELY PERMISSIVE (BORDERLINE PAPERS KEEP):
1. Focus ONLY on your assigned criterion - ignore all other criteria
2. Look for BROAD CONCEPTUAL ALIGNMENT - related terms, synonyms, similar concepts
3. If ANY hint of relevance in title/abstract → PASS (even if borderline)
4. Only FAIL if the abstract has CLEAR, UNAMBIGUOUS evidence it's completely unrelated
5. Abstracts are short - absence of specific terms doesn't mean irrelevance
6. When uncertain, ALWAYS lean toward PASS (keep for full-text review)
7. Extract evidence but don't over-interpret missing information
8. Rate confidence low-moderate for borderline cases (0.3-0.6)

EXTREMELY PERMISSIVE LOGIC:
- PASS = Paper could potentially be relevant, mentions related concepts, or is borderline
- FAIL = Only when abstract CLEARLY shows zero relevance to the research topic
- DEFAULT TO PASS for all uncertain or borderline cases

OUTPUT FORMAT - Return a JSON object:
{{
    "paper_id": "{paper.id}",
    "met": true or false,
    "confidence": 0.0-1.0,
    "evidence": "Exact quote from abstract supporting your decision",
    "reasoning": "Brief explanation - note if criterion not explicitly mentioned in abstract"
}}

IMPORTANT: DEFAULT TO PASS (met=true) for ALL uncertain or borderline cases. Only use met=false when abstract CLEARLY shows zero relevance.

JSON OBJECT:"""
        
        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            return self._parse_single_response(response.content, paper)
        except Exception as e:
            print(f"Agent {self.name} error on paper {paper.id}: {e}")
            return self._error_vote(paper)
    
    def _parse_single_response(self, content: str, paper: Paper) -> AgentVote:
        """Parse agent response for a single paper into AgentVote."""
        try:
            # Extract JSON object
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end+1]
                data = json.loads(json_str)
                
                return AgentVote(
                    agent_name=self.name,
                    agent_type=self.agent_type,
                    criterion=self.criterion,
                    paper_id=paper.id,
                    met=data.get('met', True),  # Default to PASS
                    confidence=data.get('confidence', 0.5),
                    evidence=data.get('evidence', 'No specific evidence'),
                    reasoning=data.get('reasoning', 'Broad match applied')
                )
            else:
                print(f"Agent {self.name}: No JSON object found in response for paper {paper.id}")
        except Exception as e:
            print(f"Agent {self.name} parse error for paper {paper.id}: {e}")
            print(f"Raw content: {content[:500]}")
        
        # Fallback: return permissive vote
        return AgentVote(
            agent_name=self.name,
            agent_type=self.agent_type,
            criterion=self.criterion,
            paper_id=paper.id,
            met=True,  # Default to PASS
            confidence=0.5,
            evidence="Parse fallback - broad match assumed",
            reasoning=f"Agent {self.name} response parsing failed, using permissive default"
        )
    
    def evaluate_paper_batch(self, papers: List[Paper]) -> List[AgentVote]:
        """
        Evaluate a batch of papers against this agent's criterion in one API call.
        Returns a list of AgentVote objects, one per paper.
        """
        if not self.model:
            return [self._error_vote(p) for p in papers]
        
        # Build batch prompt
        papers_text = "\n\n".join([
            f"PAPER {i+1}:\nID: {p.id}\nTitle: {p.title}\nAbstract: {p.abstract[:600] if p.abstract else 'N/A'}" 
            for i, p in enumerate(papers)
        ])
        
        prompt = f"""You are a specialist screening agent. Your ONLY job is to evaluate papers against ONE specific criterion.

YOUR IDENTITY: {self.name}
YOUR CRITERION ({self.agent_type}): {self.criterion}

INSTRUCTIONS - BE EXTREMELY PERMISSIVE (BORDERLINE PAPERS KEEP):
1. Focus ONLY on your assigned criterion - ignore all other criteria
2. Look for BROAD CONCEPTUAL ALIGNMENT - related terms, synonyms, similar concepts
3. If ANY hint of relevance in title/abstract → PASS (even if borderline)
4. Only FAIL if the abstract has CLEAR, UNAMBIGUOUS evidence it's completely unrelated
5. Abstracts are short - absence of specific terms doesn't mean irrelevance
6. When uncertain, ALWAYS lean toward PASS (keep for full-text review)
7. Extract evidence but don't over-interpret missing information
8. Rate confidence low-moderate for borderline cases (0.3-0.6)

EXTREMELY PERMISSIVE LOGIC:
- PASS = Paper could potentially be relevant, mentions related concepts, or is borderline
- FAIL = Only when abstract CLEARLY shows zero relevance to the research topic
- DEFAULT TO PASS for all uncertain or borderline cases

PAPERS TO EVALUATE ({len(papers)} total):
{papers_text}

OUTPUT FORMAT - Return a JSON array with one object per paper:
[
    {{
        "paper_id": "paper_id_here",
        "met": true or false,
        "confidence": 0.0-1.0,
        "evidence": "Exact quote from abstract supporting your decision",
        "reasoning": "Brief explanation"
    }},
    ... (for ALL {len(papers)} papers)
]

IMPORTANT: DEFAULT TO PASS (met=true) for ALL uncertain or borderline cases. Only use met=false when abstract CLEARLY shows zero relevance.

JSON ARRAY:"""
        
        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            return self._parse_response(response.content, papers)
        except Exception as e:
            print(f"Agent {self.name} error on batch: {e}")
            return [self._error_vote(p) for p in papers]
    
    def _parse_response(self, content: str, papers: List[Paper]) -> List[AgentVote]:
        """Parse agent response into AgentVote objects."""
        try:
            # Debug: print raw response
            print(f"Agent {self.name} raw response (first 200 chars): {content[:200]}")
            
            # Extract JSON array
            start = content.find('[')
            end = content.rfind(']')
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end+1]
                data = json.loads(json_str)
                
                # Map results by paper_id
                results_by_id = {r.get('paper_id'): r for r in data if isinstance(r, dict)}
                
                votes = []
                for paper in papers:
                    result = results_by_id.get(paper.id, {})
                    if not result:
                        print(f"Warning: No result found for paper {paper.id}")
                    votes.append(AgentVote(
                        agent_name=self.name,
                        agent_type=self.agent_type,
                        criterion=self.criterion,
                        paper_id=paper.id,
                        met=result.get('met', True),  # Default to PASS if unclear
                        confidence=result.get('confidence', 0.5),
                        evidence=result.get('evidence', 'No specific evidence'),
                        reasoning=result.get('reasoning', 'Broad match applied')
                    ))
                print(f"Agent {self.name} successfully parsed {len(votes)} votes")
                return votes
            else:
                print(f"Agent {self.name}: No JSON array found in response")
        except Exception as e:
            print(f"Agent {self.name} parse error: {e}")
            print(f"Raw content: {content[:500]}")
        
        # Fallback: return permissive votes instead of errors
        print(f"Agent {self.name}: Using fallback permissive votes")
        return [AgentVote(
            agent_name=self.name,
            agent_type=self.agent_type,
            criterion=self.criterion,
            paper_id=paper.id,
            met=True,  # Default to PASS on parse failure
            confidence=0.5,
            evidence="Parse fallback - broad match assumed",
            reasoning=f"Agent {self.name} response parsing failed, using permissive default"
        ) for paper in papers]
    
    def _error_vote(self, paper: Paper) -> AgentVote:
        """Create permissive fallback vote for failed evaluation."""
        return AgentVote(
            agent_name=self.name,
            agent_type=self.agent_type,
            criterion=self.criterion,
            paper_id=paper.id,
            met=True,  # Default to PASS on error
            confidence=0.5,
            evidence="Evaluation error - defaulting to PASS",
            reasoning=f"Agent {self.name} encountered error, using permissive default"
        )


class ScreeningOrchestrator:
    """
    ORCHESTRATOR: Central intelligence that decomposes criteria and manages workers.
    
    1. DECOMPOSITION: Breaks PICO + inclusion/exclusion into N specialist roles
    2. FAN-OUT: Creates CriterionAgent for each criterion, runs in parallel
    3. AGGREGATION: Collects all votes and applies consensus logic
    """
    
    def __init__(self, pico: PICOCriteria, inclusion_criteria: List[str], 
                 exclusion_criteria: List[str], model_name: str, api_keys: dict = None):
        self.pico = pico
        self.inclusion = [c for c in inclusion_criteria if c and c.strip()]
        self.exclusion = [c for c in exclusion_criteria if c and c.strip()]
        self.model_name = model_name
        self.api_keys = api_keys or {}
        
        # Create specialist agents
        self.agents: List[CriterionAgent] = []
        self._create_agents()
    
    def _create_agents(self):
        """Decompose criteria into specialist PICO agents AND inclusion/exclusion agents."""
        # 4 PICO agents (always create if criteria exist)
        if self.pico.population:
            self.agents.append(CriterionAgent(
                "Population", "PICO_P", self.pico.population, self.model_name, self.api_keys
            ))
        if self.pico.intervention:
            self.agents.append(CriterionAgent(
                "Intervention", "PICO_I", self.pico.intervention, self.model_name, self.api_keys
            ))
        if self.pico.comparator:
            self.agents.append(CriterionAgent(
                "Comparator", "PICO_C", self.pico.comparator, self.model_name, self.api_keys
            ))
        if self.pico.outcome:
            self.agents.append(CriterionAgent(
                "Outcome", "PICO_O", self.pico.outcome, self.model_name, self.api_keys
            ))
        
        # Inclusion criteria agents - use criterion text as name (truncated if needed)
        for criterion in self.inclusion:
            short_name = criterion[:50] + "..." if len(criterion) > 50 else criterion
            self.agents.append(CriterionAgent(
                short_name, "INCLUSION", criterion, self.model_name, self.api_keys
            ))
        
        # Exclusion criteria agents - use criterion text as name
        for criterion in self.exclusion:
            short_name = criterion[:50] + "..." if len(criterion) > 50 else criterion
            self.agents.append(CriterionAgent(
                short_name, "EXCLUSION", criterion, self.model_name, self.api_keys
            ))
    
    def screen_papers(self, papers: List[Paper], progress_callback=None, batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        ORCHESTRATOR: Execute full multi-agent screening pipeline.
        Uses batched evaluation for speed while maintaining accurate progress tracking.
        
        Args:
            papers: List of papers to screen
            progress_callback: Called with (completed, total, message) for progress updates
            batch_size: Number of papers to evaluate per agent call (default 5)
        
        Returns list of paper results with per-agent votes and final decision.
        """
        if not papers or not self.agents:
            return []
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from config import Config
        
        total_papers = len(papers)
        all_votes: List[AgentVote] = []
        completed_papers = 0
        
        def screen_paper_batch(paper_batch: List[Paper], batch_start_idx: int) -> Tuple[int, List[AgentVote]]:
            """Screen a batch of papers with all agents."""
            batch_votes: List[AgentVote] = []
            total_agents = len(self.agents)
            
            # FAN-OUT: Run all agents in parallel on this batch
            with ThreadPoolExecutor(max_workers=min(Config.PARALLEL_AGENT_WORKERS, total_agents)) as executor:
                future_to_agent = {
                    executor.submit(agent.evaluate_paper_batch, paper_batch): agent 
                    for agent in self.agents
                }
                
                for future in as_completed(future_to_agent):
                    agent = future_to_agent[future]
                    try:
                        votes = future.result()  # List of AgentVote for the batch
                        batch_votes.extend(votes)
                    except Exception as exc:
                        print(f"Agent {agent.name} failed for batch: {exc}")
                        # Add error votes for all papers in batch
                        for paper in paper_batch:
                            batch_votes.append(AgentVote(
                                agent_name=agent.name,
                                agent_type=agent.agent_type,
                                criterion=agent.criterion,
                                paper_id=paper.id,
                                met=True,  # Default to PASS on error
                                confidence=0.5,
                                evidence="Batch evaluation error",
                                reasoning=f"Agent {agent.name} encountered error"
                            ))
            
            return batch_start_idx, batch_votes
        
        # Process papers in parallel batches
        max_workers = min(Config.PARALLEL_SCREENING_WORKERS, (total_papers + batch_size - 1) // batch_size)
        
        # Create batches
        batches = [papers[i:i + batch_size] for i in range(0, total_papers, batch_size)]
        total_batches = len(batches)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(screen_paper_batch, batch, i * batch_size): i 
                for i, batch in enumerate(batches)
            }
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                _, batch_votes = future.result()
                all_votes.extend(batch_votes)
                
                # Calculate papers completed in this batch
                papers_in_batch = len(batches[batch_idx])
                completed_papers += papers_in_batch
                
                if progress_callback:
                    progress_callback(completed_papers, total_papers, 
                                    f"Screened {completed_papers}/{total_papers} papers")
        
        # AGGREGATE: Apply consensus logic per paper
        return self._aggregate_results(papers, all_votes)
    
    def _aggregate_results(self, papers: List[Paper], all_votes: List[AgentVote]) -> List[Dict[str, Any]]:
        """
        AGGREGATOR: Extremely permissive screening - DEFAULT TO INCLUDE.
        
        Logic:
        - DEFAULT: All papers are INCLUDED unless there's clear evidence to exclude
        - Only exclude if: (1) explicit exclusion violation, OR (2) zero PICO/inclusion alignment with clear negative evidence
        - Borderline/uncertain papers are ALWAYS kept
        """
        # Group votes by paper
        votes_by_paper: Dict[str, List[AgentVote]] = {}
        for vote in all_votes:
            if vote.paper_id not in votes_by_paper:
                votes_by_paper[vote.paper_id] = []
            votes_by_paper[vote.paper_id].append(vote)
        
        results = []
        for paper in papers:
            paper_votes = votes_by_paper.get(paper.id, [])
            
            # Separate votes by type
            pico_votes = [v for v in paper_votes if v.agent_type.startswith('PICO')]
            inclusion_votes = [v for v in paper_votes if v.agent_type == 'INCLUSION']
            exclusion_votes = [v for v in paper_votes if v.agent_type == 'EXCLUSION']
            
            # Count how many criteria passed
            pico_met_count = sum(1 for v in pico_votes if v.met)
            inclusion_met_count = sum(1 for v in inclusion_votes if v.met)
            
            # Check for explicit exclusion violations (this is the main reason to exclude)
            violated_exclusions = [v for v in exclusion_votes if v.met]
            has_exclusion_violation = len(violated_exclusions) > 0
            
            # Check for complete irrelevance (all PICO and inclusion explicitly failed with evidence)
            pico_all_failed = len(pico_votes) > 0 and pico_met_count == 0
            inclusion_all_failed = len(inclusion_votes) > 0 and inclusion_met_count == 0
            
            # Only exclude if:
            # 1. There's an explicit exclusion violation, OR
            # 2. ALL PICO and ALL inclusion criteria explicitly failed (clearly irrelevant)
            should_exclude = has_exclusion_violation or (pico_all_failed and inclusion_all_failed)
            
            # Default to INCLUDE - be permissive!
            if should_exclude:
                decision = "EXCLUDE"
                
                # Build exclusion reason
                parts = []
                
                if has_exclusion_violation:
                    for v in violated_exclusions:
                        parts.append(f"Excluded: {v.criterion}")
                
                if pico_all_failed and inclusion_all_failed:
                    parts.append("Zero relevance detected across all criteria")
                
                decision_reason = " | ".join(parts) if parts else "Excluded by screening criteria."
            else:
                decision = "INCLUDE"
                
                # Build inclusion reason
                parts = []
                
                if pico_votes:
                    if pico_met_count == len(pico_votes):
                        parts.append(f"All PICO criteria met ({pico_met_count}/{len(pico_votes)})")
                    elif pico_met_count >= 1:
                        parts.append(f"Partial PICO match ({pico_met_count}/{len(pico_votes)})")
                    else:
                        parts.append("PICO: No explicit match (kept for review)")
                
                if inclusion_votes:
                    if inclusion_met_count == len(inclusion_votes):
                        parts.append(f"All inclusion criteria met ({inclusion_met_count}/{len(inclusion_votes)})")
                    elif inclusion_met_count >= 1:
                        parts.append(f"Partial inclusion match ({inclusion_met_count}/{len(inclusion_votes)})")
                    else:
                        parts.append("Inclusion: No explicit match (kept for review)")
                
                if not parts:
                    parts.append("Kept for full-text review")
                
                decision_reason = " | ".join(parts)
            
            # Build traceability record with all votes
            agent_trace = {}
            all_agents_passed = True
            for vote in paper_votes:
                agent_trace[vote.agent_name] = {
                    "type": vote.agent_type,
                    "criterion": vote.criterion,
                    "vote": "PASS" if vote.met else "FAIL",
                    "confidence": vote.confidence,
                    "evidence": vote.evidence,
                    "reasoning": vote.reasoning
                }
                if not vote.met:
                    all_agents_passed = False
            
            # If all agents passed, definitely include
            if all_agents_passed and len(paper_votes) > 0:
                decision = "INCLUDE"
                decision_reason = "All criteria passed"
            
            results.append({
                "paper_id": paper.id,
                "Source": paper.source,
                "Title": paper.title,
                "URL": paper.url,
                "Abstract": paper.abstract,
                "Decision": decision,
                "Reason": decision_reason,
                "Agent_Count": len(paper_votes),
                "Agent_Trace": agent_trace
            })
        
        return results
    
    def get_agent_summary(self) -> Dict[str, int]:
        """Return summary of specialist agents created."""
        return {
            "pico_agents": sum(1 for a in self.agents if a.agent_type.startswith('PICO')),
            "inclusion_agents": sum(1 for a in self.agents if a.agent_type == 'INCLUSION'),
            "exclusion_agents": sum(1 for a in self.agents if a.agent_type == 'EXCLUSION'),
            "total_agents": len(self.agents)
        }


# =============================================================================
# FULL-TEXT MULTI-AGENT SCREENING ARCHITECTURE
# =============================================================================
#
# 1. ORCHESTRATOR: Decomposes inclusion/exclusion criteria into N specialist agents
# 2. WORKER TIER: Each agent evaluates ONE paper for ONE criterion
# 3. AGGREGATOR: Consensus engine with inclusion AND + exclusion veto logic
# 4. TRACEABILITY: Per-agent votes for audit trail
#
# Used for full-text evidence screening
#
# =============================================================================

@dataclass
class FullTextAgentVote:
    """Represents a single criterion agent's vote on a paper."""
    agent_name: str
    agent_type: str  # "INCLUSION" or "EXCLUSION"
    criterion: str
    paper_id: str
    met: bool  # True = criterion satisfied (good for inclusion, bad for exclusion)
    confidence: float
    evidence: str
    reasoning: str


class FullTextCriterionAgent:
    """
    WORKER AGENT: Specialist that evaluates ONE criterion on ONE paper.
    Thread-safe with explicit API key passing.
    """
    
    def __init__(self, name: str, agent_type: str, criterion: str, model_name: str, api_keys: dict = None):
        self.name = name
        self.agent_type = agent_type  # "INCLUSION" or "EXCLUSION"
        self.criterion = criterion
        self.model_name = model_name
        self.api_keys = api_keys or {}
        
        # Initialize model with explicit keys for thread safety
        if self.api_keys:
            self.model = AIService.get_model_with_keys(model_name, self.api_keys)
        else:
            self.model = AIService.get_model(model_name)
    
    def evaluate_paper(self, paper: Dict[str, Any]) -> FullTextAgentVote:
        """Evaluate a single paper against this agent's criterion."""
        if not self.model:
            return self._fallback_vote(paper)
        
        paper_text = f"""
        TITLE: {paper.get('Title', 'N/A')}
        ABSTRACT: {paper.get('Abstract', 'N/A')}
        SOURCE: {paper.get('Source', 'N/A')}
        """
        
        # Different logic for inclusion vs exclusion
        if self.agent_type == "INCLUSION":
            prompt = f"""You are a specialist screening agent evaluating INCLUSION criteria.

YOUR IDENTITY: {self.name}
YOUR INCLUSION CRITERION: {self.criterion}

TASK: Evaluate whether this paper MEETS this specific inclusion criterion.

PAPER TO EVALUATE:
{paper_text[:2000]}

EVALUATION RULES:
1. Focus ONLY on your assigned criterion
2. The paper must CLEARLY meet this criterion to PASS
3. Look for specific evidence in the title and abstract
4. Be thorough - this is full-text screening

OUTPUT FORMAT - Return a JSON object:
{{
    "met": true or false,
    "confidence": 0.0-1.0,
    "evidence": "Exact quote from text supporting your decision",
    "reasoning": "Detailed explanation of why the paper does or does not meet this criterion"
}}

JSON OBJECT:"""
        else:  # EXCLUSION
            prompt = f"""You are a specialist screening agent evaluating EXCLUSION criteria.

YOUR IDENTITY: {self.name}
YOUR EXCLUSION CRITERION: {self.criterion}

TASK: Evaluate whether this paper VIOLATES this specific exclusion criterion.

PAPER TO EVALUATE:
{paper_text[:2000]}

EVALUATION RULES:
1. Focus ONLY on your assigned criterion
2. The paper must CLEARLY violate this criterion to FAIL
3. Look for specific evidence in the title and abstract
4. Be thorough - this is full-text screening

OUTPUT FORMAT - Return a JSON object:
{{
    "met": true or false,  // true = paper violates the exclusion (bad), false = does not violate (good)
    "confidence": 0.0-1.0,
    "evidence": "Exact quote from text supporting your decision",
    "reasoning": "Detailed explanation of why the paper does or does not violate this criterion"
}}

JSON OBJECT:"""
        
        try:
            from langchain_core.messages import HumanMessage
            response = self.model.invoke([HumanMessage(content=prompt)])
            return self._parse_response(response.content, paper)
        except Exception as e:
            print(f"Agent {self.name} error: {e}")
            return self._fallback_vote(paper)
    
    def _parse_response(self, content: str, paper: Dict[str, Any]) -> FullTextAgentVote:
        """Parse agent response into FullTextAgentVote object."""
        try:
            # Extract JSON object
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end+1]
                data = json.loads(json_str)
                
                return FullTextAgentVote(
                    agent_name=self.name,
                    agent_type=self.agent_type,
                    criterion=self.criterion,
                    paper_id=paper.get('paper_id', paper.get('ID', 'unknown')),
                    met=data.get('met', False),
                    confidence=data.get('confidence', 0.5),
                    evidence=data.get('evidence', 'No specific evidence'),
                    reasoning=data.get('reasoning', 'No reasoning provided')
                )
            else:
                print(f"Agent {self.name}: No JSON object found in response")
        except Exception as e:
            print(f"Agent {self.name} parse error: {e}")
            print(f"Raw content: {content[:500]}")
        
        return self._fallback_vote(paper)
    
    def _fallback_vote(self, paper: Dict[str, Any]) -> FullTextAgentVote:
        """Create a permissive fallback vote when model fails."""
        return FullTextAgentVote(
            agent_name=self.name,
            agent_type=self.agent_type,
            criterion=self.criterion,
            paper_id=paper.get('paper_id', paper.get('ID', 'unknown')),
            met=True,  # Default to PASS (inclusion met, exclusion not violated)
            confidence=0.5,
            evidence="Model fallback - permissive default",
            reasoning=f"Agent {self.name} could not evaluate, using permissive default"
        )


class FullTextOrchestrator:
    """
    ORCHESTRATOR: Full-text screening with fan-out multi-agent architecture.
    
    1. DECOMPOSITION: Creates one agent per inclusion/exclusion criterion
    2. FAN-OUT: All agents evaluate the paper in parallel
    3. AGGREGATION: ALL inclusion must pass AND no exclusion can fail
    """
    
    def __init__(self, inclusion_criteria: List[str], exclusion_criteria: List[str],
                 model_name: str, api_keys: dict = None):
        self.inclusion = [c for c in inclusion_criteria if c and c.strip()]
        self.exclusion = [c for c in exclusion_criteria if c and c.strip()]
        self.model_name = model_name
        self.api_keys = api_keys or {}
        
        # Create specialist agents
        self.agents: List[FullTextCriterionAgent] = []
        self._create_agents()
    
    def _create_agents(self):
        """Decompose criteria into specialist agents."""
        # Create inclusion agents
        for i, criterion in enumerate(self.inclusion):
            self.agents.append(FullTextCriterionAgent(
                f"Inclusion Agent {i+1}", "INCLUSION", criterion, self.model_name, self.api_keys
            ))
        
        # Create exclusion agents
        for i, criterion in enumerate(self.exclusion):
            self.agents.append(FullTextCriterionAgent(
                f"Exclusion Agent {i+1}", "EXCLUSION", criterion, self.model_name, self.api_keys
            ))
    
    def screen_paper(self, paper: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
        """
        ORCHESTRATOR: Execute full multi-agent screening on a single paper.
        
        Returns paper result with per-agent votes and final decision.
        """
        if not self.agents:
            return {
                "decision": "Include",
                "reason": "No criteria specified - defaulting to include",
                "citation": "N/A"
            }
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from config import Config
        
        total_agents = len(self.agents)
        all_votes: List[FullTextAgentVote] = []
        
        # FAN-OUT: Run all agents in parallel on the single paper
        with ThreadPoolExecutor(max_workers=min(Config.PARALLEL_AGENT_WORKERS, total_agents)) as executor:
            future_to_agent = {
                executor.submit(agent.evaluate_paper, paper): agent 
                for agent in self.agents
            }
            
            completed_agents = 0
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    vote = future.result()  # Single FullTextAgentVote
                    all_votes.append(vote)
                except Exception as exc:
                    print(f"Agent {agent.name} failed: {exc}")
                
                completed_agents += 1
                if progress_callback:
                    progress_callback(completed_agents, total_agents,
                                    f"Agent {completed_agents}/{total_agents}: {agent.name}")
        
        # AGGREGATE: Apply consensus logic
        return self._aggregate_results(paper, all_votes)
    
    def _aggregate_results(self, paper: Dict[str, Any], all_votes: List[FullTextAgentVote]) -> Dict[str, Any]:
        """
        AGGREGATOR: Full-text consensus engine with traceability.
        
        Logic:
        - ALL inclusion agents must PASS (met=True)
        - NO exclusion agents can FAIL (met=True means violation)
        """
        # Separate votes by type
        inclusion_votes = [v for v in all_votes if v.agent_type == "INCLUSION"]
        exclusion_votes = [v for v in all_votes if v.agent_type == "EXCLUSION"]
        
        # Apply consensus logic
        all_inclusion_pass = all(v.met for v in inclusion_votes) if inclusion_votes else True
        no_exclusion_violated = all(not v.met for v in exclusion_votes) if exclusion_votes else True
        
        # Final decision
        if all_inclusion_pass and no_exclusion_violated:
            decision = "Include"
        else:
            decision = "Exclude"
        
        # Build comprehensive reasoning
        if decision == "Include":
            # All criteria satisfied
            passed_inclusions = [v for v in inclusion_votes if v.met]
            
            parts = ["This paper meets all inclusion criteria and violates no exclusion criteria."]
            
            if passed_inclusions:
                parts.append("The paper demonstrates:")
                for v in passed_inclusions:
                    parts.append(f"- {v.criterion}: {v.reasoning}")
            
            if exclusion_votes:
                parts.append("The paper avoids all exclusion criteria:")
                for v in exclusion_votes:
                    parts.append(f"- {v.criterion}: No violation found")
            
            decision_reason = " ".join(parts)
        else:
            # Some criteria failed
            failed_inclusions = [v for v in inclusion_votes if not v.met]
            violated_exclusions = [v for v in exclusion_votes if v.met]
            
            parts = []
            
            if failed_inclusions:
                parts.append("This paper is excluded because it fails to meet the following inclusion criteria:")
                for v in failed_inclusions:
                    parts.append(f"- {v.criterion}: {v.reasoning} Evidence: {v.evidence}")
            
            if violated_exclusions:
                if parts:
                    parts.append("Additionally, it violates the following exclusion criteria:")
                else:
                    parts.append("This paper is excluded because it violates the following exclusion criteria:")
                for v in violated_exclusions:
                    parts.append(f"- {v.criterion}: {v.reasoning} Evidence: {v.evidence}")
            
            decision_reason = " ".join(parts)
        
        # Build traceability record
        agent_trace = {}
        for vote in all_votes:
            agent_trace[vote.agent_name] = {
                "type": vote.agent_type,
                "criterion": vote.criterion,
                "vote": "PASS" if vote.met else "FAIL",
                "confidence": vote.confidence,
                "evidence": vote.evidence,
                "reasoning": vote.reasoning
            }
        
        # Build criteria evaluations for compatibility with existing UI
        criteria_evals = {}
        for vote in all_votes:
            criteria_evals[vote.criterion] = "INCLUDE" if vote.met else "EXCLUDE"
        
        return {
            "decision": decision,
            "reason": decision_reason,
            "citation": "See agent trace for specific quotes",
            "agent_trace": agent_trace,
            "inclusion_score": f"{sum(v.met for v in inclusion_votes)}/{len(inclusion_votes)}",
            "exclusion_violations": f"{sum(v.met for v in exclusion_votes)}/{len(exclusion_votes)}",
            "agent_count": len(all_votes),
            **criteria_evals  # Flatten for compatibility
        }
    
    def screen_papers_batch(self, papers: List[Dict[str, Any]], progress_callback=None, 
                           paper_progress_callback=None) -> List[Dict[str, Any]]:
        """
        BATCH ORCHESTRATOR: Execute multi-agent screening on multiple papers in parallel.
        
        This parallelizes at the paper level, so multiple papers are screened concurrently.
        Each paper still has its agents running in parallel via screen_paper().
        
        Args:
            papers: List of paper dictionaries to screen
            progress_callback: Called with (completed_papers, total_papers, status_message)
            paper_progress_callback: Called with (completed_agents, total_agents, agent_message) per paper
        
        Returns:
            List of screening results, one per paper
        """
        if not papers:
            return []
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from config import Config
        
        total_papers = len(papers)
        results = [None] * total_papers  # Pre-allocate to maintain order
        completed = 0
        
        # Use configured workers for Ollama parallel processing
        # Ollama can handle multiple concurrent requests, but limit depends on model size and VRAM
        max_workers = min(Config.PARALLEL_SCREENING_WORKERS, total_papers)
        
        def screen_single(paper_idx: int, paper: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
            """Screen a single paper and return its index + result."""
            try:
                result = self.screen_paper(paper, progress_callback=paper_progress_callback)
                return paper_idx, result
            except Exception as e:
                print(f"Error screening paper {paper.get('Title', 'Unknown')}: {e}")
                return paper_idx, {
                    "decision": "Exclude",
                    "reason": f"Screening error: {str(e)[:100]}",
                    "citation": "N/A",
                    "agent_trace": {},
                    "inclusion_score": "N/A",
                    "exclusion_violations": "N/A"
                }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all papers
            future_to_idx = {
                executor.submit(screen_single, idx, paper): idx 
                for idx, paper in enumerate(papers)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx, result = future.result()
                results[idx] = result
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total_papers, 
                                    f"Screened {completed}/{total_papers} papers")
        
        return results
    
    def get_agent_summary(self) -> Dict[str, int]:
        """Return summary of specialist agents created."""
        return {
            "inclusion_agents": sum(1 for a in self.agents if a.agent_type == "INCLUSION"),
            "exclusion_agents": sum(1 for a in self.agents if a.agent_type == "EXCLUSION"),
            "total_agents": len(self.agents)
        }