import re
import json
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional

# Modern LangChain v0.3 Imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage 

from config import Config
from models import Paper, PICOCriteria, ScreeningResult
from langchain_google_genai import ChatGoogleGenerativeAI

class AIService:
    # @staticmethod
    # def _extract_json(text: str) -> Optional[Any]:
    #     """Helper to extract and parse JSON from AI strings."""
    #     try:
    #         # Look for JSON structure within the text
    #         match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    #         if match:
    #             return json.loads(match.group(1))
    #         return json.loads(text)
    #     except (json.JSONDecodeError, AttributeError):
    #         return None

    @staticmethod
    def get_model(model_name: str):
            """Initializes model based on selection."""
            name_lower = model_name.lower()
            try:
                if "gpt" in name_lower:
                    return ChatOpenAI(model=model_name, api_key=Config.OPENAI_API_KEY, temperature=0)
                elif "claude" in name_lower:
                    return ChatAnthropic(model=model_name, api_key=Config.ANTHROPIC_API_KEY, temperature=0)
                elif "gemini" in name_lower:
                    # New Gemini Integration
                    return ChatGoogleGenerativeAI(
                        model=model_name, 
                        google_api_key=Config.GEMINI_API_KEY, 
                        temperature=0
                    )
                else:
                    return ChatOllama(model=model_name, temperature=0)
            except Exception as e:
                st.error(f"Error initializing model {model_name}: {str(e)}")
                return ChatOllama(model=Config.DEFAULT_MODEL, temperature=0)

    @staticmethod
    def infer_pico_and_query(goal: str, model_name: str, previous_goal: str = "") -> Dict[str, Any]:
        """Strictly extracts PICO from the NEW goal only, preventing drift."""
        model = AIService.get_model(model_name)
        
        system_msg = SystemMessage(content="You are a precise medical data extractor. Focus ONLY on the current research goal.")
        
        prompt = f"""
        Current Research Goal: "{goal}"
        
        Extract these elements into a JSON object:
        {{
            "p": "Target population/condition",
            "i": "The test or intervention",
            "c": "Control group or baseline",
            "o": "Outcome measured",
            "inclusion": ["list", "of", "rules"],
            "exclusion": ["list", "of", "rules"]
        }}
        """
        try:
            response = model.invoke([system_msg, HumanMessage(content=prompt)])
            data = AIService._extract_json(response.content)
            
            if data:
                return {
                    "p": data.get("p") or data.get("population", goal),
                    "i": data.get("i") or data.get("intervention", "N/A"),
                    "c": data.get("c") or data.get("comparator", "N/A"),
                    "o": data.get("o") or data.get("outcome", "N/A"),
                    "inclusion": data.get("inclusion", []),
                    "exclusion": data.get("exclusion", [])
                }
        except Exception:
            pass
        return {"p": goal, "i": "N/A", "c": "N/A", "o": "N/A", "inclusion": [], "exclusion": []}

    @staticmethod
    def generate_mesh_query(pico: PICOCriteria, model_name: str) -> str:
        """Generates a high-sensitivity (broad) PubMed search string."""
        model = AIService.get_model(model_name)
        
        prompt = f"""
        You are an expert Information Specialist. Convert this PICO into a high-sensitivity PubMed search string.
        
        PICO:
        - Population: {pico.population}
        - Intervention: {pico.intervention}
        
        RULES:
        1. Use the [Mesh] tag for major concepts, but ALWAYS pair it with [tiab] (Title/Abstract) synonyms to ensure broad coverage.
        2. Use the 'OR' operator between synonyms of the same concept.
        3. Use 'AND' only between the Population and Intervention. 
        4. DO NOT include Outcomes or Comparators in the search string (this makes it too narrow).
        5. Use wildcards (*) where appropriate (e.g., "diabet*").
        
        EXAMPLE FORMAT:
        ("Diabetes Mellitus"[Mesh] OR "Diabetes"[tiab] OR "diabet*"[tiab]) AND ("Metformin"[Mesh] OR "Metformin"[tiab])
        
        Return ONLY the search string. No preamble.
        """
        
        messages = [HumanMessage(content=prompt)]
        response = model.invoke(messages)
        return response.content.strip().replace("```sql", "").replace("```", "")

# --- Update these methods in AIService class within utils.py ---

    @staticmethod
    def generate_brainstorm_summary(goal: str, papers: List[Paper], model_name: str) -> str:
        """Provides a comprehensive summary of initial literature findings."""
        model = AIService.get_model(model_name)
        # Feed more context (titles + snippets) to the model for a better summary
        paper_context = "\n".join([f"- {p.title}: {p.abstract[:200]}..." for p in papers[:5]])
        
        prompt = f"""
        Research Goal: {goal}
        
        Current Literature Findings:
        {paper_context}
        
        Provide a comprehensive 3-4 sentence summary of the current landscape found in these results. 
        Focus on what is currently known, any immediate gaps observed, and how well these papers 
        align with the target population and intervention.
        """
        try:
            return model.invoke([HumanMessage(content=prompt)]).content
        except Exception:
            return "Analyzing literature alignment with research goals..."

    @staticmethod
    def get_refinement_suggestions(goal: str, papers: List[Paper], model_name: str) -> List[str]:
        """Generates specific refinement options based on found literature."""
        model = AIService.get_model(model_name)
        titles = "\n".join([f"- {p.title}" for p in papers[:5]])
        
        prompt = f"""
        Research Goal: {goal}
        Literature Found: {titles}
        
        Based on the literature above, suggest 3 specific ways to refine or narrow this research goal.
        Return ONLY a JSON list of strings.
        Example: ["Focus on pediatric populations", "Limit to randomized controlled trials", "Include specific biomarkers"]
        """
        try:
            response = model.invoke([HumanMessage(content=prompt)])
            suggestions = AIService._extract_json(response.content)
            return suggestions if isinstance(suggestions, list) else []
        except Exception:
            return ["Focus on specific study designs", "Narrow the target population", "Specify clinical outcomes"]


    @staticmethod
    def _extract_json(text: str) -> Optional[Any]:
        """Enhanced helper to find JSON even if the AI adds conversational filler."""
        try:
            # Try finding a JSON block first
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return json.loads(text)
        except Exception:
            # Fallback: Manual extraction if JSON parsing fails
            manual_data = {}
            for key in ['decision', 'design', 'sample_size', 'reason']:
                pattern = f'"{key}":\\s*"([^"]*)"'
                m = re.search(pattern, text, re.IGNORECASE)
                if m: manual_data[key] = m.group(1)
            return manual_data if manual_data else None

    @staticmethod
    def screen_paper(paper: Paper, pico: PICOCriteria, model_name: str, inclusion: List[str], exclusion: List[str]) -> dict:
        """Forcefully extracts Design, Sample Size, and Decision Reason."""
        model = AIService.get_model(model_name)
        
        prompt = f"""
        Strict Systematic Review Screening Task.
        
        GOAL: Determine if this paper fits the PICO and extract study metadata.
        
        PICO:
        - Pop: {pico.population} | Int: {pico.intervention} | Comp: {pico.comparator} | Out: {pico.outcome}
        
        RULES:
        - Inclusion: {inclusion}
        - Exclusion: {exclusion}
        
        PAPER:
        - Title: {paper.title}
        - Abstract: {paper.abstract}
        
        YOU MUST PROVIDE THESE 4 FIELDS:
        1. "Decision": Either "Include" or "Exclude".
        2. "Design": Specific study type (e.g. RCT, Cohort, Case-Control).
        3. "Sample_size": The number of subjects (e.g. N=200).
        4. "Reason": Why it was included or the specific criteria it failed.

        Return ONLY a JSON object. No intro, no outro.
        {{
            "Decision": "",
            "Design": "",
            "Sample_size": "",
            "Reason": ""
        }}
        """
        try:
            response = model.invoke([HumanMessage(content=prompt)])
            data = AIService._extract_json(response.content)
            
            if data:
                return {
                    "decision": data.get('Decision', 'Exclude'),
                    "design": data.get('Design', 'N/A'),
                    "sample_size": data.get('Sample_size', 'N/A'),
                    "reason": data.get('Reason', 'Check criteria')
                }
        except Exception as e:
            print(f"Screening error: {e}")
            
        # CHANGE THESE TO LOWERCASE TO MATCH ABOVE
        return {
            "decision": "Exclude", 
            "reason": "Error parsing response", 
            "design": "N/A", 
            "sample_size": "N/A"
        }
    @staticmethod
    def generate_search_query(pico: PICOCriteria, model_name: str) -> str:
        model = AIService.get_model(model_name)
        prompt = f"""
        Generate a single PubMed search string using MeSH terms for this PICO.
        RETURN ONLY THE RAW SEARCH STRING. 
        NO intro text, NO explanations, NO markdown code blocks.
        
        PICO: {pico.to_dict()}
        """
        response = model.invoke([HumanMessage(content=prompt)])
        # Strip common AI formatting (backticks and "Search String:" prefixes)
        clean = response.content.replace("```sql", "").replace("```", "").strip()
        if ":" in clean and "(" not in clean.split(":")[0]:
            clean = clean.split(":", 1)[-1].strip()
        return clean
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


class QueryCleaner:
    @staticmethod
    def clean_for_general_search(query: str) -> str:
        """Removes MeSH tags for general search engines."""
        cleaned = re.sub(r'\[.*?\]', '', query)
        cleaned = cleaned.replace('AND', ' ').replace('OR', ' ')
        return " ".join(cleaned.split())