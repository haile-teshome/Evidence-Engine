import re
import json
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage 

from config import Config
from models import Paper, PICOCriteria, ScreeningResult

class AIService:
    @staticmethod
    def get_model(model_name: str):
        """Initializes model based on selection."""
        name_lower = model_name.lower()
        try:
            if "gpt" in name_lower:
                return ChatOpenAI(model=model_name, api_key=Config.OPENAI_API_KEY, temperature=0)
            elif "claude" in name_lower:
                return ChatAnthropic(model=model_name, api_key=Config.ANTHROPIC_API_KEY, temperature=0)
            else:
                return ChatOllama(model=model_name, temperature=0)
        except Exception as e:
            st.error(f"Error initializing model {model_name}: {e}")
            return None
            
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
        """Enhanced helper to find JSON lists or objects."""
        try:
            clean_text = text.replace("```json", "").replace("```", "").strip()
            start_idx = min(clean_text.find('{'), clean_text.find('['))
            end_idx = max(clean_text.rfind('}'), clean_text.rfind(']'))
            
            if start_idx != -1 and end_idx != -1:
                json_str = clean_text[start_idx:end_idx+1]
                return json.loads(json_str)
            return json.loads(clean_text)
        except Exception:
            return None

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
    def generate_formal_question(pico: PICOCriteria, model_name: str, history: list) -> str:
        """Refines the research question by building on previous iterations."""
        model = AIService.get_model(model_name)
        
        past_questions = [h['formal_question'] for h in history if 'formal_question' in h]
        history_context = "\n".join([f"- Iteration {i+1}: {q}" for i, q in enumerate(past_questions)])
        
        prompt = f"""
        You are an expert Clinical Research Librarian. 
        Task: Refine the current research question based on new user input and previous iterations.

        PREVIOUS ITERATIONS:
        {history_context if history_context else "None (This is the first draft)"}

        CURRENT UPDATED PICO:
        - Population: {pico.population}
        - Intervention: {pico.intervention}
        - Comparator: {pico.comparator}
        - Outcome: {pico.outcome}

        GOAL:
        Synthesize a single, formal research question. 
        - If the user provided feedback in the latest turn, ensure the new question reflects that adjustment.
        - Maintain the "In [P], does [I] compared to [C] result in [O]?" structure.
        - Ensure it is more specific and refined than the previous versions.
        - Don't return any preamble or filler like "Based on the input here is your research question".
    
        Return ONLY the refined question.
        """
        
        try:
            from langchain_core.messages import HumanMessage
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content.strip().strip('"')
        except Exception:
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
    def screen_paper(paper: Paper, pico: PICOCriteria, model_name: str, inclusion: List[str] = None, exclusion: List[str] = None) -> Dict[str, Any]:
        """Strictly screen this paper based on PICO and Inclusion/Exclusion Criteria."""
        model = AIService.get_model(model_name)
        if not model:
            return {"decision": "Exclude", "bucket": "Error", "reason": "Model init failed"}
        
        # Ensure we have criteria text to send to the AI
        inc_text = inclusion if inclusion else getattr(pico, 'inclusion_criteria', "None specified")
        excl_text = exclusion if exclusion else getattr(pico, 'exclusion_criteria', "None specified")
        
        prompt = f"""
        Strictly screen this paper based on PICO and I/E Criteria.
        
        PICO:
        - Pop: {pico.population} | Int: {pico.intervention} | Comp: {pico.comparator} | Out: {pico.outcome}
        
        CRITERIA:
        Inclusion: {inc_text}
        Exclusion: {excl_text}

        PAPER:
        Title: {paper.title}
        Abstract: {paper.abstract}

        TASK:
        1. Decision: "Include" or "Exclude".
        2. Bucket: Select 3-5 words from the criteria above that best explains the decision (e.g., "Wrong Population").
        3. Reason: Brief explanation.

        RETURN ONLY JSON:
        {{"decision": "Exclude", "bucket": "Wrong Population", "reason": "Focuses on children, not adults."}}
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
                return {
                    "decision": data.get('decision', 'Exclude'),
                    "bucket": data.get('bucket', 'Criteria mismatch'),
                    "reason": data.get('reason', 'N/A')
                }
        except Exception as e:
            # This will show you exactly why it's failing in your terminal
            print(f"DEBUG: Screening error for {paper.title[:30]}: {e}")
            
        return {
            "decision": "Exclude", 
            "bucket": "Error",
            "reason": "AI Processing error"
        }

    @staticmethod
    def screen_full_text(text: str, pico: PICOCriteria, model_name: str) -> Dict[str, Any]:
        """Performs deeper eligibility screening on full-text or detailed abstracts."""
        model = AIService.get_model(model_name)
        if not model:
            return {"decision": "Exclude", "reason": "Model error", "citation": "N/A"}

        prompt = f"""
        You are performing the Eligibility phase of a Systematic Review.
        
        CRITERIA:
        Population: {pico.population}
        Intervention: {pico.intervention}
        Comparator: {pico.comparator}
        Outcome: {pico.outcome}

        TEXT TO ANALYZE:
        {text[:4000]} 

        TASK:
        Decide if this paper should be included. 
        Provide a 'citation' which is a direct quote from the text that supports your decision.

        RETURN ONLY JSON:
        {{
            "decision": "✅ Include" or "❌ Exclude",
            "reason": "Short reason (e.g. Wrong Study Design)",
            "citation": "Direct quote from the text..."
        }}
        """

        try:
            from langchain_core.messages import HumanMessage
            import re
            import json
            
            response = model.invoke([HumanMessage(content=prompt)])
            # Use regex to find JSON in case the model adds preamble text
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            print(f"Full-text error: {e}")
            
        return {
            "decision": "❌ Exclude",
            "reason": "AI Processing error",
            "citation": "N/A"
        }
        
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