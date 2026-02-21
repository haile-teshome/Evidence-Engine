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

# utils.py

class AIService:
    @staticmethod
    def get_model(model_name: str):
        """Initializes model based on selection."""
        name_lower = model_name.lower()
        try:
            # Check for Cloud Providers first
            if "gpt" in name_lower:
                return ChatOpenAI(model=model_name, api_key=Config.OPENAI_API_KEY, temperature=0)
            elif "claude" in name_lower:
                return ChatAnthropic(model=model_name, api_key=Config.ANTHROPIC_API_KEY, temperature=0)
            
            # DEFAULT/LOCAL: Use ChatOllama for everything else (like llama3)
            # We add a base_url to ensure it connects to your local instance
            return ChatOllama(
                model=model_name, 
                temperature=0,
                base_url="http://localhost:11434" # Explicitly point to local Ollama
            )
        except Exception as e:
            # STOP SILENCING ERRORS: Tell the user why it failed
            st.error(f"🤖 AI Connection Error: {str(e)}")
            return None

    @staticmethod
    def _extract_json(text: str) -> Optional[Any]:
        """Robust JSON extraction to prevent 'AI Processing Errors'."""
        try:
            # Remove markdown blocks if present
            clean_text = text.replace("```json", "").replace("```", "").strip()
            
            # Find the actual JSON boundaries to ignore conversational 'chatter'
            start_idx = min(clean_text.find('{'), clean_text.find('['))
            end_idx = max(clean_text.rfind('}'), clean_text.rfind(']'))
            
            if start_idx != -1 and end_idx != -1:
                json_str = clean_text[start_idx:end_idx+1]
                return json.loads(json_str)
            
            return json.loads(clean_text)
        except Exception as e:
            st.warning(f"⚠️ JSON Parsing Error: AI returned invalid format.")
            return None
            
    @staticmethod
    def infer_pico_and_query(goal: str, model_name: str, previous_goal: str = "") -> Dict[str, Any]:
        """Extracts PICO and criteria from the research goal with enhanced specificity."""
        model = AIService.get_model(model_name)
        
        system_msg = SystemMessage(content="You are a medical data extractor. Extract PICO elements from the research goal. Be specific and descriptive based on the research context.")
        
        prompt = f"""
        Current Research Goal: "{goal}"
        
        Extract these elements into a JSON object. Be specific and descriptive based on the research context:
        {{
            "p": "Specific target population/condition",
            "i": "Specific test or intervention",
            "c": "Specific control group or baseline",
            "o": "Specific outcome measured",
            "inclusion": ["specific", "actionable", "rules", "for", "including", "studies"],
            "exclusion": ["specific", "actionable", "rules", "for", "excluding", "studies"]
        }}
        
        Generate 5-8 specific inclusion criteria and 5-8 specific exclusion criteria that are directly relevant to this research question.
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
                
                return {
                    "p": data.get("p") or data.get("population", goal),
                    "i": data.get("i") or data.get("intervention", "N/A"),
                    "c": data.get("c") or data.get("comparator", "N/A"),
                    "o": data.get("o") or data.get("outcome", "N/A"),
                    "inclusion": inclusion[:8],  # Limit to 8 criteria
                    "exclusion": exclusion[:8]   # Limit to 8 criteria
                }
        except Exception as e:
            print(f"PICO analysis error: {e}")
            pass
        
        # Enhanced fallback with basic research question analysis
        goal_lower = goal.lower()
        
        # Extract key terms for more specific fallback
        population = "Target population"
        if any(word in goal_lower for word in ["children", "pediatric", "kids", "childhood"]):
            population = "Children and adolescents"
        elif any(word in goal_lower for word in ["elderly", "older adults", "senior", "geriatric"]):
            population = "Older adults"
        elif any(word in goal_lower for word in ["patients", "disease", "condition", "disorder"]):
            population = "Patients with medical condition"
        elif any(word in goal_lower for word in ["students", "education", "learning"]):
            population = "Students"
        elif any(word in goal_lower for word in ["workers", "employees", "workplace", "occupational"]):
            population = "Working adults"
        
        intervention = "Intervention"
        if any(word in goal_lower for word in ["therapy", "treatment", "intervention", "program"]):
            intervention = "Therapeutic program"
        elif any(word in goal_lower for word in ["medication", "drug", "pharmaceutical"]):
            intervention = "Medication treatment"
        elif any(word in goal_lower for word in ["surgery", "surgical", "operation"]):
            intervention = "Surgical procedure"
        elif any(word in goal_lower for word in ["exercise", "physical activity", "fitness"]):
            intervention = "Exercise program"
        elif any(word in goal_lower for word in ["diet", "nutrition", "food"]):
            intervention = "Dietary intervention"
        elif any(word in goal_lower for word in ["education", "training", "teaching"]):
            intervention = "Educational program"
        
        comparator = "Control or comparator"
        if any(word in goal_lower for word in ["control", "placebo", "sham"]):
            comparator = "Placebo control"
        elif any(word in goal_lower for word in ["usual", "standard", "routine"]):
            comparator = "Standard care"
        elif any(word in goal_lower for word in ["no", "without", "untreated"]):
            comparator = "No treatment"
        
        outcome = "Outcomes measured"
        if any(word in goal_lower for word in ["depression", "anxiety", "mental", "psychological"]):
            outcome = "Mental health outcomes"
        elif any(word in goal_lower for word in ["cardiovascular", "heart", "blood pressure"]):
            outcome = "Cardiovascular outcomes"
        elif any(word in goal_lower for word in ["diabetes", "glucose", "insulin", "blood sugar"]):
            outcome = "Diabetes outcomes"
        elif any(word in goal_lower for word in ["pain", "discomfort", "symptoms"]):
            outcome = "Pain and symptom outcomes"
        elif any(word in goal_lower for word in ["quality of life", "well-being", "satisfaction"]):
            outcome = "Quality of life outcomes"
        elif any(word in goal_lower for word in ["performance", "function", "ability"]):
            outcome = "Functional outcomes"
        
        # Generate basic but more specific criteria
        inclusion_criteria = [
            "Studies relevant to this research question",
            "Population appropriate for this research",
            "Intervention appropriate for this research",
            "Outcomes relevant to this research",
            "Study design appropriate for this research"
        ]
        
        exclusion_criteria = [
            "Studies not relevant to this research question",
            "Population not appropriate for this research",
            "Intervention not appropriate for this research",
            "Outcomes not relevant to this research",
            "Study design not appropriate for this research"
        ]
        
        return {
            "p": population, 
            "i": intervention, 
            "c": comparator, 
            "o": outcome, 
            "inclusion": inclusion_criteria, 
            "exclusion": exclusion_criteria
        }

    @staticmethod
    def generate_mesh_query(pico: PICOCriteria, model_name: str) -> str:
        """Generates a high-sensitivity (broad) PubMed search string."""
        model = AIService.get_model(model_name)
        
        prompt = f"""
        You are an expert Information Specialist. Convert this PICO into a high-sensitivity PubMed search string.
        
        PICO:
        - Population: {pico.population}
        - Intervention: {pico.intervention}
        
        CRITICAL REQUIREMENTS:
        1. Return ONLY the PubMed search string - no explanations, no examples, no extra text
        2. Use proper PubMed syntax with [Mesh] and [tiab] tags
        3. Use OR between synonyms, AND between concepts
        4. Include wildcards (*) where appropriate
        5. Do NOT include outcomes or comparators
        
        EXAMPLE: ("Diabetes Mellitus"[Mesh] OR "diabet*"[tiab]) AND ("Metformin"[Mesh] OR "metformin"[tiab])
        
        Generate a clean, executable PubMed search string for this PICO:
        """
        
        try:
            messages = [HumanMessage(content=prompt)]
            response = model.invoke(messages)
            raw_query = response.content.strip()
            
            # Clean up the response to extract only the search string
            # Remove any markdown formatting
            clean_query = raw_query.replace("```sql", "").replace("```", "").replace("```", "")
            
            # Remove any explanatory text before or after the actual query
            # Look for patterns that indicate the start of the actual query
            query_patterns = [
                r'^(.*?\()(.*)',  # Anything starting with parenthesis
                r'^"([^"]*".*)',   # Anything starting with quoted term
                r'^\((.*)',        # Anything starting with opening parenthesis
            ]
            
            for pattern in query_patterns:
                match = re.match(pattern, clean_query.strip(), re.DOTALL)
                if match:
                    # Extract the query part
                    if match.groups():
                        query_part = match.group(1) if len(match.groups()) == 1 else match.group(0)
                        # Ensure it ends properly
                        if not query_part.endswith(')') and '(' in query_part:
                            query_part += ')'
                        return query_part.strip()
            
            # If no pattern matches, try to find the first parenthesis
            if '(' in clean_query:
                start_idx = clean_query.find('(')
                query_part = clean_query[start_idx:]
                # Ensure it ends properly
                if not query_part.endswith(')') and query_part.count('(') > query_part.count(')'):
                    query_part += ')'
                return query_part.strip()
            
            # If still no match, return the cleaned version but validate it
            if clean_query and ('[' in clean_query or '"' in clean_query):
                return clean_query.strip()
            
        except Exception as e:
            print(f"Query generation error: {e}")
        
        # Fallback: Generate a basic query manually
        pop_terms = pico.population.split()[:2] if pico.population else ["adults"]
        int_terms = pico.intervention.split()[:2] if pico.intervention else ["treatment"]
        
        pop_query = " OR ".join([f'"{term}"[tiab]' for term in pop_terms])
        int_query = " OR ".join([f'"{term}"[tiab]' for term in int_terms])
        
        return f"({pop_query}) AND ({int_query})"

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
                
                # Extract JSON
                import re
                json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    results[source] = {
                        "query": data.get("optimized_query", current_query),
                        "yield": data.get("estimated_yield", 0)
                    }
                else:
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
        
        prompt = f"""
        STRICTLY screen this paper based on PICO and Inclusion/Exclusion Criteria.
        
        PICO:
        - Pop: {pico.population} | Int: {pico.intervention} | Comp: {pico.comparator} | Out: {pico.outcome}
        
        CRITERIA:
        Inclusion: {inc_text}
        Exclusion: {excl_text}

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
                    return {
                        "decision": "Exclude", 
                        "bucket": "Missing content",
                        "reason": "Missing title or abstract",
                        **{criterion: "EXCLUDE" for criterion in inclusion_criteria + exclusion_criteria}
                    }
                
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

        inc_text = ", ".join(inclusion_criteria) if inclusion_criteria else "None specified"
        excl_text = ", ".join(exclusion_criteria) if exclusion_criteria else "None specified"

        prompt = f"""
        You are performing the Full-Text Eligibility phase of a Systematic Review.
        
        PICO:
        Population: {pico.population}
        Intervention: {pico.intervention}
        Comparator: {pico.comparator}
        Outcome: {pico.outcome}
        
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
        For EACH criterion in the inclusion and exclusion lists above, evaluate whether the paper meets it.
        Return "INCLUDE" if the paper meets the criterion, "EXCLUDE" if it does not.
        
        AI REASONING SUMMARY:
        Provide a 1-2 sentence summary explaining the main reason for inclusion or exclusion.
        Focus on which specific criteria were met or violated.

        REQUIRED JSON FORMAT:
        {{
            "decision": "Include" or "Exclude",
            "reason": "Brief 1-2 sentence summary of why included/excluded",
            "citation": "Direct quote from text (max 100 words)...",
            "criteria_evaluations": {{
                "Inclusion Criterion 1": "INCLUDE" or "EXCLUDE",
                "Inclusion Criterion 2": "INCLUDE" or "EXCLUDE",
                ...
                "Exclusion Criterion 1": "INCLUDE" or "EXCLUDE",
                "Exclusion Criterion 2": "INCLUDE" or "EXCLUDE"
            }}
        }}
        
        You MUST provide: decision, reason (brief summary), citation, and criteria_evaluations for ALL criteria.
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
        """Removes MeSH tags for general search engines and simplifies complex queries."""
        # Remove MeSH tags
        cleaned = re.sub(r'\[.*?\]', '', query)
        
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
        
        # Return a simplified version for better arXiv compatibility
        return " ".join(cleaned.split())

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