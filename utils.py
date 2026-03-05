import re
import json
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
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
    def get_model_with_keys(model_name: str, keys: dict):
        """Thread-safe model initializer that doesn't rely on st.session_state."""
        name_lower = model_name.lower()
        if "gpt" in name_lower:
            return ChatOpenAI(model=model_name, api_key=keys.get('openai'), temperature=0)
        elif "claude" in name_lower:
            return ChatAnthropic(model=model_name, api_key=keys.get('anthropic'), temperature=0)
        # Add Gemini and Ollama logic following the same pattern...
        return ChatOllama(model=model_name, temperature=0, base_url="http://localhost:11434")
        
    @staticmethod
    def get_model(model_name: str):
        """Initializes model based on selection with user-provided API keys."""
        name_lower = model_name.lower()
        try:
            # Import here to avoid circular imports
            import streamlit as st
            
            # Check for Cloud Providers first with user-provided API keys
            if "gpt" in name_lower:
                # Use user-provided API key from session state or config
                api_key = st.session_state.get('openai_api_key', Config.OPENAI_API_KEY)
                if not api_key:
                    st.error("🔑 OpenAI API key required for GPT models. Please set it in the sidebar.")
                    return None
                return ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
            elif "claude" in name_lower:
                # Use user-provided API key from session state or config
                api_key = st.session_state.get('anthropic_api_key', Config.ANTHROPIC_API_KEY)
                if not api_key:
                    st.error("🔑 Anthropic API key required for Claude models. Please set it in the sidebar.")
                    return None
                return ChatAnthropic(model=model_name, api_key=api_key, temperature=0)
            elif "gemini" in name_lower:
                # Use user-provided API key from session state or config
                api_key = st.session_state.get('gemini_api_key', Config.GEMINI_API_KEY)
                if not api_key:
                    st.error("🔑 Google Gemini API key required for Gemini models. Please set it in the sidebar.")
                    return None
                return ChatGoogleGenerativeAI(model=model_name, api_key=api_key, temperature=0)
            
            # DEFAULT/LOCAL: Use ChatOllama for everything else (like llama3, mistral, phi)
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
    def generate_adversarial_query(pico: PICOCriteria, model_name: str) -> str:
        """Generates an adversarial search query to find contrasting evidence."""
        model = AIService.get_model(model_name)
        
        prompt = f"""
        You are an expert researcher specializing in finding contrasting evidence. Generate a search query that will find studies with OPPOSITE or CONTRADICTORY findings to the research question.
        
        ORIGINAL PICO:
        - Population: {pico.population}
        - Intervention: {pico.intervention}
        - Comparator: {pico.comparator}
        - Outcome: {pico.outcome}
        
        ADVERSARIAL SEARCH STRATEGY:
        1. Look for studies showing NO EFFECT, HARM, or NEGATIVE OUTCOMES
        2. Include terms like: "ineffective", "harmful", "no benefit", "adverse", "failure", "negative", "contradict"
        3. Consider alternative interventions that might be SUPERIOR
        4. Include studies with different methodologies that might challenge the findings
        5. Use terms that indicate treatment failure or complications
        
        CRITICAL REQUIREMENTS:
        1. Return ONLY the search string - no explanations
        2. Use proper Boolean operators (AND, OR, NOT)
        3. Include terms that will find contrasting evidence
        4. Keep it focused but broad enough to find opposing viewpoints
        
        EXAMPLE: If original is "metformin AND diabetes", adversarial might be "metformin AND (ineffective OR harmful OR adverse) AND diabetes"
        
        Generate an adversarial search query for this PICO:
        """
        
        try:
            messages = [HumanMessage(content=prompt)]
            response = model.invoke(messages)
            raw_query = response.content.strip()
            
            # Clean up the response
            clean_query = raw_query.replace("```sql", "").replace("```", "").replace("```", "")
            
            # Extract the actual query
            if '(' in clean_query:
                start_idx = clean_query.find('(')
                query_part = clean_query[start_idx:]
                if not query_part.endswith(')') and query_part.count('(') > query_part.count(')'):
                    query_part += ')'
                return query_part.strip()
            
            return clean_query.strip()
            
        except Exception as e:
            print(f"Adversarial query generation error: {e}")
            
            # Fallback: Create a basic adversarial query
            pop_terms = pico.population.split()[:2] if pico.population else ["adults"]
            int_terms = pico.intervention.split()[:2] if pico.intervention else ["treatment"]
            
            pop_query = " OR ".join([f'"{term}"' for term in pop_terms])
            int_query = " OR ".join([f'"{term}"' for term in int_terms])
            adverse_terms = "ineffective OR harmful OR adverse OR negative OR failure"
            
            return f"({pop_query}) AND ({int_query}) AND ({adverse_terms})"

    @staticmethod
    def agentic_optimize_per_source(
        current_query: str,
        pico: PICOCriteria,
        model_name: str,
        active_sources: List[str],
        research_goal: str = "",
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Agentic optimization using systematic review best practices:
        1. PICO/SPIDER framework structuring
        2. Boolean logic (OR for synonyms, AND for concepts)
        3. Precision syntax (phrases, truncation, wildcards)
        4. Database-specific subject headings (MeSH, Emtree)
        5. Iterative testing with title relevance analysis
        
        Returns detailed trace per database with quality metrics.
        """
        from data_services import DataAggregator
        
        model = AIService.get_model(model_name)
        if not model:
            return {
                "final_query": current_query,
                "trace": [],
                "per_source_queries": {source: current_query for source in active_sources},
                "error": "Model initialization failed"
            }
        
        # Database-specific syntax strategies
        db_syntax_guide = {
            "PubMed": {
                "mesh_tags": True,
                "field_tags": ["[Mesh]", "[tiab]", "[pt]", "[mh]"],
                "boolean": "AND/OR/NOT",
                "truncation": "*",
                "phrases": "\"exact phrase\"",
                "explode": True,
                "proximity": False,
                "syntax_example": '("Diabetes Mellitus"[Mesh] OR diabetes[tiab]) AND ("Metformin"[Mesh] OR metformin[tiab])'
            },
            "Europe PMC": {
                "mesh_tags": True,
                "field_tags": [":tiab", ":mesh", ":pt"],
                "boolean": "AND/OR/NOT",
                "truncation": "*",
                "phrases": "\"exact phrase\"",
                "explode": False,
                "proximity": False,
                "syntax_example": '(diabetes:tiab OR diabetes:mesh) AND (metformin:tiab OR metformin:mesh)'
            },
            "arXiv": {
                "mesh_tags": False,
                "field_tags": [],
                "boolean": "AND/OR/NOT",
                "truncation": "*",
                "phrases": "\"exact phrase\"",
                "explode": False,
                "proximity": False,
                "syntax_example": '(diabetes OR "blood sugar") AND (metformin OR "glucose lowering")'
            },
            "Semantic Scholar": {
                "mesh_tags": False,
                "field_tags": [],
                "boolean": "AND/OR",
                "truncation": "*",
                "phrases": "\"exact phrase\"",
                "explode": False,
                "proximity": False,
                "syntax_example": '"diabetes mellitus" AND metformin AND treatment'
            },
            "Google Scholar": {
                "mesh_tags": False,
                "field_tags": [],
                "boolean": "AND/OR",
                "truncation": "*",
                "phrases": "\"exact phrase\"",
                "explode": False,
                "proximity": False,
                "syntax_example": '"diabetes mellitus" treatment metformin'
            }
        }
        
        max_iterations = 10
        trace = []
        best_queries = {source: current_query for source in active_sources}
        best_scores = {source: (0, 0) for source in active_sources}
        
        for iteration in range(max_iterations):
            iter_data = {
                "iteration": iteration + 1,
                "sources": {},
                "total_papers": 0,
                "avg_relevance": 0,
                "status": "checking"
            }
            
            # Process each database independently
            for source in active_sources:
                source_result = {
                    "query": best_queries.get(source, current_query),
                    "count": 0,
                    "titles": [],
                    "relevance_score": 0,
                    "quality_rating": "Poor",
                    "optimization_applied": []
                }
                
                try:
                    # Debug: Print what's happening
                    print(f"\n=== ITERATION {iteration + 1} for {source} ===")
                    print(f"Current query: {source_result['query'][:80]}...")
                    
                    try:
                        # Test current query
                        print(f"Fetching papers with current query...")
                        papers, _ = DataAggregator.fetch_all(
                            source_result["query"],
                            [source],
                            max_per_source=10,
                            limit=10
                        )
                        print(f"Fetched {len(papers) if papers else 0} papers")
                        
                        count = len(papers) if papers else 0
                        titles = [p.title for p in papers] if papers else []
                        
                        # Calculate relevance score
                        relevance = AIService._analyze_title_relevance(
                            titles, research_goal, pico
                        ) if titles else 0
                        
                        source_result.update({
                            "count": count,
                            "titles": titles[:10],
                            "relevance_score": round(relevance, 2),
                            "quality_rating": AIService._score_to_rating(relevance),
                            "iteration_reasoning": ""
                        })
                    except Exception as e:
                        print(f"Error fetching papers: {e}")
                        import traceback
                        traceback.print_exc()
                        count = 0
                        titles = []
                        relevance = 0
                    # Determine what optimization strategy to apply
                    reasoning_parts = []
                    if iteration == 0:
                        reasoning_parts.append("Initial query setup using PICO framework")
                        if count < 5:
                            reasoning_parts.append(f"Low initial yield ({count} papers) - need to broaden with OR synonyms")
                        elif relevance < 0.4:
                            reasoning_parts.append(f"Low relevance ({relevance:.2f}) - need to refine precision")
                    elif count < 20:
                        reasoning_parts.append(f"Yield too low ({count} papers) - broadening with OR synonyms and removing filters")
                    elif relevance < 0.5:
                        reasoning_parts.append(f"Relevance too low ({relevance:.2f}) - adding more specific PICO terms")
                    elif relevance < 0.6:
                        reasoning_parts.append(f"Moderate relevance ({relevance:.2f}) - fine-tuning balance between sensitivity and precision")
                    else:
                        reasoning_parts.append(f"Good metrics achieved - query converged")
                    
                    source_result["iteration_reasoning"] = "; ".join(reasoning_parts)
                    
                    # If first iteration or needs improvement, optimize the query
                    if iteration == 0 or (count < 20 and relevance < 0.6):
                        syntax = db_syntax_guide.get(source, db_syntax_guide["PubMed"])
                        
                        optimization_prompt = f"""
                        You are an expert Medical Librarian optimizing search strings.
                        
                        RESEARCH GOAL: {research_goal}
                        
                        PICO FRAMEWORK:
                        - Population (P): {pico.population}
                        - Intervention (I): {pico.intervention}
                        - Comparator (C): {pico.comparator}
                        - Outcome (O): {pico.outcome}
                        
                        DATABASE: {source}
                        
                        DATABASE SYNTAX RULES:
                        - Supports MeSH/Subject Headings: {syntax['mesh_tags']}
                        - Field tags available: {syntax['field_tags']}
                        - Boolean operators: {syntax['boolean']}
                        - Truncation symbol: {syntax['truncation']}
                        - Phrase searching: {syntax['phrases']}
                        - Example format: {syntax['syntax_example']}
                        
                        CURRENT QUERY: {source_result["query"]}
                        
                        CURRENT RESULTS:
                        - Papers found: {count}
                        - Relevance score: {relevance:.2f}/1.0
                        - Sample titles: {titles[:3] if titles else "N/A"}
                        
                        OPTIMIZATION STRATEGIES TO APPLY:
                        
                        1. PICO STRUCTURING (Primary Strategy):
                           - Build query around Population AND Intervention (skip Outcome - too narrow)
                           - Group synonyms with OR within each PICO element
                           - Join major concepts with AND
                        
                        2. BOOLEAN LOGIC:
                           - OR: Use between synonyms (e.g., "heart attack" OR "myocardial infarction")
                           - AND: Join PICO categories only (e.g., [Population] AND [Intervention])
                           - Avoid NOT: Too risky for systematic reviews
                        
                        3. PRECISION SYNTAX:
                           - Phrase quotes: Use \"\" for exact phrases (\"type 2 diabetes\")
                           - Truncation (*): therap* → therapy, therapies, therapeutic
                           - Wildcards: Use ? or $ for spelling variants if supported
                        
                        4. SUBJECT HEADINGS:
                           {f"- Include MeSH terms with {syntax['field_tags'][0]} if applicable" if syntax['mesh_tags'] else "- Use keyword terms only (no controlled vocabulary)"}
                           {f"- Explode broader MeSH terms to capture sub-types" if syntax.get('explode') else ""}
                        
                        5. TESTING ITERATION {iteration + 1}/10:
                           - Current relevance: {relevance:.2f}
                           - Current papers found: {count}
                           - Target: >0.7 relevance with 20+ papers
                           - If too few results (<20): Add OR synonyms, remove filters, broaden MeSH terms
                           - If low relevance (<0.6): Narrow with more specific terms, add required concepts
                        
                        CRITICAL RULES:
                        1. You MUST return a DIFFERENT query than CURRENT QUERY
                        2. If count < 20: Add MORE synonyms with OR, use broader MeSH terms
                        3. If relevance < 0.6: Add specific terms to improve precision
                        4. NEVER return the exact same query - always modify something
                        
                        RETURN ONLY the optimized query string for {source}.
                        NO explanations, NO markdown, just the executable query.
                        MUST be different from current query."""
                        
                        try:
                            response = model.invoke([HumanMessage(content=optimization_prompt)])
                            new_query = response.content.strip()
                            print(f"AI returned query: {new_query[:80]}...")
                            
                            new_query = re.sub(r'^(Query|Optimized Query|Search String):\s*', '', new_query, flags=re.IGNORECASE)
                            new_query = new_query.replace('```sql', '').replace('```', '').replace('`', '').strip()
                            new_query = re.sub(r'^(PubMed|pubmed)\s*[:\-]?\s*', '', new_query, flags=re.IGNORECASE)
                            new_query = re.sub(r'^(Here is|This is).*$', '', new_query, flags=re.IGNORECASE | re.MULTILINE)
                            new_query = re.sub(r'^(The optimized|Final|Best).*$', '', new_query, flags=re.IGNORECASE | re.MULTILINE)
                            new_query = new_query.strip()
                            
                            print(f"After cleaning: {new_query[:80]}...")
                            print(f"Comparing to current: {source_result['query'][:80]}...")
                            
                            if new_query and len(new_query) > 10:
                                if new_query != source_result["query"]:
                                    print(f"Query changed! Updating...")
                                    source_result["query"] = new_query
                                    source_result["optimization_applied"].append(f"iteration_{iteration + 1}")
                                else:
                                    print(f"Query unchanged - AI returned same query")
                            else:
                                print(f"New query rejected: too short or empty")
                        except Exception as e:
                            print(f"Optimization error for {source}: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    iter_data["sources"][source] = source_result
                    
                    # Track best query by relevance (papers as tiebreaker)
                    current_relevance = source_result["relevance_score"]
                    current_count = source_result["count"]
                    current_query = source_result["query"]
                    
                    if source not in best_queries:
                        # First iteration - save it
                        best_queries[source] = current_query
                        best_scores[source] = (current_relevance, current_count)
                    else:
                        # Compare: higher relevance wins, if tie then more papers wins
                        prev_relevance, prev_count = best_scores[source]
                        if current_relevance > prev_relevance:
                            best_queries[source] = current_query
                            best_scores[source] = (current_relevance, current_count)
                        elif current_relevance == prev_relevance and current_count > prev_count:
                            best_queries[source] = current_query
                            best_scores[source] = (current_relevance, current_count)
                    
                    # Call progress callback if provided
                    if progress_callback:
                        try:
                            progress_callback(
                                iteration=iteration + 1,
                                total=max_iterations,
                                source=source,
                                count=count,
                                relevance=relevance,
                                reasoning=source_result.get("iteration_reasoning", "")
                            )
                        except Exception as e:
                            print(f"Progress callback error: {e}")
                        
                except Exception as e:
                    print(f"\n!!! EXCEPTION in iteration {iteration + 1} for {source}: {e}")
                    import traceback
                    traceback.print_exc()
                    iter_data["sources"][source] = {
                        "query": best_queries.get(source, current_query),
                        "count": 0,
                        "titles": [],
                        "relevance_score": 0,
                        "quality_rating": "Error",
                        "error": str(e)
                    }
            
            # Calculate aggregate metrics
            all_scores = [s["relevance_score"] for s in iter_data["sources"].values()]
            all_counts = [s["count"] for s in iter_data["sources"].values()]
            
            iter_data["avg_relevance"] = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0
            iter_data["total_papers"] = sum(all_counts)
            
            # Check stopping conditions (after minimum 3 iterations)
            if iteration >= 2:
                if iter_data["avg_relevance"] >= 0.7 and iter_data["total_papers"] >= 20:
                    iter_data["status"] = "success"
                    trace.append(iter_data)
                    break
                if iteration >= max_iterations - 1:
                    iter_data["status"] = "max_iterations"
                    trace.append(iter_data)
                    break
            
            trace.append(iter_data)
        
        return {
            "final_query": best_queries.get(active_sources[0], current_query) if active_sources else current_query,
            "per_source_queries": best_queries,
            "trace": trace,
            "iterations_run": len(trace),
            "best_relevance": max([t["avg_relevance"] for t in trace]) if trace else 0,
            "total_papers_found": trace[-1]["total_papers"] if trace else 0
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
            
        try:
            # 1. Clean up potential markdown formatting
            clean_text = text.replace("```json", "").replace("```", "").strip()
            
            # 2. Find the boundaries of the JSON object or list
            # We look for the first occurrence of { or [ and the last } or ]
            start_brace = clean_text.find('{')
            start_bracket = clean_text.find('[')
            
            # Determine which starts first
            if start_brace == -1 and start_bracket == -1:
                # No JSON structure found, try raw load as last resort
                return json.loads(clean_text)
                
            start_idx = start_brace if (start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket)) else start_bracket
            
            end_brace = clean_text.rfind('}')
            end_bracket = clean_text.rfind(']')
            end_idx = max(end_brace, end_bracket)

            if start_idx != -1 and end_idx != -1:
                json_str = clean_text[start_idx:end_idx + 1]
                return json.loads(json_str)
            
            return json.loads(clean_text)
        except Exception as e:
            # Log the error to the terminal so you can see why it failed
            print(f"❌ JSON Parsing Error: {e} | Raw Text: {text[:100]}...")
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

        inc_text = ", ".join(inclusion_criteria) if inclusion_criteria else "None specified"
        excl_text = ", ".join(exclusion_criteria) if exclusion_criteria else "None specified"

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
        Only use "INCLUDE" or "EXCLUDE" values - no "ERROR" or other values.
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
                    response = requests.get(url, params={"pageSize": max_results, "format": "json"}, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        cites = data.get('citationList', {}).get('citation', [])
                        for cite in cites:
                            citations.append({
                                'id': cite.get('id', ''),
                                'title': cite.get('title', cite.get('source', 'Unknown')),
                                'abstract': cite.get('abstractText', ''),
                                'url': f"https://pubmed.ncbi.nlm.nih.gov/{cite.get('id', '')}" if cite.get('id') else '',
                                'source': 'Europe PMC (Cited by)',
                                'citation_type': 'forward'
                            })
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
                                cites_response = requests.get(cites_url, params=cites_params, timeout=10)
                                if cites_response.status_code == 200:
                                    cites_data = cites_response.json()
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
                        if cited_by_url:
                            try:
                                cites_response = requests.get(cited_by_url, params={"per_page": max_results}, timeout=10)
                                if cites_response.status_code == 200:
                                    cites_data = cites_response.json()
                                    for cite in cites_data.get('results', []):
                                        citations.append({
                                            'id': cite.get('doi', cite.get('id', '').split('/')[-1]),
                                            'title': cite.get('display_name', 'Unknown'),
                                            'abstract': cite.get('abstract', ''),
                                            'url': cite.get('open_access', {}).get('oa_url', f"https://doi.org/{cite.get('doi')}" if cite.get('doi') else ''),
                                            'source': 'OpenAlex (Cited by)',
                                            'citation_type': 'forward'
                                        })
                            except:
                                pass
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

INSTRUCTIONS - BE PERMISSIVE AND BROAD IN YOUR EVALUATION:
1. Focus ONLY on your assigned PICO element - ignore all other criteria
2. Papers should broadly align with your criterion, NOT match exactly
3. Look for related terms, synonyms, and conceptual similarity
4. If the abstract mentions anything related to your criterion, vote PASS
5. Only vote FAIL if there's clear evidence the paper does NOT relate to your criterion
6. Extract the exact quote from the abstract that justifies your decision
7. Rate your confidence (0.0-1.0) - use 0.5+ for broad matches, lower for uncertain

PAPERS TO EVALUATE ({len(papers)} total):
{papers_text}

OUTPUT FORMAT - Return a JSON array with one object per paper:
[
    {{
        "paper_id": "paper_id_here",
        "met": true or false,
        "confidence": 0.0-1.0,
        "evidence": "Exact quote from abstract supporting your decision",
        "reasoning": "Brief explanation of your decision"
    }},
    ... (for ALL papers)
]

IMPORTANT: Be BROAD and INCLUSIVE. When in doubt, PASS the paper.

JSON ARRAY:"""
        
        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            return self._parse_response(response.content, papers)
        except Exception as e:
            print(f"Agent {self.name} error: {e}")
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
        """Create error vote for failed evaluation."""
        return AgentVote(
            agent_name=self.name,
            agent_type=self.agent_type,
            criterion=self.criterion,
            paper_id=paper.id,
            met=False,
            confidence=0.0,
            evidence="Error in evaluation",
            reasoning=f"Agent {self.name} failed to evaluate"
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
        """Decompose criteria into specialist PICO agents only."""
        # 4 PICO agents (always create if criteria exist)
        if self.pico.population:
            self.agents.append(CriterionAgent(
                "Population Agent", "PICO_P", self.pico.population, self.model_name, self.api_keys
            ))
        if self.pico.intervention:
            self.agents.append(CriterionAgent(
                "Intervention Agent", "PICO_I", self.pico.intervention, self.model_name, self.api_keys
            ))
        if self.pico.comparator:
            self.agents.append(CriterionAgent(
                "Comparator Agent", "PICO_C", self.pico.comparator, self.model_name, self.api_keys
            ))
        if self.pico.outcome:
            self.agents.append(CriterionAgent(
                "Outcome Agent", "PICO_O", self.pico.outcome, self.model_name, self.api_keys
            ))
    
    def screen_papers(self, papers: List[Paper], progress_callback=None) -> List[Dict[str, Any]]:
        """
        ORCHESTRATOR: Execute full multi-agent screening pipeline.
        
        Returns list of paper results with per-agent votes and final decision.
        """
        if not papers or not self.agents:
            return []
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        total_agents = len(self.agents)
        all_votes: List[AgentVote] = []
        
        # FAN-OUT: Run all agents in parallel, each evaluates ALL papers
        with ThreadPoolExecutor(max_workers=min(10, total_agents)) as executor:
            future_to_agent = {
                executor.submit(agent.evaluate_all_papers, papers): agent 
                for agent in self.agents
            }
            
            completed_agents = 0
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    votes = future.result()  # List of AgentVote for all papers
                    all_votes.extend(votes)
                except Exception as exc:
                    print(f"Agent {agent.name} failed: {exc}")
                
                completed_agents += 1
                if progress_callback:
                    progress_callback(completed_agents, total_agents, 
                                    f"Agent {completed_agents}/{total_agents}: {agent.name}")
        
        # AGGREGATE: Apply consensus logic per paper
        return self._aggregate_results(papers, all_votes)
    
    def _aggregate_results(self, papers: List[Paper], all_votes: List[AgentVote]) -> List[Dict[str, Any]]:
        """
        AGGREGATOR: PICO-only consensus engine with traceability.
        
        Logic:
        - PICO agents: Logical AND (must satisfy all P, I, C, O)
        - All 4 PICO elements must pass for inclusion
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
            
            # All votes are PICO votes now
            pico_votes = [v for v in paper_votes if v.agent_type.startswith('PICO')]
            
            # Apply consensus logic: ALL PICO must pass
            pico_pass = all(v.met for v in pico_votes) if pico_votes else True
            
            # Final decision with detailed reason
            if not pico_pass:
                decision = "EXCLUDE"
                failed_votes = [v for v in pico_votes if not v.met]
                passed_votes = [v for v in pico_votes if v.met]
                
                # Build a comprehensive summary of why the paper was excluded
                parts = []
                
                # List which PICO elements passed (if any)
                if passed_votes:
                    passed_elements = [v.agent_name.replace(" Agent", "") for v in passed_votes]
                    parts.append(f"This paper aligns with the {', '.join(passed_elements)} criteria")
                
                # List which PICO elements failed with full reasoning
                fail_details = []
                for v in failed_votes:
                    # Get the full reasoning without truncation
                    reason = v.reasoning if v.reasoning and v.reasoning != "No reasoning provided" else v.evidence
                    if not reason or reason == "No specific evidence":
                        reason = f"does not address the required criterion: {v.criterion}"
                    fail_details.append(f"the {v.agent_name.replace(' Agent', '').lower()} criterion - {reason}")
                
                if passed_votes:
                    parts.append(f"but fails to meet {', '.join(fail_details)}")
                else:
                    parts.append(f"This paper fails to meet {', '.join(fail_details)}")
                
                decision_reason = " ".join(parts) + "."
            else:
                decision = "INCLUDE"
                # Build a comprehensive summary for included papers
                if pico_votes:
                    pico_elements = [v.agent_name.replace(" Agent", "").lower() for v in pico_votes if v.met]
                    evidence_parts = []
                    for v in pico_votes:
                        if v.evidence and v.evidence != "No specific evidence":
                            evidence_parts.append(f"{v.agent_name.replace(' Agent', '')}: {v.evidence}")
                    
                    if evidence_parts:
                        decision_reason = f"This paper satisfies all PICO criteria. It demonstrates alignment with {', '.join(pico_elements)} elements. Key evidence: {'; '.join(evidence_parts)}."
                    else:
                        decision_reason = f"This paper satisfies all PICO criteria with alignment across {', '.join(pico_elements)} elements."
                else:
                    decision_reason = "This paper satisfies all PICO criteria based on available evidence."
            
            # Build traceability record
            agent_trace = {}
            for vote in paper_votes:
                agent_trace[vote.agent_name] = {
                    "type": vote.agent_type,
                    "criterion": vote.criterion,
                    "vote": "PASS" if vote.met else "FAIL",
                    "confidence": vote.confidence,
                    "evidence": vote.evidence,
                    "reasoning": vote.reasoning
                }
            
            results.append({
                "paper_id": paper.id,
                "Source": paper.source,
                "Title": paper.title,
                "URL": paper.url,
                "Abstract": paper.abstract,
                "Decision": decision,
                "Reason": decision_reason,
                "PICO_Score": f"{sum(v.met for v in pico_votes)}/{len(pico_votes)}",
                "Agent_Count": len(paper_votes),
                "Agent_Trace": agent_trace  # Full traceability for UI
            })
        
        return results
    
    def get_agent_summary(self) -> Dict[str, int]:
        """Return summary of specialist PICO agents created."""
        return {
            "pico_agents": sum(1 for a in self.agents if a.agent_type.startswith('PICO')),
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
        
        total_agents = len(self.agents)
        all_votes: List[FullTextAgentVote] = []
        
        # FAN-OUT: Run all agents in parallel on the single paper
        with ThreadPoolExecutor(max_workers=min(10, total_agents)) as executor:
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
    
    def get_agent_summary(self) -> Dict[str, int]:
        """Return summary of specialist agents created."""
        return {
            "inclusion_agents": sum(1 for a in self.agents if a.agent_type == "INCLUSION"),
            "exclusion_agents": sum(1 for a in self.agents if a.agent_type == "EXCLUSION"),
            "total_agents": len(self.agents)
        }