# ============================================================================
# FILE: utils.py
# Utility functions and helpers
# ============================================================================

import re
import json
from typing import Optional, Dict
import streamlit as st
import ollama


class QueryCleaner:
    """Utilities for cleaning and formatting search queries."""
    
    @staticmethod
    def clean_for_general_search(query: str) -> str:
        """Remove PubMed-specific syntax for general search engines."""
        clean = re.sub(r'\[.*?\]', '', query)
        clean = clean.replace('AND', ' ').replace('OR', ' ')
        clean = clean.replace('(', '').replace(')', '')
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', clean)
        return " ".join(clean.split())


class AIService:
    """Handles AI model interactions."""
    
    @staticmethod
    def get_json_response(prompt: str, model: str) -> Optional[Dict]:
        """Get structured JSON response from AI model."""
        try:
            response = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': prompt}]
            )
            content = response['message']['content']
            
            # Extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            
            if start == -1 or end == 0:
                return None
                
            return json.loads(content[start:end])
            
        except json.JSONDecodeError as e:
            st.sidebar.error(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            st.sidebar.error(f"AI model error: {e}")
            return None
    
    @staticmethod
    def infer_pico_and_query(goal: str, model: str) -> Optional[Dict]:
        """Infer PICO criteria and search query from research goal."""
        prompt = f"""
        Research Goal: {goal}
        
        Generate a structured analysis with the following JSON format:
        {{
            "p": "population description",
            "i": "intervention description",
            "c": "comparator description",
            "o": "outcome description",
            "query": "optimized search query"
        }}
        """
        return AIService.get_json_response(prompt, model)
    
    @staticmethod
    def screen_paper(paper, pico, model: str):
        """Screen a paper against PICO criteria using AI."""
        from models import ScreeningResult
        
        prompt = f"""
        Screening Criteria:
        - Population: {pico.population}
        - Intervention: {pico.intervention}
        - Outcome: {pico.outcome}
        
        Study Title: {paper.title}
        Abstract: {paper.abstract}
        
        Analyze this study and return JSON:
        {{
            "decision": "YES or NO",
            "reason": "brief explanation",
            "design": "study design type",
            "sample_size": "sample size if mentioned",
            "risk_of_bias": "LOW, MODERATE, or HIGH"
        }}
        """
        
        result = AIService.get_json_response(prompt, model)
        
        if result:
            return ScreeningResult(**result)
        return ScreeningResult()


