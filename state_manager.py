# ============================================================================
# FILE: state_manager.py
# Session state management
# ============================================================================

import streamlit as st
from config import Config
from models import PICOCriteria
from typing import List, Dict 
from langchain_core.messages import HumanMessage, SystemMessage
from utils import AIService

class SessionState:
    """Manages Streamlit session state including research history."""
    
    @staticmethod
    def initialize():
        """Initialize session state with default values."""
        defaults = {
            'pico': PICOCriteria(),
            'query': "",
            'research_question': "",
            'papers': [],
            'results': None,
            'goal': "",
            'custom_model': Config.DEFAULT_MODEL,
            'history': [],
            'messages': [],
            'active_session_index': None 
        }
        
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

    @staticmethod
    def generate_chat_summary(messages: List[Dict[str, str]]) -> str:
        """Generates a 3-5 word summary of the conversation."""
        if not messages:
            return "New Investigation"
        
        # Combine last few messages for context
        context = "\n".join([f"{m['role']}: {m['content'][:100]}" for m in messages[-3:]])
        from utils import AIService
        llm = AIService.get_model(st.session_state.custom_model)
        prompt = [
            SystemMessage(content="Summarize this research discussion in 4 words or less. Return ONLY the title."),
            HumanMessage(content=context)
        ]
        try:
            response = llm.invoke(prompt)
            return response.content.strip().replace('"', '')
        except:
            return "Previous Research"

    @staticmethod
    def reset():
        """Complete reset of the current investigation state."""
        reset_keys = {
            'pico': PICOCriteria(),
            'query': "",
            'papers': [],
            'results': None,
            'goal': "",
            'messages': [],
            'inclusion_list': [],
            'exclusion_list': [],
            'active_session_index': None,
            'prisma_counts': {
                'identified': 0, 'duplicates_removed': 0, 
                'screened': 0, 'excluded_total': 0, 'exclusion_breakdown': {}
            }
        }
        
        for key, value in reset_keys.items():
            st.session_state[key] = value