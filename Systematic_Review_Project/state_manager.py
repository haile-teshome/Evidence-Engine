# ============================================================================
# FILE: state_manager.py
# Session state management
# ============================================================================

import streamlit as st
from config import Config
from models import PICOCriteria


class SessionState:
    """Manages Streamlit session state."""
    
    @staticmethod
    def initialize():
        """Initialize session state with default values."""
        defaults = {
            'pico': PICOCriteria(),
            'query': "",
            'papers': [],
            'results': None,
            'goal': "",
            'custom_model': Config.DEFAULT_MODEL
        }
        
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default
    
    @staticmethod
    def reset():
        """Reset session state to defaults."""
        st.session_state.pico = PICOCriteria()
        st.session_state.query = ""
        st.session_state.papers = []
        st.session_state.results = None
        st.session_state.goal = ""


