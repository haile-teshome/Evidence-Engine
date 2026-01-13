# ============================================================================
# FILE: ui_components.py
# Reusable UI components
# ============================================================================

import streamlit as st
import pandas as pd
from config import DataSource
from state_manager import SessionState


class UIComponents:
    """Reusable UI components."""
    
    @staticmethod
    def render_sidebar() -> tuple:
        """Render sidebar controls."""
        with st.sidebar:
            st.title("⚙️ Refinement Center")
            
            # Model selection
            st.subheader("1. AI Model")
            model_choice = st.selectbox(
                "Preset Models",
                ["llama3", "mistral", "phi3", "Custom"],
                index=0
            )
            
            if model_choice == "Custom":
                model_name = st.text_input(
                    "Model Name",
                    value=st.session_state.custom_model
                )
            else:
                model_name = model_choice
            
            # Data sources
            st.subheader("2. Data Sources")
            active_sources = st.multiselect(
                "Active Sources",
                [s.value for s in DataSource],
                default=[
                    DataSource.PUBMED.value,
                    DataSource.BIG3_JOURNALS.value,
                    DataSource.BIORXIV.value
                ]
            )
            
            uploaded_files = None
            if DataSource.LOCAL_PDF.value in active_sources:
                uploaded_files = st.file_uploader(
                    "Upload PDFs",
                    accept_multiple_files=True,
                    type=['pdf']
                )
            
            # PICO override
            st.subheader("3. PICO Override")
            pico = st.session_state.pico
            
            pico.population = st.text_input(
                "Population",
                value=pico.population
            )
            pico.intervention = st.text_input(
                "Intervention",
                value=pico.intervention
            )
            pico.outcome = st.text_input(
                "Outcome",
                value=pico.outcome
            )
            
            # Settings
            num_per_source = st.slider(
                "Max papers per source",
                1, 50, 10
            )
            
            # Reset button
            if st.button("🔄 Reset App State", use_container_width=True):
                SessionState.reset()
                st.rerun()
        
        return (model_name, active_sources, uploaded_files, num_per_source)
    
    @staticmethod
    def render_results(results_df: pd.DataFrame):
        """Render screening results."""
        if results_df is None or results_df.empty:
            return
        
        st.divider()
        st.subheader("📊 Screening Results")
        
        # Display key columns
        display_cols = [
            'Source', 'decision', 'design', 'sample_size',
            'reason', 'risk_of_bias', 'Title'
        ]
        available_cols = [c for c in display_cols if c in results_df.columns]
        
        st.dataframe(
            results_df[available_cols],
            use_container_width=True
        )
        
        # Export button
        csv = results_df.to_csv(index=False)
        st.download_button(
            "📥 Export Results (CSV)",
            csv,
            "epi_screening_results.csv",
            "text/csv"
        )

