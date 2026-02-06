import streamlit as st
import pandas as pd
from config import DataSource
from state_manager import SessionState
import graphviz
import io
import os
import textwrap
from docx import Document

class UIComponents:
    """Reusable UI components."""
    
    @staticmethod
    def render_sidebar() -> tuple:
        """Render sidebar controls (Cleaned: PICO removed)."""
        with st.sidebar:
            st.title("⚙️ Refinement Center")

            # --- SEARCH HISTORY SECTION ---
            st.subheader("📜 Search History")
            history = st.session_state.get('history', [])
            
            if not history:
                st.caption("No previous iterations recorded.")
            else:
                # Limit history display to save space
                for i, entry in enumerate(reversed(history[-5:])):
                    goal_text = entry.get('goal', 'Unknown Goal')
                    label = (goal_text[:35] + '...') if len(goal_text) > 35 else goal_text
                    
                    if st.button(f"↩️ {label}", key=f"hist_{i}", use_container_width=True):
                        # Restore core states
                        st.session_state.goal = entry['goal']
                        st.session_state.query = entry['query']
                        
                        # Restore PICO dataclass
                        p_dict = entry.get('pico_dict', {})
                        st.session_state.pico.population = p_dict.get('p', '')
                        st.session_state.pico.intervention = p_dict.get('i', '')
                        st.session_state.pico.comparator = p_dict.get('c', '')
                        st.session_state.pico.outcome = p_dict.get('o', '')
                        st.session_state.pico.inclusion_criteria = p_dict.get('inclusion', '')
                        st.session_state.pico.exclusion_criteria = p_dict.get('exclusion', '')
                        
                        # Restore criteria lists used by the main editor
                        st.session_state.inclusion_list = entry.get('inclusion', [])
                        st.session_state.exclusion_list = entry.get('exclusion', [])
                        st.session_state.field_feedback = entry.get('field_feedback', {})
                        st.rerun()
            
            st.divider()
            
            # --- MODEL SELECTION ---
            st.subheader("AI Model")
            model_choice = st.selectbox(
                "Preset Models",
                ["llama3", "mistral", "phi3", "Custom"],
                index=0
            )
            
            if model_choice == "Custom":
                model_name = st.text_input(
                    "Model Name",
                    value=st.session_state.get('custom_model', 'llama3')
                )
            else:
                model_name = model_choice
            
            # --- DATA SOURCES ---
            st.subheader("Data Sources")
            active_sources = st.multiselect(
                "Active Sources",
                [s.value for s in DataSource],
                default=[
                    DataSource.PUBMED.value,
                    DataSource.BIG3_JOURNALS.value,
                    DataSource.BIORXIV.value,
                    DataSource.SEMANTIC_SCHOLAR.value
                ]
            )
            
            uploaded_files = None
            if DataSource.LOCAL_PDF.value in active_sources:
                uploaded_files = st.file_uploader(
                    "Upload PDFs",
                    accept_multiple_files=True,
                    type=['pdf']
                )
            
            # --- SEARCH PARAMETERS ---
            st.subheader("Search Depth")
            num_per_source = st.slider(
                "Max papers per source",
                1, 50, 10
            )
            
            st.divider()
            
            # --- RESET ACTION ---
            if st.button("🔄 Reset All Data", use_container_width=True, type="secondary"):
                SessionState.reset()
                st.rerun()
        
        return (model_name, active_sources, uploaded_files, num_per_source)
    
    # In ui_components.py
    @staticmethod
    def render_results(results_df: pd.DataFrame):
        if results_df is None or results_df.empty:
            st.info("No results to display yet.")
            return
        
        st.subheader("📊 Screening Results")
        
        # Change these to match the keys used in app.py (Step 5 of your app.py loop)
        display_cols = [
            'Source', 'Decision', 'Design', 'Sample Size',
            'Reason', 'Title'
        ]
        available_cols = [c for c in display_cols if c in results_df.columns]
        
        st.dataframe(
            results_df[available_cols],
            use_container_width=True
        )

    @staticmethod
    def render_deduplication_report():
        """Renders a section showing which papers were removed."""
        if 'last_duplicates' in st.session_state and st.session_state['last_duplicates']:
            with st.expander("📝 View Deduplicated Papers (Removed)", expanded=False):
                st.write("The following papers were identified as duplicates and excluded:")
                
                report_data = []
                for p in st.session_state['last_duplicates']:
                    report_data.append({
                        "Source": p.source,
                        "Title": p.title,
                        "ID/DOI": p.id
                    })
                
                st.table(pd.DataFrame(report_data))
        elif 'last_duplicates' in st.session_state:
            st.info("No duplicates were found in the last search.")

    @staticmethod
    def render_prisma_flow():
        """Renders the PRISMA 2020 flow with exact matching wording, structure, and colors."""
        counts = st.session_state.get('prisma_counts', {})
        
        # Extract counts
        id_n = counts.get('identified', 0)
        dup_n = counts.get('duplicates_removed', 0)
        scr_n = counts.get('screened', 0)
        excl_n = counts.get('excluded_total', 0)
        inc_n = scr_n - excl_n
        
        # Format exclusion reasons for the side node
        reasons_map = counts.get('exclusion_breakdown', {})
        if reasons_map:
            # Join top reasons into a list for the box
            reasons_text = "\\n".join([f"• {r[:35]}: {c}" for r, c in list(reasons_map.items())[:5]])
        else:
            reasons_text = "Criteria not met"

        # DOT syntax for EXACT compliance
        prisma_dot = f"""
        digraph PRISMA {{
            # Global settings for uniform box size and font
            node [shape=box, fontname="Arial", fontsize=10, style="filled", width=3.8, height=1.2];
            rankdir=TB;
            
            # --- IDENTIFICATION (Blue) ---
            id [label="Identification\\nRecords identified from databases\\n(n = {id_n})", 
                fillcolor="#f8f9fa", color="#007bff"];
            
            # --- DEDUPLICATION (Gold) ---
            dup [label="Deduplication\\nDuplicates removed\\n(n = {dup_n})", 
                    fillcolor="#fff3cd", color="#ffc107"];
            
            # --- SCREENING (Blue) ---
            scr [label="Screening\\nRecords screened by AI\\n(n = {scr_n})", 
                    fillcolor="#f8f9fa", color="#007bff"];
            
            # --- EXCLUSION (Blue) ---
            excl [label="Exclusion Reasons\\n{reasons_text}\\n(Total n = {excl_n})", 
                    fillcolor="#f8f9fa", color="#007bff"];
            
            # --- INCLUSION (EXACT WORDING - Blue) ---
            final [label="Included\\nStudies included in review\\n(n = {inc_n})\\nReports of included studies\\n(n = {inc_n})", 
                    fillcolor="#f8f9fa", color="#007bff"];
            
            # Alignment constraints
            {{rank=same; scr; excl;}}
            
            # Flow Connections
            id -> dup;
            dup -> scr;
            scr -> final;
            scr -> excl [label="  Excluded"];
        }}
        """
        
        st.graphviz_chart(prisma_dot)