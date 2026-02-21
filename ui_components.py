import streamlit as st
import pandas as pd
from config import DataSource
from state_manager import SessionState
import graphviz
import io
import os
import textwrap
from docx import Document
from models import PICOCriteria  
from state_manager import SessionState

class UIComponents:
    """Reusable UI components."""
    
    @staticmethod
    def render_sidebar() -> tuple:

        with st.sidebar:
            st.markdown('<div style="margin-top: -20px;"></div>', unsafe_allow_html=True)
            st.markdown("### 🧬 Evidence Engine")
            
            # 1. START NEW INVESTIGATION
            if st.button("Start New Investigation", use_container_width=True, type="primary"):
                # Clear all cached data (API results, dataframes)
                st.cache_data.clear()
                
                # Clear all session state variables
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                
                # Re-initialize the default state
                SessionState.reset() 
                
                # Force a complete app restart
                st.rerun()
            st.divider()

            # 2. PAGE NAVIGATION (Anthropic-style sidebar nav)
            st.markdown("##### Navigation")
            
            # Initialize page if not set
            if 'current_page' not in st.session_state:
                st.session_state.current_page = "home"
            
            # Navigation options with Material icons
            nav_items = [
                ("home", ":material/home:", "Home"),
                ("simulation", ":material/analytics:", "Simulation"),
                ("abstract", ":material/article:", "Abstract Screening"),
                ("fulltext", ":material/lab_research:", "Full-Text Evidence"),
                ("prisma", ":material/account_tree:", "PRISMA Flow")
            ]
            
            # Render navigation buttons
            for nav_id, icon, label in nav_items:
                is_active = st.session_state.current_page == nav_id
                
                # Use different styling for active vs inactive
                if is_active:
                    btn_type = "primary"
                else:
                    btn_type = "secondary"
                
                if st.button(
                    f"{icon} {label}",
                    key=f"nav_{nav_id}",
                    use_container_width=True,
                    type=btn_type
                ):
                    st.session_state.current_page = nav_id
                    st.rerun()
            
            st.divider()

            # 3. SETTINGS
            with st.container():
                model_choice = st.selectbox("AI Model", ["llama3", "mistral", "phi", "Custom"])
                
                # Show custom model input if "Custom" is selected
                if model_choice == "Custom":
                    custom_model = st.text_input("Enter Custom Model Name", placeholder="e.g., gpt-4, claude-3, llama3:70b")
                    if custom_model:
                        model_choice = custom_model
                
                # Data Sources Selection
                active_sources = st.multiselect("Sources", [s.value for s in DataSource], default=["PubMed"])
                
                # PDF Upload Section (only shows when Local PDFs is selected)
                local_pdfs_selected = DataSource.LOCAL_PDF.value in active_sources
                
                if local_pdfs_selected:
                    uploaded_files = st.file_uploader(
                        "Upload PDF documents",
                        type="pdf",
                        accept_multiple_files=True,
                        help="Upload research papers in PDF format for analysis"
                    )
                    
                    # Store uploaded files in session state
                    if uploaded_files:
                        st.session_state.uploaded_files = uploaded_files
                    
                    # File selection for uploaded PDFs
                    uploaded_files = st.session_state.get('uploaded_files', [])
                    
                    if uploaded_files:
                        st.markdown("### 📄 Select PDFs for Search")
                        selected_files = st.multiselect(
                            "Choose specific PDFs to include in search",
                            options=[file.name for file in uploaded_files],
                            default=[file.name for file in uploaded_files],
                            help="Select which uploaded PDFs to include in search. Leave empty to include all."
                        )
                    else:
                        selected_files = None
                else:
                    uploaded_files = None
                    selected_files = None
                
                # Depth slider
                num_per_source = st.slider("Depth", 5, 100, 20)    

        return (model_choice, active_sources, selected_files, num_per_source)
        

    @staticmethod
    def render_results(df: pd.DataFrame):
        if df.empty:
            st.info("No papers match the current criteria.")
            return

        # Style DataFrame to color cells based on decision and criteria
        def color_decisions(val):
            if 'Include' in str(val):
                return 'background-color: #d4edda; color: #155724; font-weight: 500; padding: 8px;'
            elif 'Exclude' in str(val):
                return 'background-color: #f8d7da; color: #721c24; font-weight: 500; padding: 8px;'
            else:
                return 'padding: 8px;'
        
        def color_criteria(val):
            if 'INCLUDE' in str(val):
                return 'background-color: #d4edda; color: #155724; font-weight: 500; padding: 8px;'
            elif 'EXCLUDE' in str(val):
                return 'background-color: #f8d7da; color: #721c24; font-weight: 500; padding: 8px;'
            else:
                return 'background-color: #fff3cd; color: #856404; font-weight: 500; padding: 8px;'

        # Apply styling to Decision column if it exists
        styled_df = df.copy()
        if 'Decision' in styled_df.columns:
            styled_df = styled_df.style.map(color_decisions, subset=['Decision'])
        
        # Apply styling to criteria columns
        try:
            import streamlit as st
            inclusion_criteria = st.session_state.get('inclusion_list', [])
            exclusion_criteria = st.session_state.get('exclusion_list', [])
            
            for criterion in inclusion_criteria + exclusion_criteria:
                if criterion in styled_df.columns:
                    styled_df = styled_df.map(color_criteria, subset=[criterion])
        except:
            pass

        # This configuration turns the "URL" column into a clickable link
        st.dataframe(
            styled_df,
            column_config={
                "URL": st.column_config.LinkColumn(
                    "Source Link",    
                    display_text="View Paper", 
                    width="small"
                ),
                "Score": st.column_config.NumberColumn(format="%d ⭐"),
                "Title": st.column_config.TextColumn(width="large")
            },
            hide_index=True,
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
        """Renders PRISMA 2020: Gold header centered with bucketed exclusion reasons."""
        
        counts = st.session_state.prisma_counts
        
        # LOGIC FOR STAGE 1 (ABSTRACTS) 
        screened_n = counts.get('screened', 0)
        abs_excl_n = counts.get('excluded_total', 0)
        inc_n = screened_n - abs_excl_n 
        
        # LOGIC FOR STAGE 2 (FULL-TEXT)
        final_n = counts.get('included_final', inc_n)
        ft_excluded_total = inc_n - final_n

        # SOURCE BREAKDOWN LOGIC 
        source_data = counts.get('source_counts', {})
        total_raw = counts.get('identified', 0)
        if source_data:
            source_lines = [f"{name} (n={n})" for name, n in source_data.items()]
            source_text = "\\n".join(source_lines)
            identification_label = f"Records identified from databases (n = {total_raw})\\n{source_text}"
        else:
            identification_label = f"Records identified from:\\nDatabases (n = {counts['identified']})\\nRegisters (n = 0)"


        reasons_dict = counts.get('ft_exclusion_breakdown', {})
        if reasons_dict and ft_excluded_total > 0:
            sorted_reasons = sorted(reasons_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            reasons_list = [f"• {r}: (n={c})" for r, c in sorted_reasons]
            reasons_text = "\\n".join(reasons_list)
        else:
            reasons_text = ""

        dot = graphviz.Digraph(comment='PRISMA 2020')
        dot.attr(rankdir='TB', nodesep='0.5', ranksep='0.4')
        
        dot.attr('node', shape='box', fontname='Arial', fontsize='10', 
                style='filled, rounded', fillcolor='#ffffff', color='#000000',
                width='4.0', height='1.0', penwidth='1.5')

        # ROW 0: THE MASTER HEADER
        dot.node('H1', 'Identification of studies via databases and registers', 
                fillcolor='#FFD700', color='#B8860B', fontname='Arial Bold', width='9.0')

        # ROW 1: IDENTIFICATION BOXES
        dot.node('N1', identification_label)
        dot.node('N2_side', f"Records removed before screening:\\n"
                            f"Duplicate records removed (n = {counts['duplicates_removed']})\\n"
                            f"Records marked as ineligible by automation tools (n = 0)\\n"
                            f"Records removed for other reasons (n = 0)")

        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node('N1')
            s.node('N2_side')

        dot.edge('H1', 'N1', style='invis')
        dot.edge('H1', 'N2_side', style='invis')

        # ROW 2: Abstract Screening
        dot.node('N3', f"Records screened\\n(n = {screened_n})")
        dot.node('N4_excl', f"Records excluded\\n(n = {abs_excl_n})")
        
        # ROW 3: Retrieval
        dot.node('N5_ret', f"Reports sought for retrieval\\n(n = {inc_n})")
        dot.node('N5_not_ret', f"Reports not retrieved\\n(n = 0)")
        
        # ROW 4: Eligibility (STAGE 2)
        dot.node('N6_elig', f"Reports assessed for eligibility\\n(n = {inc_n})")
        # Updated to show the Stage 2 specific buckets
        dot.node('N6_excl_side', f"Reports excluded (n={ft_excluded_total}):\\n{reasons_text}", width='4')
        
        # ROW 5: Final Result
        dot.node('N7_final', f"Studies included in review\\n(n = {final_n})")

        # Alignment for remaining rows 
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node('N3')
            s.node('N4_excl')
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node('N5_ret')
            s.node('N5_not_ret')
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node('N6_elig')
            s.node('N6_excl_side')

        # VISIBLE CONNECTIONS
        dot.edge('N1', 'N2_side')
        dot.edge('N1', 'N3')
        dot.edge('N3', 'N4_excl')
        dot.edge('N3', 'N5_ret')
        dot.edge('N5_ret', 'N5_not_ret')
        dot.edge('N5_ret', 'N6_elig')
        dot.edge('N6_elig', 'N6_excl_side')
        dot.edge('N6_elig', 'N7_final')

        st.graphviz_chart(dot, use_container_width=True)