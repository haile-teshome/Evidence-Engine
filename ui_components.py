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

            # # 2. RECENT SESSIONS
            # st.markdown("##### Recent Sessions")
            # history = st.session_state.get('history', [])

            # if not history:
            #     st.caption("No history yet.")
            # else:
            #     # Iterate through history (newest at top)
            #     for i in range(len(history) - 1, -1, -1):
            #         entry = history[i]
            #         label = entry.get('short_summary') or entry.get('goal', 'New Investigation')
                    
            #         # Create the clickable bullet
            #         if st.button(f"• {label}", key=f"hist_btn_{i}", use_container_width=True):
            #             # RESTORE ALL STATE
            #             st.session_state.active_session_index = i
            #             st.session_state.goal = entry.get('goal', "")
            #             st.session_state.query = entry.get('query', "")
            #             st.session_state.messages = entry.get('messages', [])
            #             st.session_state.results = entry.get('results')
                        
            #             # Map PICO
            #             p = entry.get('pico_dict', {})
            #             st.session_state.pico.population = p.get('p', '')
            #             st.session_state.pico.intervention = p.get('i', '')
            #             st.session_state.pico.comparator = p.get('c', '')
            #             st.session_state.pico.outcome = p.get('o', '')
            #             st.rerun()

            # st.divider()

            # 3. SETTINGS
            with st.container():
                model_choice = st.selectbox("AI Model", ["llama3", "mistral", "phi", "Custom"])
                active_sources = st.multiselect("Sources", [s.value for s in DataSource], default=["PubMed"])
                num_per_source = st.slider("Depth", 5, 100, 20)
        
        return (model_choice, active_sources, None, num_per_source)
        

    @staticmethod
    def render_results(df: pd.DataFrame):
        if df.empty:
            st.info("No papers match the current criteria.")
            return

        # This configuration turns the "URL" column into a clickable link
        st.dataframe(
            df,
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