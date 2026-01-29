import streamlit as st
import pandas as pd
import io
from config import Config
from state_manager import SessionState
from ui_components import UIComponents
from utils import AIService, Deduplicator
from data_services import DataAggregator

def main():
    """
    Main application entry point.
    Features: Sentence-based PICO, Descriptive Feedback, Editable Criteria, and AI Extraction.
    """
    st.set_page_config(
        page_title=Config.APP_TITLE,
        layout="wide",
        page_icon=Config.PAGE_ICON
    )
    
    # 1. Initialize session state FIRST to prevent AttributeErrors
    SessionState.initialize()
    
    # Supplemental state initialization for criteria and PRISMA
    if 'inclusion_list' not in st.session_state: 
        st.session_state.inclusion_list = []
    if 'exclusion_list' not in st.session_state: 
        st.session_state.exclusion_list = []
    if 'prisma_counts' not in st.session_state:
        st.session_state.prisma_counts = {
            'identified': 0, 'duplicates_removed': 0, 
            'screened': 0, 'excluded_total': 0, 'exclusion_breakdown': {}
        }

    # Custom UI Styling
    st.markdown("""
        <style>
        .stButton > button { border-radius: 10px; }
        .pico-card { 
            background-color: #f8f9fa; 
            padding: 15px; 
            border-radius: 10px; 
            border-left: 5px solid #007bff;
            min-height: 140px;
        }
        .pico-header { font-weight: bold; color: #007bff; margin-bottom: 8px; text-transform: uppercase; font-size: 0.85rem; }
        .pico-content { font-size: 0.95rem; line-height: 1.5; color: #333; }
        .summary-box {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 25px;
            border: 1px solid #e0e0e0;
            border-top: 4px solid #007bff;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        .summary-title { color: #007bff; font-size: 1.1rem; font-weight: bold; margin-bottom: 12px; }
        </style>
    """, unsafe_allow_html=True)

    # 2. Render Sidebar
    model_name, active_sources, uploaded_files, num_per_source = UIComponents.render_sidebar()

    st.markdown("<h1 style='text-align: center;'>🧬 EvidenceEngine</h1>", unsafe_allow_html=True)

# --- 3. DISPLAY LOOP (CHAT HISTORY) ---
    if not st.session_state.history:
        st.info("👋 Welcome! Describe your research goal to generate a strategy and see initial findings.")
    
    for i, entry in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.markdown(f"**Research Goal:** {entry['goal']}")
        
        with st.chat_message("assistant"):
            if entry.get('summary'):
                st.markdown(f"""
                <div class="summary-box">
                    <div class="summary-title">📝 Descriptive Literature Feedback</div>
                    <div style="line-height: 1.6; color: #444;">{entry['summary']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            p = entry.get('pico_dict', {})
            
            # Change from 4 columns to 6 columns
            cols = st.columns(4)
            
            # Define the labels and the data to pull
            # Note: We pull from 'pico_dict' for PICO and 'entry' for criteria
            cards = [
                ("Population", p.get('p')),
                ("Intervention", p.get('i')),
                ("Comparator", p.get('c')),
                ("Outcome", p.get('o'))
            ]
            
            for idx, (label, value) in enumerate(cards):
                # Clean up empty values
                display_text = value if value and str(value).strip() else "None specified"
                
                cols[idx].markdown(f"""
                    <div class="pico-card">
                        <div class="pico-header">{label}</div>
                        <div class="pico-content">{display_text}</div>
                    </div>
                """, unsafe_allow_html=True)

            with st.expander("🧬 Strategy: Criteria & Search String", expanded=False):
                col_inc, col_excl = st.columns(2)
                
                with col_inc:
                    st.markdown("**✅ Inclusion Criteria**")
                    inc_list = entry.get('inclusion', [])
                    if isinstance(inc_list, list) and inc_list:
                        for item in inc_list:
                            st.markdown(f"- {item}")
                    else:
                        st.write("None specified")
                
                with col_excl:
                    st.markdown("**❌ Exclusion Criteria**")
                    excl_list = entry.get('exclusion', [])
                    if isinstance(excl_list, list) and excl_list:
                        for item in excl_list:
                            st.markdown(f"- {item}")
                    else:
                        st.write("None specified")
                
                st.divider()
                st.markdown("**🔍 Final MeSH Search String**")
                st.code(entry.get('query', ''), language="sql")

    # --- 4. REFINEMENT SUGGESTIONS ---
    suggestion_to_process = None
    if st.session_state.history:
        last_entry = st.session_state.history[-1]
        suggs = last_entry.get('suggestions', [])
        if suggs:
            st.write("---")
            st.caption("✨ **Suggested Question Refinements**")
            s_cols = st.columns(len(suggs))
            for idx, s in enumerate(suggs):
                if s_cols[idx].button(s, key=f"btn_sugg_{len(st.session_state.history)}_{idx}", use_container_width=True):
                    suggestion_to_process = s

    # --- 5. INPUT HANDLING ---
    user_input = st.chat_input("Ask a question or refine your research goal...")
    final_input = suggestion_to_process if suggestion_to_process else user_input

    if final_input:
        with st.status("🧬 Analyzing Evidence...", expanded=True):
            analysis = AIService.infer_pico_and_query(final_input, model_name, st.session_state.goal)
            
            # Update PICO and Criteria State
            st.session_state.pico.population = analysis.get('p', '')
            st.session_state.pico.intervention = analysis.get('i', '')
            st.session_state.pico.comparator = analysis.get('c', '')
            st.session_state.pico.outcome = analysis.get('o', '')
            st.session_state.inclusion_list = analysis.get('inclusion', [])
            st.session_state.exclusion_list = analysis.get('exclusion', [])
            
            mesh_query = analysis.get('query') or AIService.generate_mesh_query(st.session_state.pico, model_name)
            st.session_state.query = mesh_query 

            quick_papers = DataAggregator.fetch_all(mesh_query, active_sources, limit=5)
            summary = AIService.generate_brainstorm_summary(final_input, quick_papers, model_name)
            suggs = AIService.get_refinement_suggestions(final_input, quick_papers, model_name)

            st.session_state.history.append({
                "goal": final_input,
                "query": mesh_query,
                "summary": summary,
                "pico_dict": analysis,
                "suggestions": suggs,
                "inclusion": st.session_state.inclusion_list,
                "exclusion": st.session_state.exclusion_list
            })
            st.session_state.goal = final_input
            st.rerun()

# --- 6. STRATEGY COMMAND CENTER (Editable review) ---
    if st.session_state.history:
        st.write("---")
        st.subheader("🛠️ Strategy Review & Execution")

        with st.container(border=True):
            st.markdown("**🎯 Review PICO & Criteria**")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.session_state.pico.population = st.text_area("Population", value=st.session_state.pico.population, height=70)
                st.session_state.pico.intervention = st.text_area("Intervention", value=st.session_state.pico.intervention, height=70)
                
                # Inclusion Criteria Input
                current_inc = ", ".join(st.session_state.inclusion_list) if isinstance(st.session_state.inclusion_list, list) else st.session_state.inclusion_list
                new_inc = st.text_area("Inclusion Criteria (comma separated)", value=current_inc, height=70)
                st.session_state.inclusion_list = [x.strip() for x in new_inc.split(",") if x.strip()]
                
            with p_col2:
                st.session_state.pico.comparator = st.text_area("Comparator", value=st.session_state.pico.comparator, height=70)
                st.session_state.pico.outcome = st.text_area("Outcome", value=st.session_state.pico.outcome, height=70)
                
                # Exclusion Criteria Input
                current_excl = ", ".join(st.session_state.exclusion_list) if isinstance(st.session_state.exclusion_list, list) else st.session_state.exclusion_list
                new_excl = st.text_area("Exclusion Criteria (comma separated)", value=current_excl, height=70)
                st.session_state.exclusion_list = [x.strip() for x in new_excl.split(",") if x.strip()]

            st.session_state.query = st.text_area("Final Search String", value=st.session_state.query, height=100)

        if st.button("🚀 Run Full Systematic Review", type="primary", use_container_width=True):
            with st.status("🔍 Searching and AI-Screening...", expanded=True) as status:
                # 1. Fetching
                all_p = DataAggregator.fetch_all(
                    st.session_state.query, 
                    active_sources, 
                    max_per_source=num_per_source, 
                    uploaded_files=uploaded_files
                )
                
                # 2. Deduplication
                unique, _ = Deduplicator.run(all_p)
                
                # 3. Setup Screening Variables
                screened = []
                reasons = {}
                screened_count = 0 
                progress_bar = st.progress(0)
                
                # 4. Screening Loop
                for idx, p in enumerate(unique):
                    res = AIService.screen_paper(
                        p, 
                        st.session_state.pico, 
                        model_name, 
                        st.session_state.inclusion_list, 
                        st.session_state.exclusion_list
                    )
                    
                    # Normalize logic for Inclusion
                    decision_val = str(res.get('decision', 'Exclude')).strip().lower()
                    is_included = "include" in decision_val
                    
                    # Create the row for the Results Table (matches UIComponents expectations)
                    screened.append({
                        "Source": p.source,
                        "Title": p.title,
                        "Decision": "✅ Include" if is_included else "❌ Exclude",
                        "Design": res.get('design', 'N/A'),
                        "Sample Size": res.get('sample_size', 'N/A'),
                        "Reason": res.get('reason', 'N/A')
                    })

                    # Track reasons specifically for PRISMA Diagram
                    if not is_included:
                        r = res.get('reason', 'Excluded by criteria')
                        reasons[r] = reasons.get(r, 0) + 1
                    
                    screened_count += 1
                    progress_bar.progress((idx + 1) / len(unique))
                
                # 5. Final PRISMA State Update
                st.session_state.prisma_counts.update({
                    'identified': len(all_p),
                    'duplicates_removed': len(all_p) - len(unique),
                    'screened': screened_count,
                    'excluded_total': sum(reasons.values()),
                    'exclusion_breakdown': reasons
                })
                
                # 6. Save Results and Rerun
                st.session_state.results = pd.DataFrame(screened)
                status.update(label="Systematic Review Complete!", state="complete")
                st.rerun()
                
    # --- 7. RESULTS ---
    if st.session_state.results is not None:
        st.divider()
        t1, t2 = st.tabs(["📊 Evidence Extraction", "🔍 PRISMA Flow"])
        with t1: UIComponents.render_results(st.session_state.results)
        with t2: UIComponents.render_prisma_flow()


if __name__ == "__main__":
    main()