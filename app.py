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
    if 'search_simulation' not in st.session_state:
        st.session_state.search_simulation = None
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


    # 3. DISPLAY LOOP (CHAT HISTORY) ---
    if not st.session_state.history:
        st.info("👋 Welcome! Describe your research goal to generate a strategy and see initial findings.")
    
    for i, entry in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.markdown(f"**Research Goal:** {entry['goal']}")
        
        with st.chat_message("assistant"):
            if entry.get('formal_question'):
                st.info(f"**Research Question:** *{entry['formal_question']}*")
            if entry.get('summary'):
                with st.container():
                    st.markdown(entry['summary'], unsafe_allow_html=True)
            
            p = entry.get('pico_dict', {})
            cols = st.columns(4)
            # Define the labels and the data to pull

            cards = [
                ("Population", p.get('p')),
                ("Intervention", p.get('i')),
                ("Comparator", p.get('c')),
                ("Outcome", p.get('o'))
            ]
            
            for idx, (label, value) in enumerate(cards):
                display_text = value if value and str(value).strip() else "None specified"
                cols[idx].markdown(f"""
                    <div class="pico-card">
                        <div class="pico-header">{label}</div>
                        <div class="pico-content">{display_text}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div style="margin-top: 25px;"></div>', unsafe_allow_html=True)
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

    # --- 4. REFINEMENT SUGGESTIONS (Inline) ---
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

    # --- 5. CLINICAL BRAINSTORMING BUBBLES ---
    if st.session_state.get('goal') and st.session_state.results is None:
        st.write("---")
        st.caption("**Refinements Suggestions**")
        cat_cols = st.columns([1, 1, 1, 1, 3])
        categories = ["Population", "Intervention", "Comparator", "Outcome"]
        
        for idx, cat in enumerate(categories):
            if cat_cols[idx].button(cat, key=f"brainstorm_{cat}"):
                with st.spinner(f"Analyzing {cat} for your specific goal..."):
                    st.session_state['active_cat'] = cat.lower()
                    st.session_state['suggestions'] = AIService.get_pico_suggestion(
                        st.session_state.goal, 
                        cat.lower()
                    )
                st.rerun()

        if st.session_state.get('active_cat') and st.session_state.get('suggestions'):
            active_cat = st.session_state['active_cat']
            st.info(f"Clinical suggestions to refine your **{active_cat.upper()}**:")
            sug_cols = st.columns(3)
            for i, opt in enumerate(st.session_state['suggestions']):
                if sug_cols[i].button(opt, key=f"val_{i}", use_container_width=True):
                    setattr(st.session_state.pico, active_cat, opt)
                    del st.session_state['active_cat']
                    del st.session_state['suggestions']
                    st.rerun()

    # 6. INPUT HANDLING (KEEP THIS AT THE BOTTOM, NOT NESTED) ---
    user_input = st.chat_input("Ask a question or refine your research goal...")
    
    final_input = suggestion_to_process if suggestion_to_process else user_input

    if final_input:
        with st.status("🧬 Analyzing Evidence...", expanded=True):
            analysis = AIService.infer_pico_and_query(final_input, model_name, st.session_state.goal)

            st.session_state.pico.population = analysis.get('p', '')
            st.session_state.pico.intervention = analysis.get('i', '')
            st.session_state.pico.comparator = analysis.get('c', '')
            st.session_state.pico.outcome = analysis.get('o', '')
            st.session_state.inclusion_list = analysis.get('inclusion', [])
            st.session_state.exclusion_list = analysis.get('exclusion', [])
            
            formal_q = AIService.generate_formal_question(
                st.session_state.pico, 
                model_name, 
                st.session_state.history
            )
            mesh_query = analysis.get('query') or AIService.generate_mesh_query(st.session_state.pico, model_name)
            st.session_state.query = mesh_query 

            quick_papers, _ = DataAggregator.fetch_all(mesh_query, active_sources, limit=5)
            summary = AIService.generate_brainstorm_summary(final_input, quick_papers, model_name)
            suggs = AIService.get_refinement_suggestions(final_input, quick_papers, model_name)

            st.session_state.history.append({
                "goal": final_input,
                "query": mesh_query,
                "formal_question": formal_q,
                "summary": summary,
                "pico_dict": analysis,
                "suggestions": suggs,
                "inclusion": st.session_state.inclusion_list,
                "exclusion": st.session_state.exclusion_list
            })
            st.session_state.goal = final_input
            st.rerun()

    # 6. STRATEGY COMMAND CENTER (Editable review)
    if st.session_state.history:
        st.write("---")
        st.subheader("Strategy Review")

        with st.container(border=True):
            st.markdown("**Review PICO & Criteria**")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.session_state.pico.population = st.text_area("Population", value=st.session_state.pico.population, height=70)
                st.session_state.pico.intervention = st.text_area("Intervention", value=st.session_state.pico.intervention, height=70)
                
                current_inc = ", ".join(st.session_state.inclusion_list) if isinstance(st.session_state.inclusion_list, list) else st.session_state.inclusion_list
                new_inc = st.text_area("Inclusion Criteria (comma separated)", value=current_inc, height=70)
                st.session_state.inclusion_list = [x.strip() for x in new_inc.split(",") if x.strip()]
                
            with p_col2:
                st.session_state.pico.comparator = st.text_area("Comparator", value=st.session_state.pico.comparator, height=70)
                st.session_state.pico.outcome = st.text_area("Outcome", value=st.session_state.pico.outcome, height=70)
                
                current_excl = ", ".join(st.session_state.exclusion_list) if isinstance(st.session_state.exclusion_list, list) else st.session_state.exclusion_list
                new_excl = st.text_area("Exclusion Criteria (comma separated)", value=current_excl, height=70)
                st.session_state.exclusion_list = [x.strip() for x in new_excl.split(",") if x.strip()]

            st.session_state.query = st.text_area("Final Search String", value=st.session_state.query, height=100)

            # SIMULATION AND SEARCH BUTTONS 
            col_sim, col_run = st.columns([1, 2])
            
            with col_sim:
                if st.button("Simulate Yield", use_container_width=True, type="primary"):
                    api_sources = [s for s in active_sources if s not in ["Local PDFs", "Big 3 Journals"]]
                    yield_results = DataAggregator.simulate_yield(st.session_state.query, api_sources)
                    st.session_state.search_simulation = yield_results
                    with st.spinner("Calculating..."):
                        yield_results = DataAggregator.simulate_yield(st.session_state.query, api_sources)
                        st.session_state.search_simulation = yield_results

            with col_run:
                run_search = st.button("Run Database Search", type="primary", use_container_width=True, key="run_search_main")

        # DISPLAY SIMULATION TABLE 
        if st.session_state.search_simulation:
            with st.expander("📈 Predicted Yield per Database", expanded=True):
                sim_rows = []
                total_yield = 0
                for source, count in st.session_state.search_simulation.items():
                    sim_rows.append({"Database": source, "Paper Count": count, "Query": st.session_state.query})
                    if isinstance(count, int): total_yield += count
                
                st.table(pd.DataFrame(sim_rows))
                st.metric("Aggregate Potential Results", f"{total_yield:,} papers")
                if st.button("Clear Simulation"):
                    st.session_state.search_simulation = None
                    st.rerun()

        if run_search:
            st.session_state.search_simulation = None # Clear sim on real run
            
            with st.status("🔍 Searching and AI-Screening...", expanded=True) as status:
                # 1. Fetching
                all_p, source_counts = DataAggregator.fetch_all(
                    st.session_state.query, 
                    active_sources, 
                    max_per_source=num_per_source, 
                    uploaded_files=uploaded_files
                )
                
                # 2. Deduplication
                unique, duplicates = Deduplicator.run(all_p)
                
                # 3. Setup Screening Variables
                screened = []
                reasons = {}
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
                    
                    decision_val = str(res.get('decision', 'Exclude')).strip().lower()
                    is_included = "include" in decision_val
                    
                    screened.append({
                        "Source": p.source,
                        "Title": p.title,
                        "URL": p.url,
                        "Decision": "✅ Include" if is_included else "❌ Exclude",
                        "Reason": res.get('reason', 'N/A'),
                        "Abstract": p.abstract 
                    })

                    if not is_included:
                        r = res.get('reason', 'Excluded by criteria')
                        reasons[r] = reasons.get(r, 0) + 1
                    
                    progress_bar.progress((idx + 1) / len(unique))
                
                # 5. Final PRISMA State Update
                raw_total = len(all_p)
                unique_total = len(unique)
                dupes_removed = len(duplicates)
                total_excluded = sum(reasons.values())
                final_included = unique_total - total_excluded

                st.session_state.prisma_counts.update({
                    'identified': raw_total,
                    'source_counts': source_counts,
                    'duplicates_removed': dupes_removed,
                    'screened': unique_total,
                    'excluded_total': total_excluded,
                    'exclusion_breakdown': reasons,
                    'included_final': final_included 
                })
                
                # 6. Save Results and Rerun
                st.session_state.results = pd.DataFrame(screened)
                status.update(label="Data Fetching Complete!", state="complete")
                st.rerun()

    # 7. RESULTS & EVIDENCE TABLE 
    if st.session_state.results is not None:
        st.divider()
        t1, t2, t3 = st.tabs(["📄 Abstract Screening", "🔬 Full-Text Evidence", "🔍 PRISMA Flow"])
        
        with t1:
            UIComponents.render_results(st.session_state.results)
            passed = st.session_state.results[st.session_state.results['Decision'].str.contains("Include")]
            if not passed.empty and 'full_text_results' not in st.session_state:
                st.info(f"🎯 {len(passed)} papers are ready for Full-Text Extraction.")
                if st.button("Begin Full-Text Screening", type="primary", use_container_width=True):
                    with st.status("Performing Full-Text Analysis...", expanded=True) as status:
                        final_rows = []
                        ft_reasons = {} 

                        for _, row in passed.iterrows():
                            res = AIService.screen_full_text(row.get('Abstract', ""), st.session_state.pico, st.session_state.custom_model)
                            
                            is_included = "Include" in str(res.get('decision', ''))
                            entry = row.to_dict()
                            entry['Final Decision'] = "✅ Included" if is_included else "❌ Excluded"
                            entry['Decision Logic'] = res.get('reason', 'N/A')
                            entry['Supporting Citation'] = res.get('citation', 'N/A')
                            final_rows.append(entry)

                            if not is_included:
                                raw_reason = res.get('reason', 'Criteria mismatch')
                                bucket = " ".join(raw_reason.split()[:4]).strip().title()
                                ft_reasons[bucket] = ft_reasons.get(bucket, 0) + 1

                        st.session_state.full_text_results = pd.DataFrame(final_rows)
                        
                        st.session_state.prisma_counts.update({
                            'ft_exclusion_breakdown': ft_reasons, 
                            'included_final': len([d for d in final_rows if "✅" in d['Final Decision']])
                        })
                        st.rerun()

        with t2:
            if 'full_text_results' in st.session_state:
                st.dataframe(
                    st.session_state.full_text_results[['Title', 'Final Decision', 'Decision Logic', 'Supporting Citation']],
                    column_config={
                        "Supporting Citation": st.column_config.TextColumn("Direct Quote / Line Citation", width="large"),
                        "Decision Logic": st.column_config.TextColumn("Exclusion/Inclusion Reason", width="medium"),
                    },
                    hide_index=True, use_container_width=True
                )
            else:
                st.info("Complete the Abstract Screening in Tab 1 to unlock Full-Text evidence.")

        with t3:
            UIComponents.render_prisma_flow()


if __name__ == "__main__":
    main()