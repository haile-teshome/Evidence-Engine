import streamlit as st
import pandas as pd
import io
import time
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
        
        /* Color entire cells based on decision */
        .dataframe td {
            padding: 12px !important;
        }
        .cell-include {
            background-color: #d4edda !important;
            color: #155724 !important;
            font-weight: 500;
        }
        .cell-exclude {
            background-color: #f8d7da !important;
            color: #721c24 !important;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. Render Sidebar (includes navigation)
    model_name, active_sources, uploaded_files, num_per_source = UIComponents.render_sidebar()

    # 3. MAIN CONTENT - Based on sidebar navigation
    current_page = st.session_state.get('current_page', 'home')
    
    # --- PAGE 1: HOME / SEARCH & CHAT ---
    if current_page == "home":
        
        if not st.session_state.history:
            st.info("👋 Welcome! Describe your research goal to generate a strategy and see initial findings.")
        
        # Display Chat History
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
                        st.markdown("**Include Criteria**")
                        inc_list = entry.get('inclusion', [])
                        if isinstance(inc_list, list) and inc_list:
                            for item in inc_list:
                                st.markdown(f"- {item}")
                        else:
                            st.write("None specified")
                    
                    with col_excl:
                        st.markdown("**Exclude Criteria**")
                        excl_list = entry.get('exclusion', [])
                        if isinstance(excl_list, list) and excl_list:
                            for item in excl_list:
                                st.markdown(f"- {item}")
                        else:
                            st.write("None specified")
                    
                    st.divider()
                    st.markdown("**🔍 Final MeSH Search String**")
                    st.code(entry.get('query', ''), language="sql")
        
        # Refinement Suggestions
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
        
        # Clinical Brainstorming Bubbles
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
        
        # Strategy Review Section (when history exists)
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
        
        # Chat Input (at bottom)
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
    
    # --- PAGE 2: SIMULATION ---
    elif current_page == "simulation":
        
        if st.session_state.history:
            # Editable Search String Section
            st.markdown("### 🔍 Search String Editor")
            
            # Initialize simulation query in session state if not exists
            if 'sim_query' not in st.session_state:
                st.session_state.sim_query = st.session_state.query
            
            # Editable search string with live updates
            edited_query = st.text_area(
                "Edit search string to see yield changes:",
                value=st.session_state.sim_query,
                height=100,
                key="sim_query_editor"
            )
            
            # Update session state when query changes
            if edited_query != st.session_state.sim_query:
                st.session_state.sim_query = edited_query
                # Clear previous simulation results when query changes
                st.session_state.search_simulation = None
            
            # Agentic AI Optimization Button - Per Source
            col_optimize, col_sim, col_clear = st.columns([1, 1, 2])
            
            with col_optimize:
                if st.button("🤖 AI Optimize Per Source", use_container_width=True, type="secondary"):
                    with st.spinner("AI optimizing search strings for each database..."):
                        per_source_results = AIService.optimize_search_string_per_source(
                            st.session_state.sim_query,
                            st.session_state.pico,
                            model_name,
                            [s for s in active_sources if s not in ["Local PDFs", "Big 3 Journals"]]
                        )
                        st.session_state.per_source_optimization = per_source_results
                        st.session_state.search_simulation = {source: data["yield"] for source, data in per_source_results.items()}
                    st.rerun()
            
            with col_sim:
                if st.button("🚀 Run Simulation", use_container_width=True, type="primary"):
                    api_sources = [s for s in active_sources if s not in ["Local PDFs", "Big 3 Journals"]]
                    with st.spinner("Calculating yields..."):
                        yield_results = DataAggregator.simulate_yield(st.session_state.sim_query, api_sources)
                        st.session_state.search_simulation = yield_results
                    st.rerun()
            
            with col_clear:
                if st.session_state.search_simulation:
                    if st.button("Clear Simulation", use_container_width=True):
                        st.session_state.search_simulation = None
                        st.rerun()
            
            # Display Simulation Results
            if st.session_state.search_simulation:
                st.success("✅ Simulation complete!")
                
                # Check if we have per-source optimization results
                if st.session_state.get('per_source_optimization'):
                    st.markdown("### 📊 Per-Source Optimized Results")
                    
                    for source, data in st.session_state.per_source_optimization.items():
                        with st.expander(f"**{source}** - {data['yield']:,} papers", expanded=True):
                            st.markdown("**Optimized Query:**")
                            st.code(data['query'], language="sql")
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                # Show the modified query vs original
                                if data['query'] != st.session_state.sim_query:
                                    st.markdown("*Query was optimized for this database*")
                                else:
                                    st.markdown("*Using base query*")
                            with col2:
                                if st.button(f"Use for {source}", key=f"use_{source}"):
                                    st.session_state.sim_query = data['query']
                                    st.rerun()
                else:
                    # Standard simulation display
                    sim_rows = []
                    total_yield = 0
                    for source, count in st.session_state.search_simulation.items():
                        sim_rows.append({"Database": source, "Paper Count": count, "Query": st.session_state.sim_query[:50] + "..."})
                        if isinstance(count, int): 
                            total_yield += count
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.dataframe(pd.DataFrame(sim_rows), hide_index=True, use_container_width=True)
                    with col2:
                        st.metric("Total Potential", f"{total_yield:,}", help="Aggregate potential results across all databases")
                    
                # Option to apply optimized query to main search
                if st.session_state.sim_query != st.session_state.query:
                    if st.button("✅ Apply This Search String", use_container_width=True, type="primary"):
                        st.session_state.query = st.session_state.sim_query
                        st.success("Search string updated!")
                        st.rerun()
        else:
            st.info("Enter a research goal in the 'Search & Chat' tab first to enable simulation.")
    
    # --- PAGE 3: ABSTRACT SCREENING ---
    elif current_page == "abstract":
        
        # Run Search Button
        if st.session_state.history:
            if st.button("🔍 Run Database Search", type="primary", use_container_width=True, key="run_search_tab"):
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
                            "Decision": "Include" if is_included else "Exclude",
                            "Reason": res.get('reason', 'N/A'),
                            "Abstract": p.abstract,
                            **{criterion: res.get(criterion, 'ERROR') for criterion in st.session_state.get('inclusion_list', []) + st.session_state.get('exclusion_list', [])}
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
                    
                    # 6. Save Results
                    st.session_state.results = pd.DataFrame(screened)
                    status.update(label=f"✅ Found {len(screened)} papers!", state="complete")
                    st.rerun()
        
        # Display Results
        if st.session_state.results is not None and not st.session_state.results.empty:
            st.success(f"✅ {len(st.session_state.results)} papers screened")
            UIComponents.render_results(st.session_state.results)
            
            passed = st.session_state.results[st.session_state.results['Decision'].str.contains("Include")]
            if not passed.empty:
                st.info(f"🎯 {len(passed)} papers passed abstract screening and are ready for Full-Text Extraction.")
        elif st.session_state.results is not None and st.session_state.results.empty:
            st.warning("No papers found. Try adjusting your query or criteria.")
        else:
            st.info("Click 'Run Database Search' to start screening papers.")
    
    # --- PAGE 4: FULL-TEXT EVIDENCE ---
    elif current_page == "fulltext":
        
        if st.session_state.results is not None:
            passed = st.session_state.results[st.session_state.results['Decision'].str.contains("Include")]
            
            if not passed.empty:
                if 'full_text_results' not in st.session_state:
                    st.info(f"🎯 {len(passed)} papers ready for full-text analysis.")
                    if st.button("🚀 Begin Full-Text Screening", type="primary", use_container_width=True):
                        with st.status("Performing Full-Text Analysis...", expanded=True) as status:
                            final_rows = []
                            ft_reasons = {} 
                            
                            # Create progress bar
                            total_papers = len(passed)
                            progress_bar = st.progress(0, text=f"Screening 0/{total_papers} papers")
                            current_paper = st.empty()

                            for idx, (_, row) in enumerate(passed.iterrows()):
                                # Update current paper display
                                current_paper.info(f"📄 Currently screening: {row.get('Title', 'Unknown Paper')[:100]}...")
                                
                                # Update progress bar
                                progress = (idx + 1) / total_papers
                                progress_bar.progress(progress, text=f"Screening {idx + 1}/{total_papers} papers")
                                
                                res = AIService.screen_full_text(row.to_dict(), st.session_state.pico, st.session_state.custom_model)
                                
                                is_included = "Include" in str(res.get('decision', ''))
                                entry = row.to_dict()
                                entry['Decision'] = "Include" if is_included else "Exclude"
                                entry['Reason'] = res.get('reason', 'N/A')
                                entry['Abstract'] = row.get('Abstract', 'N/A')
                                # Add criteria evaluations
                                for criterion in st.session_state.get('inclusion_list', []) + st.session_state.get('exclusion_list', []):
                                    entry[criterion] = res.get(criterion, 'ERROR')
                                final_rows.append(entry)

                                if not is_included:
                                    raw_reason = res.get('reason', 'Criteria mismatch')
                                    bucket = " ".join(raw_reason.split()[:4]).strip().title()
                                    ft_reasons[bucket] = ft_reasons.get(bucket, 0) + 1

                            # Clear current paper display and progress bar
                            current_paper.empty()
                            progress_bar.empty()

                            st.session_state.full_text_results = pd.DataFrame(final_rows)
                            
                            st.session_state.prisma_counts.update({
                                'ft_exclusion_breakdown': ft_reasons, 
                                'included_final': len([d for d in final_rows if "Include" in d['Decision']])
                            })
                            status.update(label="✅ Full-Text Analysis Complete!", state="complete")
                            st.rerun()
                else:
                    st.success("✅ Full-text screening complete!")
                    
                    # Style Full-Text DataFrame to color cells based on decision and criteria
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
                    styled_ft_results = st.session_state.full_text_results.copy()
                    if 'Decision' in styled_ft_results.columns:
                        styled_ft_results = styled_ft_results.style.map(color_decisions, subset=['Decision'])
                    
                    # Apply styling to criteria columns
                    inclusion_criteria = st.session_state.get('inclusion_list', [])
                    exclusion_criteria = st.session_state.get('exclusion_list', [])
                    
                    for criterion in inclusion_criteria + exclusion_criteria:
                        if criterion in styled_ft_results.columns:
                            styled_ft_results = styled_ft_results.map(color_criteria, subset=[criterion])
                    
                    st.dataframe(
                        styled_ft_results,
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
            else:
                st.warning("No papers passed abstract screening. Adjust your criteria and rerun the search.")
        else:
            st.info("Complete the Abstract Screening tab first to unlock Full-Text evidence.")
    
    # --- PAGE 5: PRISMA FLOW ---
    elif current_page == "prisma":
        UIComponents.render_prisma_flow()


if __name__ == "__main__":
    main()