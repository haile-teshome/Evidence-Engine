import streamlit as st
import pandas as pd
import io
import time
import requests
from bs4 import BeautifulSoup
import re
from config import Config
from state_manager import SessionState
from ui_components import UIComponents
from utils import AIService, Deduplicator
from data_services import DataAggregator

def extract_tables_from_paper(paper_title: str, paper_url: str, paper_source: str, extraction_type: str) -> list:
    """
    Extract tables from paper full text using various methods.
    Returns a list of dictionaries with table information.
    """
    tables = []
    
    try:
        # Method 1: Try to fetch and parse HTML content (for online papers)
        if paper_url and paper_url.startswith('http'):
            response = requests.get(paper_url, timeout=20)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Method 1a: Look for HTML tables
                html_tables = soup.find_all('table')
                for i, table in enumerate(html_tables):
                    table_data = []
                    rows = table.find_all('tr')
                    
                    for row in rows:
                        row_data = []
                        cells = row.find_all(['td', 'th'])
                        for cell in cells:
                            text = cell.get_text(strip=True)
                            row_data.append(text)
                        
                        if row_data:  # Only add non-empty rows
                            table_data.append(row_data)
                    
                    if table_data and len(table_data) > 1:  # At least header + 1 row
                        tables.append({
                            'title': f"Table {i+1} (HTML)",
                            'type': classify_table_type(table_data, extraction_type),
                            'data': table_data
                        })
                
                # Method 1b: Look for table-like structures in divs
                if not tables:
                    # Look for div elements that might contain tables
                    potential_tables = soup.find_all(['div'], class_=re.compile(r'.*table.*', re.I))
                    for i, div in enumerate(potential_tables):
                        div_text = div.get_text(strip=True)
                        if div_text and len(div_text.split('\n')) > 2:
                            # Try to parse as table
                            lines = div_text.split('\n')
                            table_data = []
                            for line in lines:
                                if '|' in line or '\t' in line or line.count('   ') >= 2:
                                    if '|' in line:
                                        row = [cell.strip() for cell in line.split('|') if cell.strip()]
                                    elif '\t' in line:
                                        row = [cell.strip() for cell in line.split('\t') if cell.strip()]
                                    else:
                                        row = [cell.strip() for cell in line.split('  ') if cell.strip()]
                                    
                                    if len(row) > 1:
                                        table_data.append(row)
                            
                            if table_data and len(table_data) > 1:
                                tables.append({
                                    'title': f"Table {i+1} (Div)",
                                    'type': classify_table_type(table_data, extraction_type),
                                    'data': table_data
                                })
                
                # Method 1c: Look for pre-formatted text in entire page
                if not tables:
                    text_content = soup.get_text()
                    lines = text_content.split('\n')
                    current_table = []
                    
                    for line_num, line in enumerate(lines):
                        # Look for table-like patterns (more aggressive)
                        if any(sep in line for sep in ['|', '\t', '   ', '    ']):
                            # Split by common separators
                            if '|' in line:
                                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                            elif '\t' in line:
                                row = [cell.strip() for cell in line.split('\t') if cell.strip()]
                            elif '    ' in line:
                                row = [cell.strip() for cell in line.split('    ') if cell.strip()]
                            elif '   ' in line:
                                row = [cell.strip() for cell in line.split('   ') if cell.strip()]
                            else:
                                row = [cell.strip() for cell in line.split('  ') if cell.strip()]
                            
                            if len(row) > 1:  # At least 2 columns
                                current_table.append(row)
                        else:
                            # End of table if we hit a non-table line after having table data
                            if current_table and len(current_table) > 2:  # At least 3 rows for a real table
                                tables.append({
                                    'title': f"Table {len(tables)+1} (Text)",
                                    'type': classify_table_type(current_table, extraction_type),
                                    'data': current_table
                                })
                                current_table = []
                    
                    # Add any remaining table
                    if current_table and len(current_table) > 2:
                        tables.append({
                            'title': f"Table {len(tables)+1} (Text)",
                            'type': classify_table_type(current_table, extraction_type),
                            'data': current_table
                        })
                
                # Method 1d: Look for data in <pre> tags
                if not tables:
                    pre_tags = soup.find_all('pre')
                    for i, pre in enumerate(pre_tags):
                        pre_text = pre.get_text()
                        if pre_text and len(pre_text.split('\n')) > 2:
                            lines = pre_text.split('\n')
                            table_data = []
                            for line in lines:
                                if '|' in line or '\t' in line or line.count('   ') >= 2:
                                    if '|' in line:
                                        row = [cell.strip() for cell in line.split('|') if cell.strip()]
                                    elif '\t' in line:
                                        row = [cell.strip() for cell in line.split('\t') if cell.strip()]
                                    else:
                                        row = [cell.strip() for cell in line.split('  ') if cell.strip()]
                                    
                                    if len(row) > 1:
                                        table_data.append(row)
                            
                            if table_data and len(table_data) > 1:
                                tables.append({
                                    'title': f"Table {i+1} (Pre)",
                                    'type': classify_table_type(table_data, extraction_type),
                                    'data': table_data
                                })
                
                # Method 1e: Look for table captions and extract surrounding content
                if not tables:
                    # Look for common table caption patterns
                    caption_patterns = ['table', 'tab.', 'figure', 'fig.', 'results', 'data', 'summary']
                    all_text = soup.get_text().lower()
                    
                    for pattern in caption_patterns:
                        if pattern in all_text:
                            # Find lines containing table-related words
                            lines = soup.get_text().split('\n')
                            for i, line in enumerate(lines):
                                if any(word in line.lower() for word in caption_patterns):
                                    # Look at surrounding lines for tabular data
                                    surrounding_lines = lines[max(0, i-2):min(len(lines), i+5)]
                                    table_data = []
                                    
                                    for s_line in surrounding_lines:
                                        if '|' in s_line or '\t' in s_line or s_line.count('   ') >= 3:
                                            if '|' in s_line:
                                                row = [cell.strip() for cell in s_line.split('|') if cell.strip()]
                                            elif '\t' in s_line:
                                                row = [cell.strip() for cell in s_line.split('\t') if cell.strip()]
                                            else:
                                                row = [cell.strip() for cell in s_line.split('  ') if cell.strip()]
                                            
                                            if len(row) > 1:
                                                table_data.append(row)
                                    
                                    if table_data and len(table_data) > 2:
                                        tables.append({
                                            'title': f"Table {len(tables)+1} (Caption)",
                                            'type': classify_table_type(table_data, extraction_type),
                                            'data': table_data
                                        })
                                        break  # Found a table, move to next pattern
                            
                            if tables:
                                break  # Found tables, stop searching
    
    except Exception as e:
        print(f"Error extracting tables from {paper_title}: {e}")
    
    return tables

def classify_table_type(table_data: list, extraction_type: str) -> str:
    """
    Classify table type based on content and extraction preference.
    """
    if extraction_type != "All Tables":
        return extraction_type
    
    # Simple classification based on headers
    if not table_data or len(table_data) < 2:
        return "Unknown"
    
    headers = [cell.lower() for cell in table_data[0] if cell]
    
    if any(keyword in ' '.join(headers) for keyword in ['age', 'gender', 'sex', 'demographic', 'participant']):
        return "Demographics"
    elif any(keyword in ' '.join(headers) for keyword in ['outcome', 'result', 'effect', 'response']):
        return "Outcomes"
    elif any(keyword in ' '.join(headers) for keyword in ['intervention', 'treatment', 'drug', 'therapy']):
        return "Interventions"
    elif any(keyword in ' '.join(headers) for keyword in ['p-value', 'statistic', 'test', 'significance']):
        return "Statistical Results"
    elif any(keyword in ' '.join(headers) for keyword in ['adverse', 'event', 'side', 'complication']):
        return "Adverse Events"
    
    return "General"

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
        
        # Adversarial References Section
        if st.session_state.history:
            last_entry = st.session_state.history[-1]
            adversarial_papers = last_entry.get('adversarial_papers', [])
            adversarial_query = last_entry.get('adversarial_query', '')
            
            if adversarial_papers:
                st.write("---")
                st.caption("⚖️ **Adversarial References for Balanced View**")
                st.info(f"**Adversarial Search Query:** *{adversarial_query}*")
                
                st.markdown("**🔄 Contrasting Evidence:**")
                for i, paper in enumerate(adversarial_papers):
                    with st.expander(f"📄 {paper.title[:80]}{'...' if len(paper.title) > 80 else ''}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**Source:** {paper.source}")
                            if paper.abstract:
                                st.markdown(f"**Abstract:** {paper.abstract[:300]}{'...' if len(paper.abstract) > 300 else ''}")
                        with col2:
                            if paper.url:
                                st.link_button("📖 View Paper", paper.url, use_container_width=True)
                            st.markdown(f"**ID:** {paper.id}")
                
                st.markdown("*These references provide contrasting perspectives to ensure a balanced and comprehensive review of the evidence.*")
        
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
                
                # Generate adversarial search for balanced view (only on chat input)
                adversarial_query = AIService.generate_adversarial_query(st.session_state.pico, model_name)
                adversarial_papers, _ = DataAggregator.fetch_all(adversarial_query, active_sources, limit=3)
                
                st.session_state.history.append({
                    "goal": final_input,
                    "query": mesh_query,
                    "formal_question": formal_q,
                    "summary": summary,
                    "pico_dict": analysis,
                    "suggestions": suggs,
                    "inclusion": st.session_state.inclusion_list,
                    "exclusion": st.session_state.exclusion_list,
                    "adversarial_query": adversarial_query,
                    "adversarial_papers": adversarial_papers
                })
                st.session_state.goal = final_input
                st.rerun()
    
    # --- PAGE 2: SIMULATION ---
    elif current_page == "simulation":
        
        if st.session_state.history:
            # Initialize per-database search strings if not exists
            if 'per_db_queries' not in st.session_state:
                st.session_state.per_db_queries = {}
                api_sources = [s for s in active_sources if s not in ["Local PDFs"]]
                for source in api_sources:
                    st.session_state.per_db_queries[source] = st.session_state.query
            
            # Per-database editing (permanent mode)
            api_sources = [s for s in active_sources if s not in ["Local PDFs"]]
            
            for source in api_sources:
                with st.expander(f"**{source}**", expanded=False):
                    current_query = st.session_state.per_db_queries.get(source, st.session_state.query)
                    
                    col_query, col_simulate = st.columns([3, 1])
                    with col_query:
                        edited_db_query = st.text_area(
                            f"Search string for {source}:",
                            value=current_query,
                            height=80,
                            key=f"query_{source}"
                        )
                    
                    with col_simulate:
                        # Display paper count centered above button
                        if 'db_test_results' in st.session_state and st.session_state.db_test_results is not None and source in st.session_state.db_test_results:
                            test_result = st.session_state.db_test_results[source]
                            st.markdown(f"<div style='text-align: center; font-size: 1.5em; font-weight: bold; margin-bottom: 10px;'>{test_result['total_found']} Papers</div>", unsafe_allow_html=True)
                        
                        if st.button(f"Test", key=f"test_{source}", use_container_width=True, help="Run simulation for this database"):
                            with st.spinner(f"Testing {source}..."):
                                try:
                                    # For arXiv, get actual count using simulation method
                                    if source.lower() == "arxiv":
                                        # Use the same count method as simulation
                                        count_result = DataAggregator.simulate_yield(edited_db_query, [source])
                                        actual_count = count_result.get(source, 0)
                                        
                                        # Get top 10 papers for display
                                        papers, _ = DataAggregator.fetch_all(
                                            edited_db_query, 
                                            [source], 
                                            max_per_source=10, 
                                            uploaded_files=[]
                                        )
                                        
                                        # Store results in session state
                                        if 'db_test_results' not in st.session_state:
                                            st.session_state.db_test_results = {}
                                        st.session_state.db_test_results[source] = {
                                            'query': edited_db_query,
                                            'papers': papers[:10],  # Top 10 papers for display
                                            'total_found': actual_count  # Actual total from arXiv
                                        }
                                    else:
                                        # For other databases, use regular method
                                        all_papers, _ = DataAggregator.fetch_all(
                                            edited_db_query, 
                                            [source], 
                                            max_per_source=1000,  # Get more to get accurate total
                                            uploaded_files=[]
                                        )
                                        
                                        # Store results in session state
                                        if 'db_test_results' not in st.session_state:
                                            st.session_state.db_test_results = {}
                                        st.session_state.db_test_results[source] = {
                                            'query': edited_db_query,
                                            'papers': all_papers[:10],  # Top 10 papers for display
                                            'total_found': len(all_papers)  # Total papers found
                                        }
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error testing {source}: {str(e)}")
                    
                    if edited_db_query != current_query:
                        st.session_state.per_db_queries[source] = edited_db_query
                        st.session_state.search_simulation = None
                    
                    # Display top 10 results if available
                    if 'db_test_results' in st.session_state and st.session_state.db_test_results is not None and source in st.session_state.db_test_results:
                        test_result = st.session_state.db_test_results[source]
                        
                        if test_result['papers']:
                            st.markdown("**Top 10 Results:**")
                            for i, paper in enumerate(test_result['papers'], 1):
                                with st.container():
                                    st.markdown(f"**{i}.** {paper.title}")
                                    st.caption(f"🔗 {paper.url}")
                                    st.divider()
            
            # Action buttons
            api_sources = [s for s in active_sources if s not in ["Local PDFs"]]
            col_optimize, col_sim, col_clear = st.columns([1, 1, 2])
            
            with col_optimize:
                if st.button("🤖 AI Optimize Per Source", use_container_width=True, type="secondary"):
                    with st.spinner("AI optimizing search strings for each database..."):
                        per_source_results = AIService.optimize_search_string_per_source(
                            st.session_state.sim_query if edit_mode == "Unified Search String" else st.session_state.per_db_queries.get(api_sources[0], st.session_state.query),
                            st.session_state.pico,
                            model_name,
                            api_sources
                        )
                        st.session_state.per_source_optimization = per_source_results
                        # Update per-database queries with optimized versions
                        for source, data in per_source_results.items():
                            st.session_state.per_db_queries[source] = data['query']
                        st.session_state.search_simulation = {source: data["yield"] for source, data in per_source_results.items()}
                    st.rerun()
            
            with col_sim:
                if st.button("🚀 Run Simulation", use_container_width=True, type="primary"):
                    with st.spinner("Calculating yields..."):
                        # Simulate with individual queries per database
                        yield_results = {}
                        for source in api_sources:
                            source_query = st.session_state.per_db_queries.get(source, st.session_state.query)
                            source_yield = DataAggregator.simulate_yield(source_query, [source])
                            yield_results.update(source_yield)
                        st.session_state.search_simulation = yield_results
                    st.rerun()
            
            with col_clear:
                # Clear both simulation and test results
                clear_text = "Clear Results"
                if st.session_state.search_simulation and st.session_state.get('db_test_results'):
                    clear_text = "Clear All Results"
                elif st.session_state.get('db_test_results'):
                    clear_text = "Clear Test Results"
                elif st.session_state.search_simulation:
                    clear_text = "Clear Simulation"
                    
                if (st.session_state.search_simulation or st.session_state.get('db_test_results')):
                    if st.button(clear_text, use_container_width=True):
                        st.session_state.search_simulation = None
                        st.session_state.per_source_optimization = None
                        st.session_state.db_test_results = None
                        st.rerun()
            
            # Display Simulation Results
            if st.session_state.search_simulation:
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
                                    # Update per-database queries
                                    api_sources = [s for s in active_sources if s not in ["Local PDFs"]]
                                    for src in api_sources:
                                        st.session_state.per_db_queries[src] = data['query'] if src == source else st.session_state.per_db_queries.get(src, st.session_state.query)
                                    st.rerun()
                else:
                    # Enhanced simulation display with per-database queries
                    sim_rows = []
                    total_yield = 0
                    
                    for source in api_sources:
                        count = st.session_state.search_simulation.get(source, 0)
                        query_preview = st.session_state.per_db_queries.get(source, st.session_state.query)[:50] + "..."
                        sim_rows.append({
                            "Database": source, 
                            "Paper Count": count, 
                            "Query Preview": query_preview
                        })
                        if isinstance(count, int): 
                            total_yield += count
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.dataframe(pd.DataFrame(sim_rows), hide_index=True, use_container_width=True)
                    with col2:
                        st.metric("Total Potential", f"{total_yield:,}", help="Aggregate potential results across all databases")
                
                # Option to apply optimized query to main search
                current_main_query = st.session_state.per_db_queries.get(api_sources[0], st.session_state.query)
                if current_main_query != st.session_state.query:
                    if st.button("✅ Apply This Search String", use_container_width=True, type="primary"):
                        st.session_state.query = current_main_query
                        st.success("Search string updated!")
                        st.rerun()
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
                        # Add criteria evaluations based on content analysis
                        def evaluate_criteria_against_content(title, abstract, criteria_list, is_inclusion=True):
                            """Evaluate criteria based on actual title/abstract content."""
                            if not criteria_list:
                                return {}
                            
                            results = {}
                            content_text = f"{title} {abstract}".lower()
                            
                            for criterion in criteria_list:
                                criterion_words = criterion.lower().split()
                                # Remove common words and focus on meaningful terms
                                stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'this', 'that', 'these', 'those'}
                                key_terms = [w for w in criterion_words if len(w) > 2 and w not in stop_words]
                                
                                if key_terms:
                                    matches = sum(1 for term in key_terms if term in content_text)
                                    match_ratio = matches / len(key_terms)
                                    
                                    # Check if criterion is explicitly mentioned as NOT met
                                    exclusion_phrases = ['not', 'no', 'without', 'lacking', 'absent', 'missing', 'none', 'never', 'failed to', 'did not', 'excluded']
                                    has_exclusion_phrase = any(phrase in content_text for phrase in exclusion_phrases)
                                    
                                    # If exclusion phrases are found near criterion terms, mark as EXCLUDE
                                    if has_exclusion_phrase and match_ratio > 0.3:
                                        results[criterion] = "EXCLUDE"
                                    # If good match (>50%), mark as INCLUDE
                                    elif match_ratio > 0.5:
                                        results[criterion] = "INCLUDE"
                                    # If some match but not explicit, mark as UNSPECIFIED
                                    elif match_ratio > 0.1:
                                        results[criterion] = "UNSPECIFIED"
                                    # No mention at all
                                    else:
                                        results[criterion] = "UNSPECIFIED"
                                else:
                                    results[criterion] = "UNSPECIFIED"
                            
                            return results

                        def make_final_decision(criteria_evaluations, inclusion_criteria, exclusion_criteria):
                            """Make final decision based on ALL criteria."""
                            # Check if ALL inclusion criteria are met (INCLUDE or UNSPECIFIED)
                            for criterion in inclusion_criteria:
                                eval_result = criteria_evaluations.get(criterion, 'UNSPECIFIED')
                                if eval_result == 'EXCLUDE':
                                    return False, f"Excluded by inclusion criterion: {criterion}"
                            
                            # Check if ANY exclusion criteria are violated (EXCLUDE)
                            for criterion in exclusion_criteria:
                                eval_result = criteria_evaluations.get(criterion, 'UNSPECIFIED')
                                if eval_result == 'EXCLUDE':
                                    return False, f"Excluded by exclusion criterion: {criterion}"
                            
                            # If no exclusions and no inclusion violations, include
                            return True, "All criteria satisfied"

                        try:
                            res = AIService.screen_paper(
                                p, 
                                st.session_state.pico, 
                                model_name, 
                                st.session_state.inclusion_list, 
                                st.session_state.exclusion_list
                            )
                        except Exception as e:
                            # Fallback if screening fails
                            res = {
                                "decision": "Include",
                                "bucket": "Screening error",
                                "reason": f"Screening failed: {str(e)[:50]}",
                                **{criterion: "ERROR" for criterion in st.session_state.get('inclusion_list', []) + st.session_state.get('exclusion_list', [])}
                            }
                        
                        # Get criteria evaluations from AI response first, then fallback to content analysis
                        ai_criteria = {criterion: res.get(criterion, 'ERROR') for criterion in st.session_state.get('inclusion_list', []) + st.session_state.get('exclusion_list', [])}
                        content_criteria = evaluate_criteria_against_content(p.title, p.abstract, st.session_state.get('inclusion_list', []) + st.session_state.get('exclusion_list', []))
                        
                        # Use AI criteria when available, otherwise use content analysis
                        final_criteria = {}
                        for criterion in st.session_state.get('inclusion_list', []) + st.session_state.get('exclusion_list', []):
                            if ai_criteria.get(criterion) not in ['ERROR', 'EXCLUDE', 'INCLUDE']:
                                final_criteria[criterion] = ai_criteria[criterion]
                            else:
                                final_criteria[criterion] = content_criteria.get(criterion, 'UNSPECIFIED')
                        
                        # Make final decision based on ALL criteria
                        is_included, decision_reason = make_final_decision(
                            final_criteria, 
                            st.session_state.get('inclusion_list', []), 
                            st.session_state.get('exclusion_list', [])
                        )
                        
                        screened.append({
                            "Source": p.source,
                            "Title": p.title,
                            "URL": p.url,
                            "Decision": "Include" if is_included else "Exclude",
                            "Reason": decision_reason,
                            "Abstract": p.abstract,
                            **final_criteria
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
                        elif 'UNSPECIFIED' in str(val):
                            return 'background-color: #fff3cd; color: #856404; font-weight: 500; padding: 8px;'
                        else:
                            return 'background-color: #e2e3e5; color: #383d41; font-weight: 500; padding: 8px;'
                    
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
    
    # --- PAGE 5: TEXT EXTRACTION ---
    elif current_page == "extraction":
        

        if st.session_state.results is not None:
            passed = st.session_state.results[st.session_state.results['Decision'].str.contains("Include")]
            
            if not passed.empty:
                st.info(f"🎯 {len(passed)} papers available for table extraction.")
                
                # Table extraction options
                st.markdown("### Extraction Settings")
                col1, col2 = st.columns(2)
                
                with col1:
                    extraction_type = st.selectbox(
                        "Table Type to Extract:",
                        ["All Tables", "Demographics", "Outcomes", "Interventions", "Statistical Results", "Adverse Events"]
                    )
                
                with col2:
                    output_format = st.selectbox(
                        "Output Format:",
                        ["DataFrame", "CSV Export", "JSON Export", "Excel Export"]
                    )
                
                if st.button("🚀 Start Table Extraction", type="primary", use_container_width=True):
                    with st.status("📊 Extracting tables from papers...", expanded=True) as status:
                        extracted_data = []
                        
                        for idx, (_, row) in enumerate(passed.iterrows()):
                            paper_title = row.get('Title', 'Unknown Paper')
                            paper_url = row.get('URL', '')
                            paper_source = row.get('Source', 'Unknown')
                            
                            # Extract tables from full text (placeholder for actual extraction)
                            extracted_tables = extract_tables_from_paper(paper_title, paper_url, paper_source, extraction_type)
                            
                            extracted_data.append({
                                "Paper_Title": paper_title,
                                "Paper_URL": paper_url,
                                "Source": paper_source,
                                "Extracted_Tables": extracted_tables
                            })
                        
                        st.session_state.extracted_papers_data = extracted_data
                        status.update(label=f"✅ Tables extracted from {len(extracted_data)} papers!", state="complete")
                        st.rerun()
                
                # Display extracted tables if available
                    
                    for idx, (_, row) in enumerate(passed.iterrows()):
                        paper_title = row.get('Title', 'Unknown Paper')
                        paper_url = row.get('URL', '')
                        paper_source = row.get('Source', 'Unknown')
                        
                        # Extract tables from full text (placeholder for actual extraction)
                        tables = []
                        
                        # Method 1: Try to fetch and parse HTML content (for online papers)
                        if paper_url and paper_url.startswith('http'):
                            st.info(f" Extracting tables from: {paper_title}")
                            
                            response = requests.get(paper_url, timeout=20)
                            if response.status_code == 200:
                                # Parse HTML content
                                soup = BeautifulSoup(response.content, 'html.parser')
                                
                                # Method 1a: Look for HTML tables
                                html_tables = soup.find_all('table')
                                for i, table in enumerate(html_tables):
                                    table_data = []
                                    rows = table.find_all('tr')
                                    
                                    for row in rows:
                                        row_data = []
                                        cells = row.find_all(['td', 'th'])
                                        for cell in cells:
                                            text = cell.get_text(strip=True)
                                            row_data.append(text)
                                        
                                        if row_data:  # Only add non-empty rows
                                            table_data.append(row_data)
                                    
                                    if table_data and len(table_data) > 1:  # At least header + 1 row
                                        table_type = classify_table_type(table_data, extraction_type)
                                        if extraction_type == "All Tables" or table_type == extraction_type:
                                            tables.append({
                                                'title': f"Table {i+1} (HTML)",
                                                'type': table_type,
                                                'data': table_data,
                                                'source': 'HTML'
                                            })
                                
                                # Method 1b: Look for table-like structures in divs
                                if not html_tables:
                                    # Look for div elements that might contain tables
                                    potential_tables = soup.find_all(['div'], class_=re.compile(r'.*table.*', re.I))
                                    for i, div in enumerate(potential_tables):
                                        div_text = div.get_text(strip=True)
                                        if div_text and len(div_text.split('\n')) > 2:
                                            # Try to parse as table
                                            lines = div_text.split('\n')
                                            table_data = []
                                            for line in lines:
                                                if '|' in line or '\t' in line or line.count('   ') >= 2:
                                                    if '|' in line:
                                                        row = [cell.strip() for cell in line.split('|') if cell.strip()]
                                                    elif '\t' in line:
                                                        row = [cell.strip() for cell in line.split('\t') if cell.strip()]
                                                    else:
                                                        row = [cell.strip() for cell in line.split('  ') if cell.strip()]
                                                    
                                                    if len(row) > 1:
                                                        table_data.append(row)
                                            
                                            if table_data and len(table_data) > 1:
                                                tables.append({
                                                    'title': f"Table {i+1} (Div)",
                                                    'type': classify_table_type(table_data, extraction_type),
                                                    'data': table_data,
                                                    'source': 'Div'
                                                })
                                
                                # Method 1c: Look for pre-formatted text in entire page
                                if not tables:
                                    text_content = soup.get_text()
                                    lines = text_content.split('\n')
                                    current_table = []
                                    
                                    for line_num, line in enumerate(lines):
                                        # Look for table-like patterns (more aggressive)
                                        if any(sep in line for sep in ['|', '\t', '   ', '    ']):
                                            # Split by common separators
                                            if '|' in line:
                                                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                                            elif '\t' in line:
                                                row = [cell.strip() for cell in line.split('\t') if cell.strip()]
                                            elif '    ' in line:
                                                row = [cell.strip() for cell in line.split('    ') if cell.strip()]
                                            elif '   ' in line:
                                                row = [cell.strip() for cell in line.split('   ') if cell.strip()]
                                            else:
                                                row = [cell.strip() for cell in line.split('  ') if cell.strip()]
                                            
                                            if len(row) > 1:  # At least 2 columns
                                                current_table.append(row)
                                        else:
                                            # End of table if we hit a non-table line after having table data
                                            if current_table and len(current_table) > 2:  # At least 3 rows for a real table
                                                table_type = classify_table_type(current_table, extraction_type)
                                                if extraction_type == "All Tables" or table_type == extraction_type:
                                                    tables.append({
                                                        'title': f"Table {len(tables)+1} (Text)",
                                                        'type': table_type,
                                                        'data': current_table,
                                                        'source': 'Text'
                                                    })
                                                current_table = []
                                    
                                    # Add any remaining table
                                    if current_table and len(current_table) > 2:
                                        table_type = classify_table_type(current_table, extraction_type)
                                        if extraction_type == "All Tables" or table_type == extraction_type:
                                            tables.append({
                                                'title': f"Table {len(tables)+1} (Text)",
                                                'type': table_type,
                                                'data': current_table,
                                                'source': 'Text'
                                            })
                                
                                # Method 1d: Look for data in <pre> tags
                                if not tables:
                                    pre_tags = soup.find_all('pre')
                                    for i, pre in enumerate(pre_tags):
                                        pre_text = pre.get_text()
                                        if pre_text and len(pre_text.split('\n')) > 2:
                                            lines = pre_text.split('\n')
                                            table_data = []
                                            for line in lines:
                                                if '|' in line or '\t' in line or line.count('   ') >= 2:
                                                    if '|' in line:
                                                        row = [cell.strip() for cell in line.split('|') if cell.strip()]
                                                    elif '\t' in line:
                                                        row = [cell.strip() for cell in line.split('\t') if cell.strip()]
                                                    else:
                                                        row = [cell.strip() for cell in line.split('  ') if cell.strip()]
                                                    
                                                    if len(row) > 1:
                                                        table_data.append(row)
                                            
                                            if table_data and len(table_data) > 1:
                                                tables.append({
                                                    'title': f"Table {i+1} (Pre)",
                                                    'type': classify_table_type(table_data, extraction_type),
                                                    'data': table_data,
                                                    'source': 'Pre'
                                                })
                            else:
                                st.warning(f" Could not access {paper_url} (status: {response.status_code})")
                        
                        extracted_data.append({
                            "Paper_Title": paper_title,
                            "Paper_URL": paper_url,
                            "Source": paper_source,
                            "Extracted_Tables": tables
                        })
                    
                    st.session_state.extracted_papers_data = extracted_data
                    status.update(label=f" Tables extracted from {len(extracted_data)} papers!", state="complete")
                    st.rerun()
            
            # Display extracted tables if available
            if 'extracted_papers_data' in st.session_state and st.session_state.extracted_papers_data:
                st.success(f" Tables extracted from {len(st.session_state.extracted_papers_data)} papers")
                
                # Display each paper's tables
                for paper_data in st.session_state.extracted_papers_data:
                    with st.expander(f"📄 {paper_data['Paper_Title']}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Source:** {paper_data['Source']}")
                            if paper_data['Paper_URL']:
                                st.link_button("📖 View Full Paper", paper_data['Paper_URL'], use_container_width=True)
                        
                        with col2:
                            st.metric("Tables Found", len(paper_data['Extracted_Tables']))
                        
                        # Display extracted tables
                        if paper_data['Extracted_Tables']:
                            st.markdown("### 📊 Extracted Tables")
                            
                            # Use tabs instead of nested expanders
                            table_tabs = st.tabs([f"📋 Table {i+1}: {table['title']}" for i, table in enumerate(paper_data['Extracted_Tables'])])
                            
                            for i, (table, tab) in enumerate(zip(paper_data['Extracted_Tables'], table_tabs)):
                                with tab:
                                    # Convert table data to DataFrame for display
                                    if table['data']:
                                        df = pd.DataFrame(table['data'])
                                        st.dataframe(df, use_container_width=True)
                                        
                                        # Table metadata
                                        col_meta1, col_meta2, col_meta3 = st.columns(3)
                                        with col_meta1:
                                            st.metric("Rows", len(table['data']))
                                        with col_meta2:
                                            st.metric("Columns", len(table['data'][0]) if table['data'] else 0)
                                        with col_meta3:
                                            st.metric("Type", table['type'])
                                        
                                        # Export individual table
                                        if st.button(f"📥 Export Table {i+1}", key=f"export_table_{paper_data['Paper_Title'][:20]}_{i}"):
                                            csv_data = df.to_csv(index=False)
                                            st.download_button(
                                                label=f"Download Table {i+1} CSV",
                                                data=csv_data,
                                                file_name=f"table_{i+1}_{paper_data['Paper_Title'][:30].replace(' ', '_')}.csv",
                                                mime="text/csv"
                                            )
                        else:
                            st.info("No tables found in this paper.")
                
                if st.button("📄 Export All Tables to JSON", use_container_width=True):
                    import json
                    json_data = json.dumps(st.session_state.extracted_papers_data, indent=2)
                    st.download_button(
                        label="Download All Tables (JSON)",
                        data=json_data,
                        file_name="all_extracted_tables.json",
                        mime="application/json"
                    )
                
            else:
                st.info("No papers have passed screening yet. Complete the Abstract Screening tab first.")
        else:
            st.info("Complete the Abstract Screening tab first to unlock Table Extraction.")
    
    # --- PAGE 6: PRISMA FLOW ---
    elif current_page == "prisma":
        UIComponents.render_prisma_flow()


if __name__ == "__main__":
    main()