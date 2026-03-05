import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
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

def extract_tables_from_paper(paper_title: str, paper_url: str, paper_source: str, paper_id: str, extraction_type: str) -> list:
    """
    Extract tables from paper full text using Europe PMC API and HTML parsing.
    Returns a list of dictionaries with table information.
    """
    tables = []
    
    # Extract paper_id from URL if not provided
    if not paper_id and paper_url:
        # Try to extract PubMed ID from URL
        pmid_match = re.search(r'/(\d+)/?$', paper_url)
        if pmid_match:
            paper_id = pmid_match.group(1)
            print(f"Extracted ID {paper_id} from URL")
    
    print(f"\n=== DEBUG: Extracting tables from: {paper_title} ===")
    print(f"Source: {paper_source}, ID: {paper_id}, URL: {paper_url[:50] if paper_url else 'None'}...")
    
    try:
        # Method 1: Try Europe PMC API for full text (best for PubMed papers)
        if paper_source == "PubMed" or paper_source == "Europe PMC":
            print(f"Attempting Europe PMC extraction for ID: {paper_id}")
            epmc_tables = fetch_epmc_tables(paper_id)
            print(f"Europe PMC returned: {len(epmc_tables)} tables")
            if epmc_tables:
                tables.extend(epmc_tables)
                print(f"Added {len(epmc_tables)} tables from Europe PMC")
            
            # Method 1b: Try PubMed Central (PMC) if EPMC failed
            if not tables:
                print(f"Europe PMC failed, trying PubMed Central (PMC)...")
                pmc_tables = fetch_pmc_tables(paper_id)
                print(f"PubMed Central returned: {len(pmc_tables)} tables")
                if pmc_tables:
                    tables.extend(pmc_tables)
                    print(f"Added {len(pmc_tables)} tables from PubMed Central")
        else:
            print(f"Skipping Europe PMC/PMC - source is {paper_source}")
        
        # Method 2: Try to fetch and parse HTML content
        if not tables and paper_url and paper_url.startswith('http'):
            print(f"Attempting HTML extraction from: {paper_url[:60]}...")
            try:
                response = requests.get(paper_url, timeout=20, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                print(f"HTTP Response: {response.status_code}")
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for HTML tables
                    html_tables = soup.find_all('table')
                    print(f"Found {len(html_tables)} HTML <table> elements")
                    for i, table in enumerate(html_tables):
                        table_data = []
                        rows = table.find_all('tr')
                        
                        for row in rows:
                            row_data = []
                            cells = row.find_all(['td', 'th'])
                            for cell in cells:
                                text = cell.get_text(strip=True)
                                row_data.append(text)
                            
                            if row_data:
                                table_data.append(row_data)
                        
                        if table_data and len(table_data) > 1:
                            tables.append({
                                'title': f"Table {i+1} (HTML)",
                                'type': classify_table_type(table_data, extraction_type),
                                'data': table_data
                            })
                            print(f"  Added HTML Table {i+1}: {len(table_data)} rows")
                else:
                    print(f"HTTP request failed with status: {response.status_code}")
            except Exception as e:
                print(f"HTML extraction failed for {paper_title}: {e}")
        else:
            if tables:
                print(f"Skipping HTML extraction - already have {len(tables)} tables from EPMC")
            elif not paper_url:
                print(f"Skipping HTML extraction - no URL available")
            elif not paper_url.startswith('http'):
                print(f"Skipping HTML extraction - URL doesn't start with http: {paper_url[:30]}...")
    
    except Exception as e:
        print(f"Error extracting tables from {paper_title}: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"=== FINAL RESULT: {len(tables)} tables found ===\n")
    return tables


def fetch_epmc_tables(paper_id: str) -> list:
    """
    Fetch tables from Europe PMC API using the structured full-text endpoint.
    """
    tables = []
    
    print(f"  [EPMC] Starting extraction for paper ID: {paper_id}")
    
    if not paper_id:
        print(f"  [EPMC] ERROR: No paper ID provided")
        return tables
    
    try:
        # Europe PMC API endpoint for full text
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{paper_id}/fullText"
        print(f"  [EPMC] Calling URL: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/xml'
        }
        
        response = requests.get(url, timeout=30, headers=headers)
        print(f"  [EPMC] Response status: {response.status_code}")
        
        if response.status_code == 200:
            content_preview = response.text[:500] if response.text else "EMPTY"
            print(f"  [EPMC] Response content preview: {content_preview}...")
            
            # Parse XML response
            soup = BeautifulSoup(response.content, 'xml')
            
            # Find all table-wrap elements (standard JATS/XML format)
            table_wraps = soup.find_all('table-wrap')
            print(f"  [EPMC] Found {len(table_wraps)} <table-wrap> elements")
            
            for i, table_wrap in enumerate(table_wraps):
                # Get table label/caption
                label = table_wrap.find('label')
                caption = table_wrap.find('caption')
                
                label_text = label.get_text(strip=True) if label else f"Table {i+1}"
                caption_text = caption.get_text(strip=True) if caption else ""
                
                print(f"  [EPMC] Processing table-wrap {i+1}: label='{label_text}'")
                
                # Extract table data
                table_data = []
                
                # Look for thead (header)
                thead = table_wrap.find('thead')
                if thead:
                    header_rows = thead.find_all('tr')
                    print(f"    - thead found with {len(header_rows)} rows")
                    for row in header_rows:
                        row_data = []
                        cells = row.find_all(['th', 'td'])
                        for cell in cells:
                            row_data.append(cell.get_text(strip=True))
                        if row_data:
                            table_data.append(row_data)
                
                # Look for tbody (body)
                tbody = table_wrap.find('tbody')
                if tbody:
                    body_rows = tbody.find_all('tr')
                    print(f"    - tbody found with {len(body_rows)} rows")
                    for row in body_rows:
                        row_data = []
                        cells = row.find_all(['th', 'td'])
                        for cell in cells:
                            row_data.append(cell.get_text(strip=True))
                        if row_data:
                            table_data.append(row_data)
                
                # If no thead/tbody structure, just get all rows
                if not table_data:
                    all_rows = table_wrap.find_all('tr')
                    print(f"    - no thead/tbody, found {len(all_rows)} raw rows")
                    for row in all_rows:
                        row_data = []
                        cells = row.find_all(['th', 'td'])
                        for cell in cells:
                            row_data.append(cell.get_text(strip=True))
                        if row_data:
                            table_data.append(row_data)
                
                print(f"    - Total rows extracted: {len(table_data)}")
                
                if table_data and len(table_data) > 1:
                    tables.append({
                        'title': label_text,
                        'type': classify_table_type(table_data, 'All Tables'),
                        'data': table_data,
                        'caption': caption_text
                    })
                    print(f"    -> Table ADDED successfully")
                else:
                    print(f"    -> Table REJECTED (insufficient data: {len(table_data)} rows)")
            
            # Also try to find simple <table> elements if no table-wrap found
            if not tables:
                simple_tables = soup.find_all('table')
                print(f"  [EPMC] No table-wrap found, trying {len(simple_tables)} simple <table> elements")
                for i, table in enumerate(simple_tables):
                    table_data = []
                    rows = table.find_all('tr')
                    
                    for row in rows:
                        row_data = []
                        cells = row.find_all(['td', 'th'])
                        for cell in cells:
                            row_data.append(cell.get_text(strip=True))
                        if row_data:
                            table_data.append(row_data)
                    
                    if table_data and len(table_data) > 1:
                        tables.append({
                            'title': f"Table {i+1}",
                            'type': classify_table_type(table_data, 'All Tables'),
                            'data': table_data
                        })
                        print(f"    -> Simple table {i+1} ADDED: {len(table_data)} rows")
        else:
            print(f"  [EPMC] ERROR: HTTP {response.status_code} - {response.text[:200]}")
        
    except Exception as e:
        print(f"  [EPMC] EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"  [EPMC] Returning {len(tables)} tables")
    return tables


def fetch_pmc_tables(paper_id: str) -> list:
    """
    Fetch tables from PubMed Central (PMC) using Entrez API.
    Many PubMed papers have free full text in PMC.
    """
    tables = []
    
    print(f"  [PMC] Starting extraction for PMID: {paper_id}")
    
    if not paper_id:
        print(f"  [PMC] ERROR: No paper ID provided")
        return tables
    
    try:
        from Bio import Entrez
        from config import Config
        
        Entrez.email = Config.ENTREZ_EMAIL
        
        # Step 1: Check if paper is in PMC
        print(f"  [PMC] Checking if PMID {paper_id} has PMC article...")
        handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=paper_id)
        link_results = Entrez.read(handle)
        handle.close()
        
        pmc_id = None
        if link_results and len(link_results) > 0:
            linksets = link_results[0].get('LinkSetDb', [])
            for linkset in linksets:
                if linkset.get('DbTo') == 'pmc':
                    links = linkset.get('Link', [])
                    if links:
                        # Extract the PMC ID from the dictionary
                        pmc_id_obj = links[0]
                        if isinstance(pmc_id_obj, dict):
                            pmc_id = pmc_id_obj.get('Id')
                        else:
                            pmc_id = str(pmc_id_obj)
                        print(f"  [PMC] Found PMC ID: {pmc_id}")
                        break
        
        if not pmc_id:
            print(f"  [PMC] No PMC article available for PMID {paper_id}")
            return tables
        
        # Step 2: Fetch the PMC article in XML format
        print(f"  [PMC] Fetching PMC article {pmc_id}...")
        handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="xml", retmode="xml")
        xml_content = handle.read()
        handle.close()
        
        print(f"  [PMC] Got XML content: {len(xml_content)} bytes")
        
        # Step 3: Parse XML to find tables
        soup = BeautifulSoup(xml_content, 'xml')
        
        # Find all table-wrap elements
        table_wraps = soup.find_all('table-wrap')
        print(f"  [PMC] Found {len(table_wraps)} <table-wrap> elements")
        
        for i, table_wrap in enumerate(table_wraps):
            # Get table label/caption
            label = table_wrap.find('label')
            caption = table_wrap.find('caption')
            
            label_text = label.get_text(strip=True) if label else f"Table {i+1}"
            caption_text = caption.get_text(strip=True) if caption else ""
            
            print(f"  [PMC] Processing table-wrap {i+1}: label='{label_text}'")
            
            # Extract table data
            table_data = []
            
            # Look for thead (header)
            thead = table_wrap.find('thead')
            if thead:
                header_rows = thead.find_all('tr')
                print(f"    - thead found with {len(header_rows)} rows")
                for row in header_rows:
                    row_data = []
                    cells = row.find_all(['th', 'td'])
                    for cell in cells:
                        row_data.append(cell.get_text(strip=True))
                    if row_data:
                        table_data.append(row_data)
            
            # Look for tbody (body)
            tbody = table_wrap.find('tbody')
            if tbody:
                body_rows = tbody.find_all('tr')
                print(f"    - tbody found with {len(body_rows)} rows")
                for row in body_rows:
                    row_data = []
                    cells = row.find_all(['th', 'td'])
                    for cell in cells:
                        row_data.append(cell.get_text(strip=True))
                    if row_data:
                        table_data.append(row_data)
            
            # If no thead/tbody structure, just get all rows
            if not table_data:
                all_rows = table_wrap.find_all('tr')
                print(f"    - no thead/tbody, found {len(all_rows)} raw rows")
                for row in all_rows:
                    row_data = []
                    cells = row.find_all(['th', 'td'])
                    for cell in cells:
                        row_data.append(cell.get_text(strip=True))
                    if row_data:
                        table_data.append(row_data)
            
            print(f"    - Total rows extracted: {len(table_data)}")
            
            if table_data and len(table_data) > 1:
                tables.append({
                    'title': label_text,
                    'type': classify_table_type(table_data, 'All Tables'),
                    'data': table_data,
                    'caption': caption_text
                })
                print(f"    -> Table ADDED successfully")
            else:
                print(f"    -> Table REJECTED (insufficient data: {len(table_data)} rows)")
        
    except Exception as e:
        print(f"  [PMC] EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"  [PMC] Returning {len(tables)} tables")
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
    if 'unified_search_query' not in st.session_state:
        st.session_state.unified_search_query = st.session_state.get('query', '')
    if 'prisma_counts' not in st.session_state:
        st.session_state.prisma_counts = {
            'identified': 0, 'duplicates_removed': 0, 
            'screened': 0, 'excluded_total': 0, 'exclusion_breakdown': {}
        }
    
    # Initialize API keys in session state
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = Config.OPENAI_API_KEY
    if 'anthropic_api_key' not in st.session_state:
        st.session_state.anthropic_api_key = Config.ANTHROPIC_API_KEY
    if 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = Config.GEMINI_API_KEY

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
    
    # Clear any running status indicators when switching tabs
    if 'current_page' in st.session_state and st.session_state.get('last_page') != current_page:
        st.session_state.last_page = current_page
        # Force a rerun to clear any lingering status indicators
        st.rerun()
    else:
        st.session_state.last_page = current_page
    
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
                    st.markdown("**Final MeSH Search String**")
                    st.code(entry.get('query', ''), language="sql")
        
        # Refinement Suggestions
        suggestion_to_process = None
        if st.session_state.history:
            last_entry = st.session_state.history[-1]
            suggs = last_entry.get('suggestions', [])
            # if suggs:
            #     st.write("---")
            #     st.caption("✨ **Suggested Question Refinements**")
            #     s_cols = st.columns(len(suggs))
            #     for idx, s in enumerate(suggs):
            #         if s_cols[idx].button(s, key=f"btn_sugg_{len(st.session_state.history)}_{idx}", use_container_width=True):
            #             suggestion_to_process = s
        
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
                    
                    current_inc = "\n".join([f"• {item}" for item in st.session_state.inclusion_list]) if isinstance(st.session_state.inclusion_list, list) and st.session_state.inclusion_list else ""
                    new_inc = st.text_area("Inclusion Criteria (one per line)", value=current_inc, height=100, help="Enter one criteria per line. Press Enter for new bullet point.")
                    # Parse bullet points - handle both • and plain text
                    lines = [line.strip() for line in new_inc.split('\n') if line.strip()]
                    st.session_state.inclusion_list = [line.replace('•', '').strip() for line in lines]
                    
                with p_col2:
                    st.session_state.pico.comparator = st.text_area("Comparator", value=st.session_state.pico.comparator, height=70)
                    st.session_state.pico.outcome = st.text_area("Outcome", value=st.session_state.pico.outcome, height=70)
                    
                    current_excl = "\n".join([f"• {item}" for item in st.session_state.exclusion_list]) if isinstance(st.session_state.exclusion_list, list) and st.session_state.exclusion_list else ""
                    new_excl = st.text_area("Exclusion Criteria (one per line)", value=current_excl, height=100, help="Enter one criteria per line. Press Enter for new bullet point.")
                    # Parse bullet points - handle both • and plain text
                    lines = [line.strip() for line in new_excl.split('\n') if line.strip()]
                    st.session_state.exclusion_list = [line.replace('•', '').strip() for line in lines]
                
                st.session_state.query = st.text_area("Final Search String", value=st.session_state.query, height=100)
        
        # Chat Input (at bottom)
        user_input = st.chat_input("Ask a question or refine your research goal...")
        
        final_input = suggestion_to_process if suggestion_to_process else user_input
        
        if final_input:
            with st.status("🧬 Analyzing Evidence...", expanded=True) as status:
                status.write("Extracting PICO criteria...")
                analysis = AIService.infer_pico_and_query(final_input, model_name, st.session_state.goal)
                
                st.session_state.pico.population = analysis.get('p', '')
                st.session_state.pico.intervention = analysis.get('i', '')
                st.session_state.pico.comparator = analysis.get('c', '')
                st.session_state.pico.outcome = analysis.get('o', '')
                st.session_state.inclusion_list = analysis.get('inclusion', [])
                st.session_state.exclusion_list = analysis.get('exclusion', [])
                
                status.write("Generating formal research question...")
                formal_q = AIService.generate_formal_question(
                    st.session_state.pico, 
                    model_name, 
                    st.session_state.history
                )
                
                status.write("Building search query...")
                mesh_query = analysis.get('query') or AIService.generate_mesh_query(st.session_state.pico, model_name)
                st.session_state.query = mesh_query 
                
                status.write("Searching literature...")
                quick_papers, _ = DataAggregator.fetch_all(mesh_query, active_sources, limit=10)
                
                status.write("Analyzing papers with AI...")
                summary = AIService.generate_comprehensive_summary(final_input, quick_papers, model_name)
                suggs = AIService.get_refinement_suggestions(final_input, quick_papers, model_name)
                
                adversarial_query = AIService.generate_adversarial_query(st.session_state.pico, model_name)
                adversarial_papers, _ = DataAggregator.fetch_all(adversarial_query, active_sources, limit=5)
                
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
                status.update(label="✅ Analysis complete!", state="complete")
                st.rerun()
    
    # --- PAGE 2: SIMULATION ---
    elif current_page == "simulation":
        
        if st.session_state.history:
            # Initialize unified search string if not exists
            if 'unified_search_query' not in st.session_state:
                st.session_state.unified_search_query = st.session_state.query
            
            # Initialize per-database search strings if not exists
            if 'per_db_queries' not in st.session_state:
                st.session_state.per_db_queries = {}
                api_sources = [s for s in active_sources if s not in ["Local PDFs"]]
                for source in api_sources:
                    st.session_state.per_db_queries[source] = st.session_state.query
            

            # Debug: Check what's in unified_search_query
            debug_query = st.session_state.get('unified_search_query', 'NOT_SET')
            
            # Ensure we have a valid query value
            query_value = st.session_state.get('unified_search_query') or st.session_state.get('query', '')
            
            unified_query = st.text_area(
                "Search all databases with:",
                value=query_value,
                height=100,
                key="unified_search_input",
                help="This search string will be used for the 'Run Simulation' button to search all selected databases"
            )
            
            # Update session state when unified query changes
            if unified_query != st.session_state.unified_search_query:
                st.session_state.unified_search_query = unified_query
                # Clear simulation results when unified query changes
                st.session_state.search_simulation = None
            
            st.divider()
            
            # Per-database editing (permanent mode)
            api_sources = [s for s in active_sources if s not in ["Local PDFs"]]
            
            st.caption("Customize search strings for individual databases (overrides unified search):")
            
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
                                    # Use the same method as Run Simulation for consistency
                                    count_result = DataAggregator.simulate_yield(edited_db_query, [source])
                                    
                                    # Check if count_result is valid
                                    if count_result is None:
                                        st.error(f"Simulation returned None for {source}")
                                        return
                                    
                                    actual_count = count_result.get(source, 0)
                                    
                                    # Get top 10 papers for display
                                    papers, _ = DataAggregator.fetch_all(
                                        edited_db_query, 
                                        [source], 
                                        max_per_source=10, 
                                        uploaded_files=[]
                                    )
                                    
                                    # Store results in session state
                                    if 'db_test_results' not in st.session_state or st.session_state.db_test_results is None:
                                        st.session_state.db_test_results = {}
                                    
                                    # Ensure papers is not None before slicing
                                    papers_list = papers[:10] if papers is not None else []
                                    
                                    st.session_state.db_test_results[source] = {
                                        'query': edited_db_query,
                                        'papers': papers_list,  # Top 10 papers for display
                                        'total_found': actual_count  # Use same method as simulation
                                    }
                                    st.success(f"{source}: {actual_count} papers found")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error testing {source}: {str(e)}")
                                    import traceback
                                    traceback.print_exc()
                    
                    if edited_db_query != current_query:
                        st.session_state.per_db_queries[source] = edited_db_query
                        # Don't clear search_simulation when editing individual queries
                        # Only clear it if the unified query changes
                    
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
                if st.button("🤖 AI Optimize Per Source", use_container_width=True, key="ai_optimize_btn"):
                    st.session_state.run_ai_optimize = True
                    st.rerun()
            
            # Run AI Optimization outside columns for full-width status
            if st.session_state.get('run_ai_optimize', False):
                with st.status("🤖 Running agentic optimization...", expanded=True) as status:
                    goal = st.session_state.history[-1].get('goal', '') if st.session_state.history else ''
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Container for iteration history
                    iteration_log = st.empty()
                    iteration_history = []
                    
                    def optimization_progress(iteration, total, source, count, relevance, reasoning):
                        """Track optimization progress - show each iteration once."""
                        progress = iteration / total
                        progress_bar.progress(progress)
                        status_text.text(f"Iteration {iteration}/{total}: Testing {source}...")
                        
                        # Build iteration message
                        if reasoning:
                            msg = f"**Iter {iteration} - {source}:** {count} papers, rel: {relevance:.2f} - {reasoning}"
                        else:
                            msg = f"**Iter {iteration} - {source}:** {count} papers found (rel: {relevance:.2f})"
                        
                        iteration_history.append(msg)
                        # Update the log display with all history - each on separate line
                        iteration_log.markdown("<br>".join(iteration_history), unsafe_allow_html=True)
                    
                    # Use new agentic optimization function with progress callback
                    agentic_result = AIService.agentic_optimize_per_source(
                        st.session_state.unified_search_query,
                        st.session_state.pico,
                        model_name,
                        api_sources,
                        research_goal=goal,
                        progress_callback=optimization_progress
                    )
                    
                    # Store the full trace for display
                    st.session_state.agentic_trace = agentic_result.get('trace', [])
                    st.session_state.agentic_optimization = agentic_result
                    
                    # Update per-database queries with optimized versions
                    per_source_queries = agentic_result.get('per_source_queries', {})
                    for source, query in per_source_queries.items():
                        st.session_state.per_db_queries[source] = query
                    
                    # Set simulation results from final iteration
                    if agentic_result.get('trace'):
                        final_iter = agentic_result['trace'][-1]
                        source_counts = {}
                        for source, data in final_iter.get('sources', {}).items():
                            source_counts[source] = data.get('count', 0)
                        st.session_state.search_simulation = source_counts
                    
                    # Store per-source optimization for display
                    per_source_opt = {}
                    for source in api_sources:
                        query = per_source_queries.get(source, st.session_state.unified_search_query)
                        final_sources = agentic_result['trace'][-1]['sources'] if agentic_result.get('trace') else {}
                        source_data = final_sources.get(source, {})
                        per_source_opt[source] = {
                            'query': query,
                            'yield': source_data.get('count', 0),
                            'relevance': source_data.get('relevance_score', 0),
                            'quality': source_data.get('quality_rating', 'Poor')
                        }
                    st.session_state.per_source_optimization = per_source_opt
                    
                    # Update status with completion
                    status.update(label=f"✅ Optimization complete! {agentic_result.get('iterations_run', 0)} iterations", state="complete")
                
                # Clear the flag
                st.session_state.run_ai_optimize = False
                st.rerun()
            
            with col_sim:
                if st.button("Run Simulation", use_container_width=True, type="primary"):
                    with st.spinner("Calculating yields..."):
                        
                        # Use unified search query for all databases
                        yield_results = {}
                        for source in api_sources:
                            try:
                                source_yield = DataAggregator.simulate_yield(st.session_state.unified_search_query, [source])
                                st.success(f"{source}: {source_yield}")
                                yield_results.update(source_yield)
                            except Exception as e:
                                st.error(f"Error simulating {source}: {e}")
                                yield_results[source] = 0
                        
                        st.session_state.search_simulation = yield_results
                        st.success(f"Simulation complete! Results: {yield_results}")
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
                    st.markdown("### Per-Source Optimized Results")
                    
                    for source, data in st.session_state.per_source_optimization.items():
                        with st.expander(f"**{source}** - {data['yield']:,} papers", expanded=True):
                            # Clean and display query with highlighting
                            raw_query = data['query']
                            base_query = st.session_state.unified_search_query
                            
                            # Strip explanatory text patterns
                            import re
                            cleaned = re.sub(r'(?i)^(here is .*?(?:query|search string).*?for .*?:?\s*)', '', raw_query)
                            cleaned = re.sub(r'(?i)^(optimized query|search string|query):?\s*', '', cleaned)
                            cleaned = re.sub(r'(?i)let me know if you.*$', '', cleaned, flags=re.MULTILINE)
                            cleaned = cleaned.strip()
                            
                            # Highlight new keywords compared to base query
                            if base_query and cleaned != base_query:
                                base_words = set(re.findall(r'\b\w+\b', base_query.lower()))
                                
                                highlighted_parts = []
                                for word in re.findall(r'\S+', cleaned):
                                    word_clean = re.sub(r'[^\w]', '', word.lower())
                                    if word_clean and word_clean not in base_words and len(word_clean) > 2:
                                        highlighted_parts.append(f'<span style="color: #28a745; font-weight: 600;">{word}</span>')
                                    else:
                                        highlighted_parts.append(f'<span style="color: #6c757d;">{word}</span>')
                                
                                query_html = ' '.join(highlighted_parts)
                            else:
                                query_html = f'<span style="color: #6c757d;">{cleaned}</span>'
                            
                            # Display with code-like styling
                            st.markdown(
                                f'<div style="background-color: #f8f9fa; padding: 12px; border-radius: 4px; '
                                f'font-family: monospace; font-size: 0.9em; line-height: 1.5; overflow-x: auto; '
                                f'border: 1px solid #e9ecef;">{query_html}</div>',
                                unsafe_allow_html=True
                            )
                            
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                # Show the modified query vs original
                                if cleaned != base_query:
                                    st.markdown("*Query was optimized for this database*")
                                else:
                                    st.markdown("*Using base query*")
                            with col2:
                                if st.button(f"Use for {source}", key=f"use_{source}"):
                                    st.session_state.unified_search_query = cleaned
                                    # Update per-database queries
                                    api_sources = [s for s in active_sources if s not in ["Local PDFs"]]
                                    for src in api_sources:
                                        st.session_state.per_db_queries[src] = cleaned if src == source else st.session_state.per_db_queries.get(src, st.session_state.query)
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
                    if st.button("Apply This Search String", use_container_width=True, type="primary"):
                        st.session_state.query = current_main_query
                        st.success("Search string updated!")
                        st.rerun()
                
                # Display Agentic Optimization Trace - Grouped by Database
                if st.session_state.get('agentic_trace') and len(st.session_state.agentic_trace) > 0:
                    st.markdown("---")
                    st.markdown("### Agentic Optimization Trace")
                    st.caption("Per-database optimization using PICO framework, Boolean logic, and title relevance analysis")
                    
                    # Summary metrics
                    agentic_result = st.session_state.get('agentic_optimization', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Iterations", agentic_result.get('iterations_run', 0))
                    with col2:
                        st.metric("Total Papers", agentic_result.get('total_papers_found', 0))
                    with col3:
                        st.metric("Best Relevance", f"{agentic_result.get('best_relevance', 0):.2f}")
                    
                    # Group trace data by database
                    db_traces = {}
                    for trace_item in st.session_state.agentic_trace:
                        for source, src_data in trace_item.get('sources', {}).items():
                            if source not in db_traces:
                                db_traces[source] = []
                            db_traces[source].append({
                                'iteration': trace_item.get('iteration', 0),
                                'count': src_data.get('count', 0),
                                'relevance': src_data.get('relevance_score', 0),
                                'quality': src_data.get('quality_rating', 'N/A'),
                                'query': src_data.get('query', ''),
                                'titles': src_data.get('titles', []),
                                'reasoning': src_data.get('iteration_reasoning', '')
                            })
                    
                    # Show per-database optimization history
                    st.markdown("**Optimization History by Database:**")
                    
                    for source, trace_history in db_traces.items():
                        # Calculate final metrics for this database
                        final_iter = trace_history[-1]
                        total_iters = len(trace_history)
                        final_count = final_iter['count']
                        final_relevance = final_iter['relevance']
                        quality = final_iter['quality']
                        
                        with st.expander(f"{source}: {final_count} papers, {final_relevance:.2f} relevance ({quality}) - {total_iters} iterations", expanded=False):
                            
                            # Show iteration history for this database
                            prev_query = None
                            for i, step in enumerate(trace_history):
                                is_last = (i == len(trace_history) - 1)
                                step_icon = "✅" if is_last else f"{i+1}"
                                
                                with st.container():
                                    st.markdown(f"**{step_icon} Iteration {step['iteration']}:** {step['count']} papers, relevance {step['relevance']:.2f} ({step['quality']})")
                                    
                                    # Show reasoning
                                    if step['reasoning']:
                                        st.markdown(f"<small> {step['reasoning']}</small>", unsafe_allow_html=True)
                                    
                                    # Clean and display query with highlighting
                                    raw_query = step['query']
                                    
                                    # Strip explanatory text patterns
                                    import re
                                    # Remove lines like "Here is the optimized query..." or "Query:"
                                    cleaned = re.sub(r'(?i)^(here is .*?(?:query|search string).*?for .*?:?\s*)', '', raw_query)
                                    cleaned = re.sub(r'(?i)^(optimized query|search string|query):?\s*', '', cleaned)
                                    cleaned = re.sub(r'(?i)let me know if you.*$', '', cleaned, flags=re.MULTILINE)
                                    cleaned = cleaned.strip()
                                    
                                    # Highlight new keywords compared to previous iteration
                                    if prev_query:
                                        # Extract word tokens from both queries
                                        prev_words = set(re.findall(r'\b\w+\b', prev_query.lower()))
                                        curr_words = re.findall(r'\b\w+\b', cleaned.lower())
                                        
                                        # Build highlighted HTML
                                        highlighted_parts = []
                                        for word in re.findall(r'\S+', cleaned):
                                            word_clean = re.sub(r'[^\w]', '', word.lower())
                                            if word_clean and word_clean not in prev_words and len(word_clean) > 2:
                                                # New word - highlight in green
                                                highlighted_parts.append(f'<span style="color: #28a745; font-weight: 600;">{word}</span>')
                                            else:
                                                # Existing word - grey color
                                                highlighted_parts.append(f'<span style="color: #6c757d;">{word}</span>')
                                        
                                        query_html = ' '.join(highlighted_parts)
                                    else:
                                        # First iteration - all grey
                                        query_html = f'<span style="color: #6c757d;">{cleaned}</span>'
                                    
                                    # Display with code-like styling
                                    st.markdown(
                                        f'<div style="background-color: #f8f9fa; padding: 12px; border-radius: 4px; '
                                        f'font-family: monospace; font-size: 0.9em; line-height: 1.5; overflow-x: auto; '
                                        f'border: 1px solid #e9ecef;">{query_html}</div>',
                                        unsafe_allow_html=True
                                    )
                                    
                                    prev_query = cleaned
                                    
                                    # Show sample titles found (using checkbox instead of nested expander)
                                    if step['titles']:
                                        show_titles_key = f"show_titles_{source}_{step['iteration']}"
                                        if st.checkbox(f"Show Sample Titles ({len(step['titles'])} found)", key=show_titles_key):
                                            for title in step['titles'][:10]:
                                                st.markdown(f"• {title}")
                                    
                                    if not is_last:
                                        st.divider()
                    
                    # Show overall comparison table
                    st.markdown("---")
                    st.markdown("**Final Comparison:**")
                    comparison_rows = []
                    for source, trace_history in db_traces.items():
                        final = trace_history[-1]
                        comparison_rows.append({
                            "Database": source,
                            "Papers": final['count'],
                            "Relevance": f"{final['relevance']:.2f}",
                            "Quality": final['quality'],
                            "Iterations": len(trace_history),
                            "Final Query": final['query'][:80] + "..." if len(final['query']) > 80 else final['query']
                        })
                    
                    if comparison_rows:
                        st.dataframe(
                            pd.DataFrame(comparison_rows),
                            column_config={
                                "Database": st.column_config.TextColumn(width="small"),
                                "Papers": st.column_config.NumberColumn(width="small"),
                                "Relevance": st.column_config.TextColumn(width="small"),
                                "Quality": st.column_config.TextColumn(width="small"),
                                "Iterations": st.column_config.NumberColumn(width="small"),
                                "Final Query": st.column_config.TextColumn(width="large")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
    # --- PAGE 3: ABSTRACT SCREENING ---
    elif current_page == "abstract":
        
        # Run Search Button
        if st.session_state.history:
            if st.button("Run Database Search", type="primary", use_container_width=True, key="run_search_tab"):
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
                    
                    # 3. MULTI-AGENT SCREENING ARCHITECTURE
                    from utils import ScreeningOrchestrator
                    
                    # Prepare API keys for thread-safe model initialization
                    api_keys = {
                        'openai': st.session_state.get('openai_api_key', ''),
                        'anthropic': st.session_state.get('anthropic_api_key', ''),
                        'gemini': st.session_state.get('gemini_api_key', '')
                    }
                    
                    # ORCHESTRATOR: Initialize with decomposition
                    orchestrator = ScreeningOrchestrator(
                        pico=st.session_state.pico,
                        inclusion_criteria=st.session_state.get('inclusion_list', []),
                        exclusion_criteria=st.session_state.get('exclusion_list', []),
                        model_name=model_name,
                        api_keys=api_keys
                    )
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def agent_progress(current, total, message=""):
                        """Track agent completion progress."""
                        progress_bar.progress(current / total)
                        status_text.text(message)
                    
                    # ORCHESTRATOR: Execute fan-out screening
                    # Each specialist agent evaluates ALL papers for their ONE criterion
                    screened_results = orchestrator.screen_papers(
                        papers=unique,
                        progress_callback=agent_progress
                    )
                    
                    # AGGREGATOR: Process results with traceability
                    screened = []
                    reasons = {}
                    
                    for result in screened_results:
                        row = {
                            "paper_id": result['paper_id'],
                            "Source": result['Source'],
                            "Title": result['Title'],
                            "URL": result['URL'],
                            "Decision": result['Decision'],
                            "Reason": result['Reason'],
                            "Abstract": result['Abstract'],
                            "PICO_Score": result['PICO_Score'],
                            "Agent_Trace": result['Agent_Trace']  # Full audit trail
                        }
                        screened.append(row)
                        
                        # Track exclusion reasons for PRISMA
                        if result['Decision'] == "EXCLUDE":
                            reason_key = result['Reason'][:50]
                            reasons[reason_key] = reasons.get(reason_key, 0) + 1
                    
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
        
        # Display Results with Traceability
        if st.session_state.results is not None and not st.session_state.results.empty:
            
            # Show agent scores summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Papers Screened", len(st.session_state.results))
            with col2:
                passed = st.session_state.results[st.session_state.results['Decision'].str.contains("Include", case=False, na=False)]
                st.metric("Included", len(passed))
            with col3:
                excluded = st.session_state.results[st.session_state.results['Decision'].str.contains("Exclude", case=False, na=False)]
                st.metric("Excluded", len(excluded))
            
            # Sample papers with agent trace
            st.markdown("### Screening Results")
            
            # Prepare results with expanded agent trace columns
            display_results = st.session_state.results.copy()
            
            # Extract individual agent votes into separate columns
            agent_columns = {}
            for idx, row in display_results.iterrows():
                trace = row.get('Agent_Trace', {})
                if not isinstance(trace, dict):
                    continue  # Skip rows with NaN or non-dict trace
                for agent_name, vote_info in trace.items():
                    if agent_name not in agent_columns:
                        agent_columns[agent_name] = []
                    vote = vote_info.get('vote', 'N/A')
                    conf = vote_info.get('confidence', 0)
                    agent_columns[agent_name].append(f"{vote} ({conf:.1f})")
            
            # Add agent columns to dataframe
            for agent_name, votes in agent_columns.items():
                # Pad with N/A if missing
                while len(votes) < len(display_results):
                    votes.append("N/A")
                display_results[agent_name] = votes
            
            # Define color styling functions
            def color_decision(val):
                val_str = str(val).upper()
                if 'INCLUDE' in val_str or 'PASS' in val_str:
                    return 'background-color: #d4edda; color: #155724; font-weight: 500;'
                elif 'EXCLUDE' in val_str or 'FAIL' in val_str or 'EXCLUDE' in str(val):
                    return 'background-color: #f8d7da; color: #721c24; font-weight: 500;'
                return ''
            
            def color_agent_vote(val):
                val_str = str(val).upper()
                if 'PASS' in val_str:
                    return 'background-color: #d4edda; color: #155724; font-weight: 500;'
                elif 'FAIL' in val_str:
                    return 'background-color: #f8d7da; color: #721c24; font-weight: 500;'
                return ''
            
            # Apply styling
            styled_results = display_results.style
            
            # Style Decision column
            if 'Decision' in display_results.columns:
                styled_results = styled_results.map(color_decision, subset=['Decision'])
            
            # Style all agent vote columns
            for agent_name in agent_columns.keys():
                if agent_name in display_results.columns:
                    styled_results = styled_results.map(color_agent_vote, subset=[agent_name])
            
            # Show styled results table
            st.dataframe(
                styled_results,
                column_config={
                    "URL": st.column_config.LinkColumn("Source Link", display_text="View Paper", width="small"),
                    "Title": st.column_config.TextColumn(width="large"),
                    "Abstract": st.column_config.TextColumn(width="large"),
                    "Decision": st.column_config.TextColumn(width="small"),
                    "PICO_Score": st.column_config.TextColumn(width="small"),
                },
                hide_index=True,
                use_container_width=True,
                height=600
            )
            
            passed = st.session_state.results[st.session_state.results['Decision'].str.contains("Include", case=False, na=False)]
            if not passed.empty:
                st.info(f" {len(passed)} papers passed abstract screening and are ready for Full-Text Extraction.")
        elif st.session_state.results is not None and st.session_state.results.empty:
            st.warning("No papers found. Try adjusting your query or criteria.")
        else:
            st.info("Click 'Run Database Search' to start screening papers.")
    
    # --- PAGE 4: FULL-TEXT EVIDENCE ---
    elif current_page == "fulltext":
        
        if st.session_state.results is not None:
            # Use case-insensitive matching to find included papers
            passed = st.session_state.results[st.session_state.results['Decision'].str.contains("Include", case=False, na=False)]
            
            if not passed.empty:
                if 'full_text_results' not in st.session_state:
                    st.info(f"🎯 {len(passed)} papers ready for full-text analysis.")
                    if st.button("Begin Full-Text Screening", type="primary", use_container_width=True):
                        with st.status("Performing Full-Text Analysis...", expanded=True) as status:
                            final_rows = []
                            ft_reasons = {} 
                            
                            # Create progress bar
                            total_papers = len(passed)
                            progress_bar = st.progress(0, text=f"Screening 0/{total_papers} papers")
                            current_paper = st.empty()

                            # 4. MULTI-AGENT FULL-TEXT SCREENING
                            from utils import FullTextOrchestrator
                            
                            # Prepare API keys for thread-safe model initialization
                            api_keys = {
                                'openai': st.session_state.get('openai_api_key', ''),
                                'anthropic': st.session_state.get('anthropic_api_key', ''),
                                'gemini': st.session_state.get('gemini_api_key', '')
                            }
                            
                            # Initialize FullTextOrchestrator with inclusion/exclusion agents
                            ft_orchestrator = FullTextOrchestrator(
                                inclusion_criteria=st.session_state.get('inclusion_list', []),
                                exclusion_criteria=st.session_state.get('exclusion_list', []),
                                model_name=st.session_state.custom_model,
                                api_keys=api_keys
                            )
                            
                            for idx, (_, row) in enumerate(passed.iterrows()):
                                # Update current paper display
                                current_paper.info(f"📄 Currently screening: {row.get('Title', 'Unknown Paper')[:100]}...")
                                
                                # Update progress bar
                                progress = (idx + 1) / total_papers
                                progress_bar.progress(progress, text=f"Screening {idx + 1}/{total_papers} papers")
                                
                                # FAN-OUT: All criterion agents evaluate this paper in parallel
                                res = ft_orchestrator.screen_paper(row.to_dict())
                                
                                is_included = "Include" in str(res.get('decision', ''))
                                entry = row.to_dict()
                                entry['Decision'] = "Include" if is_included else "Exclude"
                                entry['Reason'] = res.get('reason', 'N/A')
                                entry['Abstract'] = row.get('Abstract', 'N/A')
                                
                                # Add ALL inclusion and exclusion criteria evaluations from agent trace
                                all_criteria = st.session_state.get('inclusion_list', []) + st.session_state.get('exclusion_list', [])
                                for criterion in all_criteria:
                                    evaluation = res.get(criterion, 'EXCLUDE')
                                    if evaluation not in ['INCLUDE', 'EXCLUDE']:
                                        evaluation = 'EXCLUDE'
                                    entry[criterion] = evaluation
                                
                                # Add agent trace for traceability
                                entry['Agent_Trace'] = res.get('agent_trace', {})
                                entry['Inclusion_Score'] = res.get('inclusion_score', 'N/A')
                                entry['Exclusion_Violations'] = res.get('exclusion_violations', 'N/A')
                                
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
                        val_str = str(val).upper().strip()
                        if 'INCLUDE' in val_str or 'Include' in str(val):
                            return 'background-color: #d4edda; color: #155724; font-weight: 500; padding: 8px;'
                        elif 'EXCLUDE' in val_str or 'Exclude' in str(val):
                            return 'background-color: #f8d7da; color: #721c24; font-weight: 500; padding: 8px;'
                        else:
                            return 'background-color: #e2e3e5; color: #383d41; font-weight: 500; padding: 8px;'
                    
                    # Apply styling to Decision column if it exists
                    styled_ft_results = st.session_state.full_text_results.copy()
                    if 'Decision' in styled_ft_results.columns:
                        styled_ft_results = styled_ft_results.style.map(color_decisions, subset=['Decision'])
                    
                    # Apply styling to ALL inclusion and exclusion criteria columns
                    inclusion_criteria = st.session_state.get('inclusion_list', [])
                    exclusion_criteria = st.session_state.get('exclusion_list', [])
                    all_criteria = inclusion_criteria + exclusion_criteria
                    
                    # Also apply to any PICO columns that might exist
                    pico_criteria = []
                    if hasattr(st.session_state, 'pico'):
                        if st.session_state.pico.population:
                            pico_criteria.append(st.session_state.pico.population)
                        if st.session_state.pico.intervention:
                            pico_criteria.append(st.session_state.pico.intervention)
                        if st.session_state.pico.comparator:
                            pico_criteria.append(st.session_state.pico.comparator)
                        if st.session_state.pico.outcome:
                            pico_criteria.append(st.session_state.pico.outcome)
                    
                    # Combine all criteria
                    all_columns_to_style = list(set(all_criteria + pico_criteria))
                    
                    for criterion in all_columns_to_style:
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
    
    # --- PAGE 5: CITATION SNOWBALLING ---
    elif current_page == "citation_snowball":
        
        # Check if full-text screening is complete
        if 'full_text_results' not in st.session_state or st.session_state.full_text_results is None:
            st.info("Complete Full-Text Evidence screening first to unlock Citation Snowballing.")
        else:
            # Get papers that passed full-text screening
            included_papers = st.session_state.full_text_results[st.session_state.full_text_results['Decision'].str.contains("Include", case=False, na=False)]
            
            if included_papers.empty:
                st.warning("No papers passed full-text screening. Cannot perform citation snowballing.")
            else:
                st.markdown("### Citation Snowballing")
                
                # Show count of seed papers
                st.info(f"**Seed Papers:** {len(included_papers)} papers passed full-text screening and will be used for snowballing.")
                
                # Initialize snowball results if not exists
                if 'snowball_results' not in st.session_state:
                    st.session_state.snowball_results = None
                if 'snowball_screened' not in st.session_state:
                    st.session_state.snowball_screened = None
                
                # Snowballing controls
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    snowball_type = st.selectbox(
                        "Snowball Type:",
                        ["Both", "Backward (References)", "Forward (Cited by)"],
                        help="Backward: Papers cited by seed papers. Forward: Papers citing seed papers."
                    )
                
                with col2:
                    max_citations = st.slider("Max per paper", 5, 50, 20, 
                        help="Maximum citations to fetch per seed paper")
                
                with col3:
                    st.write("")
                    st.write("")
                    start_snowball = st.button("Start Snowballing", type="primary", use_container_width=True)
                
                if start_snowball:
                    with st.status("Fetching citations from databases...", expanded=True) as status:
                        snowball_papers = []
                        
                        for idx, (_, paper) in enumerate(included_papers.iterrows()):
                            status.update(label=f"Processing {idx+1}/{len(included_papers)}: {paper['Title'][:50]}...")
                            
                            # Try to fetch citations based on paper ID/source
                            citations = AIService.fetch_citations(
                                paper.get('ID', ''),
                                paper.get('Source', ''),
                                paper.get('Title', ''),
                                snowball_type,
                                max_citations,
                                active_sources
                            )
                            
                            for cit in citations:
                                cit['seed_paper_id'] = paper.get('ID', '')
                                cit['seed_paper_title'] = paper.get('Title', '')
                                cit['discovery_method'] = 'citation_snowball'
                            
                            snowball_papers.extend(citations)
                        
                        # Remove duplicates
                        seen_titles = set()
                        unique_snowball = []
                        for p in snowball_papers:
                            if p['title'].lower() not in seen_titles:
                                seen_titles.add(p['title'].lower())
                                unique_snowball.append(p)
                        
                        st.session_state.snowball_results = unique_snowball
                        status.update(label=f"✅ Found {len(unique_snowball)} unique papers via citation snowballing!", state="complete")
                    st.rerun()
                
                # Display snowball results
                if st.session_state.snowball_results:
                    st.markdown("---")                    
                    # Convert to DataFrame for display
                    snowball_df = pd.DataFrame(st.session_state.snowball_results)
                    
                    # Reorder and select columns for display
                    display_cols = ['title', 'abstract', 'source', 'seed_paper_title', 'citation_type', 'url', 'id']
                    available_cols = [c for c in display_cols if c in snowball_df.columns]
                    snowball_display = snowball_df[available_cols].copy()
                    
                    # Rename columns for better display
                    col_rename = {
                        'title': 'Title',
                        'abstract': 'Abstract',
                        'source': 'Source Database',
                        'seed_paper_title': 'Seed Paper',
                        'citation_type': 'Citation Type',
                        'url': 'URL',
                        'id': 'Paper ID'
                    }
                    snowball_display.rename(columns=col_rename, inplace=True)
                    
                    # Show table with configuration
                    st.markdown("#### Discovered Papers Table")
                    st.dataframe(
                        snowball_display,
                        column_config={
                            "Title": st.column_config.TextColumn(width="large"),
                            "Abstract": st.column_config.TextColumn(width="large"),
                            "Source Database": st.column_config.TextColumn(width="medium"),
                            "Seed Paper": st.column_config.TextColumn(width="large"),
                            "Citation Type": st.column_config.TextColumn(width="small"),
                            "URL": st.column_config.LinkColumn("View Paper", display_text="Open", width="small"),
                            "Paper ID": st.column_config.TextColumn(width="small")
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Show summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Found", len(snowball_df))
                    with col2:
                        if 'citation_type' in snowball_df.columns:
                            backward_count = len(snowball_df[snowball_df['citation_type'] == 'backward'])
                            st.metric("References (Backward)", backward_count)
                    with col3:
                        if 'citation_type' in snowball_df.columns:
                            forward_count = len(snowball_df[snowball_df['citation_type'] == 'forward'])
                            st.metric("Cited By (Forward)", forward_count)
                    
                    # Check if already screened
                    if st.session_state.snowball_screened is None:
                        st.info("These papers need to be screened with your inclusion/exclusion criteria.")
                        
                        if st.button("🧬 Screen Snowballed Papers", type="primary", use_container_width=True):
                            with st.status("AI-screening snowballed papers...", expanded=True) as status:
                                screened_rows = []
                                
                                for idx, paper in enumerate(st.session_state.snowball_results):
                                    status.update(label=f"Screening {idx+1}/{len(st.session_state.snowball_results)}...")
                                    
                                    # Create Paper object for screening
                                    from models import Paper
                                    p = Paper(
                                        id=paper.get('id', f'snowball_{idx}'),
                                        title=paper.get('title', 'Unknown'),
                                        abstract=paper.get('abstract', ''),
                                        url=paper.get('url', ''),
                                        source=paper.get('source', 'Snowball'),
                                        score=0.5
                                    )
                                    
                                    try:
                                        res = AIService.screen_paper(
                                            p, 
                                            st.session_state.pico, 
                                            model_name, 
                                            st.session_state.inclusion_list, 
                                            st.session_state.exclusion_list
                                        )
                                        
                                        screened_rows.append({
                                            "Source": paper.get('source', 'Snowball'),
                                            "Title": paper.get('title', 'Unknown'),
                                            "URL": paper.get('url', ''),
                                            "Decision": res.get('decision', 'Exclude'),
                                            "Reason": res.get('reason', 'N/A'),
                                            "Abstract": paper.get('abstract', ''),
                                            "Seed Paper": paper.get('seed_paper_title', 'Unknown'),
                                            "Discovery": paper.get('discovery_method', 'citation_snowball')
                                        })
                                    except Exception as e:
                                        screened_rows.append({
                                            "Source": paper.get('source', 'Snowball'),
                                            "Title": paper.get('title', 'Unknown'),
                                            "URL": paper.get('url', ''),
                                            "Decision": "Exclude",
                                            "Reason": f"Screening error: {str(e)[:50]}",
                                            "Abstract": paper.get('abstract', ''),
                                            "Seed Paper": paper.get('seed_paper_title', 'Unknown'),
                                            "Discovery": paper.get('discovery_method', 'citation_snowball')
                                        })
                                
                                st.session_state.snowball_screened = pd.DataFrame(screened_rows)
                                status.update(label=f"✅ Screening complete!", state="complete")
                            st.rerun()
                    else:
                        # Show screened results
                        snowball_passed = st.session_state.snowball_screened[st.session_state.snowball_screened['Decision'].str.contains("Include", case=False, na=False)]
                        
                        st.success(f"✅ {len(snowball_passed)} of {len(st.session_state.snowball_screened)} snowballed papers passed screening")
                        
                        UIComponents.render_results(st.session_state.snowball_screened)
                        
                        if not snowball_passed.empty:
                            st.info(f"🎯 {len(snowball_passed)} additional papers from citation snowballing passed screening!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("➕ Add to Main Results", use_container_width=True, type="primary"):
                                    # Add snowballed papers to main results
                                    combined = pd.concat([st.session_state.results, snowball_passed], ignore_index=True)
                                    st.session_state.results = combined
                                    
                                    # Update PRISMA counts
                                    st.session_state.prisma_counts['identified'] += len(st.session_state.snowball_results)
                                    st.session_state.prisma_counts['included_final'] += len(snowball_passed)
                                    
                                    st.success(f"Added {len(snowball_passed)} papers to main results!")
                                    st.rerun()
                            
                            with col2:
                                if st.button("🗑️ Clear Snowball Results", use_container_width=True):
                                    st.session_state.snowball_results = None
                                    st.session_state.snowball_screened = None
                                    st.rerun()

    # --- PAGE 6: TEXT EXTRACTION ---
    elif current_page == "extraction":
        

        if st.session_state.results is not None:
            passed = st.session_state.results[st.session_state.results['Decision'].str.contains("Include", case=False, na=False)]
            
            if not passed.empty:
                st.info(f"🎯 {len(passed)} papers available for table extraction.")
                
                # Table extraction options
                st.markdown("### Extraction Settings")
                
                output_format = st.selectbox(
                    "Output Format:",
                    ["DataFrame", "CSV Export", "JSON Export", "Excel Export"]
                )
                
                if st.button("Start Table Extraction", type="primary", use_container_width=True):
                    with st.status("Extracting tables from papers...", expanded=True) as status:
                        extracted_data = []
                        
                        for idx, (_, row) in enumerate(passed.iterrows()):
                            paper_title = row.get('Title', 'Unknown Paper')
                            paper_url = row.get('URL', '')
                            paper_source = row.get('Source', 'Unknown')
                            paper_id = row.get('ID', '')
                            
                            # Extract all tables using EPMC and other sources
                            extracted_tables = extract_tables_from_paper(paper_title, paper_url, paper_source, paper_id, "All Tables")
                            
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
                if 'extracted_papers_data' in st.session_state and st.session_state.extracted_papers_data:
                    
                    # Display each paper's tables
                    for paper_data in st.session_state.extracted_papers_data:
                        with st.expander(f"📄 {paper_data['Paper_Title']}", expanded=False):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Source:** {paper_data['Source']}")
                                if paper_data['Paper_URL']:
                                    # Red styled button matching primary button color
                                    st.markdown(f"""
                                        <a href="{paper_data['Paper_URL']}" target="_blank" style="
                                            display: block;
                                            background-color: #FF4444;
                                            color: white;
                                            text-align: center;
                                            padding: 10px 20px;
                                            border-radius: 8px;
                                            text-decoration: none;
                                            font-weight: 600;
                                            margin-top: 10px;
                                        ">📖 View Full Paper</a>
                                    """, unsafe_allow_html=True)
                            
                            with col2:
                                st.metric("Tables Found", len(paper_data['Extracted_Tables']))
                            
                            # Display extracted tables
                            if paper_data['Extracted_Tables']:
                                st.markdown("### Extracted Tables")
                                
                                # Use tabs instead of nested expanders - improve naming
                                tab_labels = []
                                for i, table in enumerate(paper_data['Extracted_Tables']):
                                    title = table.get('title', f'Table {i+1}')
                                    table_type = table.get('type', 'General')
                                    # If title is generic like "Table 1", "Table 2", use type instead
                                    if title.startswith('Table ') and title.replace('Table ', '').isdigit():
                                        tab_labels.append(f"{table_type} {i+1}")
                                    else:
                                        tab_labels.append(f"{table_type}: {title}")
                                
                                table_tabs = st.tabs(tab_labels)
                                
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
                                            # Create unique key using hash of full title to avoid collisions
                                            import hashlib
                                            title_hash = hashlib.md5(paper_data['Paper_Title'].encode()).hexdigest()[:8]
                                            if st.button(f"📥 Export Table {i+1}", key=f"export_table_{title_hash}_{i}"):
                                                csv_data = df.to_csv(index=False)
                                                st.download_button(
                                                    label=f"Download Table {i+1} CSV",
                                                    data=csv_data,
                                                    file_name=f"table_{i+1}_{paper_data['Paper_Title'][:30].replace(' ', '_')}.csv",
                                                    mime="text/csv"
                                                )
                            else:
                                st.info("No tables found in this paper.")
                
                    # Dynamic export button based on selected output format
                    export_label = f"Export All Tables ({output_format.split()[0]})"
                    
                    if st.button(export_label, use_container_width=True):
                        if "JSON" in output_format:
                            import json
                            json_data = json.dumps(st.session_state.extracted_papers_data, indent=2)
                            st.download_button(
                                label="Download All Tables (JSON)",
                                data=json_data,
                                file_name="all_extracted_tables.json",
                                mime="application/json"
                            )
                        elif "CSV" in output_format:
                            import csv
                            import io
                            csv_buffer = io.StringIO()
                            writer = csv.writer(csv_buffer)
                            for paper_data in st.session_state.extracted_papers_data:
                                writer.writerow([paper_data['Paper_Title'], paper_data['Source']])
                            st.download_button(
                                label="Download Summary (CSV)",
                                data=csv_buffer.getvalue(),
                                file_name="all_extracted_tables_summary.csv",
                                mime="text/csv"
                            )
                        elif "Excel" in output_format:
                            st.info("Excel export would be implemented here with openpyxl")
                        else:
                            st.info("DataFrame view - export functionality available per table above")
                
    
        else:
            st.info("Complete the Abstract Screening tab first to unlock Table Extraction.")
    
    # --- PAGE 6: PRISMA FLOW ---
    elif current_page == "prisma":
        UIComponents.render_prisma_flow()


if __name__ == "__main__":
    main()