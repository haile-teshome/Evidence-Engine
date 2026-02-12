from Bio import Entrez
import requests
import xml.etree.ElementTree as ET
import urllib.parse
from pypdf import PdfReader
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import streamlit as st
from config import Config, DataSource
from models import Paper
from utils import QueryCleaner
import time

_last_request_time = 0.0

def throttled_request(url: str, params: dict = None, headers: dict = None, method: str = "GET") -> requests.Response:
    """Ensures all outgoing requests respect a 1-request-per-second limit."""
    global _last_request_time
    
    elapsed = time.time() - _last_request_time
    if elapsed < 1.1:  
        time.sleep(1.1 - elapsed)
    
    if method.upper() == "POST":
        response = requests.post(url, json=params, headers=headers, timeout=15)
    else:
        response = requests.get(url, params=params, headers=headers, timeout=15)
    
    _last_request_time = time.time()
    return response

class PubMedService:
    """Handles PubMed data fetching."""
    
    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        """Fetch papers from PubMed."""
        Entrez.email = Config.ENTREZ_EMAIL
        
        # Add title/abstract search if not specified
        if "[tiab]" not in query.lower() and "[" not in query:
            query = f"({query})[tiab]"
        
        try:
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results
            )
            id_list = Entrez.read(search_handle)["IdList"]
            
            if not id_list:
                return []
            
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=id_list,
                rettype="xml",
                retmode="text"
            )
            records = Entrez.read(fetch_handle)
            
            papers = []
            for article in records['PubmedArticle']:
                citation = article['MedlineCitation']
                pmid = str(citation['PMID']) 
                
                abstract_text = citation['Article'].get('Abstract', {}).get(
                    'AbstractText', ["N/A"]
                )[0]
                
                papers.append(Paper(
                    source=DataSource.PUBMED.value,
                    id=pmid,
                    title=citation['Article']['ArticleTitle'],
                    abstract=str(abstract_text),
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                ))
            
            return papers
            
        except Exception as e:
            st.error(f"PubMed fetch error: {e}")
            return []




class TopJournalsService:
    """Fetches from top epidemiology journals."""
    
    JOURNALS = [
        '"Am J Epidemiol"[Journal]',
        '"Int J Epidemiol"[Journal]',
        '"Eur J Epidemiol"[Journal]'
    ]
    
    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        """Fetch from AJE, IJE, and EJE."""
        Entrez.email = Config.ENTREZ_EMAIL
        
        journal_filter = ' OR '.join(TopJournalsService.JOURNALS)
        full_query = f"({query}) AND ({journal_filter})"
        
        try:
            search_handle = Entrez.esearch(
                db="pubmed",
                term=full_query,
                retmax=max_results
            )
            id_list = Entrez.read(search_handle)["IdList"]
            
            if not id_list:
                return []
            
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=id_list,
                rettype="xml",
                retmode="text"
            )
            records = Entrez.read(fetch_handle)
            
            papers = []
            for article in records['PubmedArticle']:
                citation = article['MedlineCitation']
                abstract_text = citation['Article'].get('Abstract', {}).get(
                    'AbstractText', ["N/A"]
                )[0]
                
                papers.append(Paper(
                    source=DataSource.BIG3_JOURNALS.value,
                    id=str(citation['PMID']),
                    title=citation['Article']['ArticleTitle'],
                    abstract=str(abstract_text)
                ))
            
            return papers
            
        except Exception as e:
            st.error(f"Top journals fetch error: {e}")
            return []


class ArXivService:
    """Handles arXiv data fetching."""
    
    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        """Fetch papers from arXiv."""
        clean_query = QueryCleaner.clean_for_general_search(query)
        params = {
            'search_query': f'all:{clean_query}',
            'start': 0,
            'max_results': max_results
        }
        
        try:
            response = throttled_request(Config.ARXIV_API_URL, params=params)
            root = ET.fromstring(response.content)
            
            papers = []
            ns = {'ns': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('ns:entry', ns):
                full_id = entry.find('ns:id', ns).text
                paper_id = full_id.split('/')[-1]
                
                papers.append(Paper(
                    source=DataSource.ARXIV.value,
                    id=paper_id,
                    title=entry.find('ns:title', ns).text.strip().replace('\n', ' '),
                    abstract=entry.find('ns:summary', ns).text.strip(),
                    url=f"https://arxiv.org/abs/{paper_id}"
                ))
            return papers
        except Exception as e:
            st.error(f"ArXiv fetch error: {e}")
            return []

class BioRxivService:
    """Handles BioRxiv data fetching."""
    
    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        """Fetch papers from BioRxiv (recent papers only)."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=Config.BIORXIV_LOOKBACK_DAYS)
        
        date_str = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        url = f"{Config.BIORXIV_API_URL}/{date_str}"
        
        try:
            response = throttled_request(url)
            data = response.json()
            
            papers = []
            keywords = [k.lower() for k in query.split() if len(k) > 2]
            
            for preprint in data.get('collection', []):
                text_to_search = (preprint['title'] + " " + preprint['abstract']).lower()
                
                if any(k in text_to_search for k in keywords):
                    papers.append(Paper(
                        source=DataSource.BIORXIV.value,
                        id=preprint.get('doi', 'N/A'),
                        title=preprint['title'],
                        abstract=preprint['abstract'],
                        url=f"https://doi.org/{preprint['doi']}"
                    ))
                
                if len(papers) >= max_results:
                    break
                    
            return papers
        except Exception as e:
            st.error(f"BioRxiv fetch error: {e}")
            return []


class PDFService:
    """Handles local PDF processing."""
    
    @staticmethod
    def process_files(files) -> List[Paper]:
        """Extract text from uploaded PDF files."""
        papers = []
        
        for file in files:
            try:
                reader = PdfReader(file)
                text_parts = []
                
                for page in reader.pages[:Config.PDF_MAX_PAGES]:
                    text_parts.append(page.extract_text())
                
                full_text = "".join(text_parts)
                truncated = full_text[:Config.PDF_MAX_CHARS]
                
                papers.append(Paper(
                    source=DataSource.LOCAL_PDF.value,
                    id=file.name,
                    title=file.name,
                    abstract=truncated
                ))
                
            except Exception as e:
                st.warning(f"Failed to process {file.name}: {e}")
                continue
        
        return papers


class SemanticScholarService:
    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,paperId"
        }
        headers = {"x-api-key": Config.SEMANTIC_SCHOLAR_API_KEY} if hasattr(Config, 'SEMANTIC_SCHOLAR_API_KEY') else {}
        
        try:
            url = Config.SEMANTIC_SCHOLAR_API_URL if hasattr(Config, 'SEMANTIC_SCHOLAR_API_URL') else "https://api.semanticscholar.org/graph/v1/paper/search"
            # Use the throttled_request helper instead of requests.get
            response = throttled_request(url, params=params, headers=headers)
            data = response.json()
            papers = []
            for item in data.get("data", []):
                # FIX: Define the ID variable here
                s2_id = item.get("paperId", "N/A")
                
                papers.append(Paper(
                    source="Semantic Scholar",
                    id=s2_id,
                    title=item.get("title", "Untitled"),
                    abstract=item.get("abstract") or "No abstract available.",
                    # FIX: Use the variable 's2_id' defined above
                    url=f"https://www.semanticscholar.org/paper/{s2_id}"
                ))
            return papers
        except Exception as e:
            st.error(f"Semantic Scholar Error: {e}")
            return []

class COREService:
    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        params = {"q": query, "limit": max_results}
        headers = {"Authorization": f"Bearer {Config.CORE_API_KEY}"} if hasattr(Config, 'CORE_API_KEY') else {}
        
        try:
            url = Config.CORE_API_URL if hasattr(Config, 'CORE_API_URL') else "https://api.core.ac.uk/v3/search/works"
            # Use the throttled_request helper
            response = throttled_request(url, params=params, headers=headers)
            data = response.json()
            papers = []
            for item in data.get("results", []):
                papers.append(Paper(
                    source="CORE",
                    id=str(item.get("id", "")),
                    title=item.get("title", "Untitled"),
                    abstract=item.get("abstract") or "No abstract available."
                ))
            return papers
        except Exception as e:
            st.error(f"CORE API Error: {e}")
            return []

class DataAggregator:
    """Aggregates data from all active sources while respecting rate limits."""
    
    SERVICE_MAP = {
        DataSource.PUBMED.value: PubMedService.fetch,
        # DataSource.BIG3_JOURNALS.value: TopJournalsService.fetch,
        DataSource.ARXIV.value: ArXivService.fetch,
        DataSource.BIORXIV.value: BioRxivService.fetch,
        "Semantic Scholar": SemanticScholarService.fetch,
        "CORE": COREService.fetch
    }
    
    @staticmethod
    def fetch_all(query: str, active_sources: List[str], max_per_source: int = 10, uploaded_files=None, limit: int = None):
        """
        Aggregates raw data from all active sources. 
        Deduplication is removed to ensure PRISMA counts accurately reflect total records.
        """
        all_papers = []
        source_counts = {} 
        
        search_count = limit if limit is not None else max_per_source
        
        for source in active_sources:
            papers = []
            status_text = st.empty()
            status_text.write(f"🔍 Searching {source}...")
            
            try:
                # 1. Handle local files
                if source == DataSource.LOCAL_PDF.value:
                    if uploaded_files:
                        from data_services import PDFService
                        papers = PDFService.process_files(uploaded_files)
                
                # 2. Handle API-based sources (PubMed, ArXiv, Semantic Scholar, CORE, etc.)
                elif source in DataAggregator.SERVICE_MAP:
                    fetch_func = DataAggregator.SERVICE_MAP[source]
                    papers = fetch_func(query, search_count)
                
                # 3. Track RAW counts per source and update UI
                count = len(papers)
                all_papers.extend(papers)
                source_counts[source] = count
                
                # Update the status placeholder with the final count
                if count > 0:
                    status_text.write(f"✅ {source}: {count} papers found")
                else:
                    status_text.write(f"ℹ️ {source}: 0 papers found")

            except Exception as e:
                # Catch failures so one source doesn't break the entire search
                status_text.write(f"❌ {source}: Error occurred")
                st.error(f"Error fetching from {source}: {str(e)}")
                source_counts[source] = 0

        if limit is not None:
            return all_papers[:limit], source_counts
            
        return all_papers, source_counts

    @staticmethod
    def simulate_yield(query: str, active_sources: List[str]) -> Dict[str, int]:
        """Returns the absolute total of papers matching the query in each database."""
        results = {}
        for source in active_sources:
            try:
                # PubMed & Top Journals (Uses E-Search 'Count' field)
                if source in [DataSource.PUBMED.value, DataSource.BIG3_JOURNALS.value]:
                    from Bio import Entrez
                    Entrez.email = Config.ENTREZ_EMAIL
                    search_query = PubMedService.get_query(query) if source == DataSource.PUBMED.value else TopJournalsService.get_query(query)
                    
                    # retmax=0 makes the request instant as no records are downloaded
                    handle = Entrez.esearch(db="pubmed", term=search_query, retmax=0)
                    record = Entrez.read(handle)
                    results[source] = int(record.get("Count", 0))

                # ArXiv (Uses OpenSearch 'totalResults' field)
                elif source == DataSource.ARXIV.value:
                    from utils import QueryCleaner
                    import xml.etree.ElementTree as ET
                    clean_query = QueryCleaner.clean_for_general_search(query)
                    url = f"{Config.ARXIV_API_URL}?search_query=all:{clean_query}&max_results=0"
                    resp = throttled_request(url)
                    root = ET.fromstring(resp.content)
                    ns = {'os': 'http://a9.com/-/spec/opensearch/1.1/'}
                    total_node = root.find('os:totalResults', ns)
                    results[source] = int(total_node.text) if total_node is not None else 0

                # Inside DataAggregator.simulate_yield:
                elif source == DataSource.BIORXIV.value:
                    # BioRxiv doesn't have a 'total count' search API easily accessible via GET 
                    # without fetching data, so we use the fetch method to see what we get 
                    # in the current lookback window.
                    temp_papers = BioRxivService.fetch(query, max_results=100)
                    results[source] = len(temp_papers)

                # Semantic Scholar (Uses the 'total' metadata field)
                elif source == "Semantic Scholar":
                    params = {'query': query, 'limit': 1} # Minimal request
                    headers = {'x-api-key': Config.SEMANTIC_SCHOLAR_KEY} if hasattr(Config, 'SEMANTIC_SCHOLAR_KEY') else {}
                    url = "https://api.semanticscholar.org/graph/v1/paper/search"
                    resp = throttled_request(url, params=params, headers=headers).json()
                    results[source] = int(resp.get('total', 0))

                # CORE (Uses the 'totalHits' field)
                elif source == "CORE":
                    headers = {"Authorization": f"Bearer {Config.CORE_API_KEY}"} if hasattr(Config, 'CORE_API_KEY') else {}
                    payload = {"q": query, "limit": 1}
                    resp = throttled_request(Config.CORE_API_URL, params=payload, headers=headers).json()
                    results[source] = int(resp.get('totalHits', 0))
                
            except Exception as e:
                results[source] = 0
        return results

    @staticmethod
    def get_total_counts(query: str, sources: List[str]) -> Dict[str, int]:
        """Fetches only the total result count for a query from selected sources."""
        results = {}
        clean_query = QueryCleaner.clean_for_general_search(query)
        
        for source in sources:
            try:
                # PubMed: Use esearch with retmax=0
                if source == DataSource.PUBMED.value:
                    Entrez.email = Config.ENTREZ_EMAIL
                    handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
                    record = Entrez.read(handle)
                    results[source] = int(record.get('Count', 0))

                # ArXiv: Parse the totalResults from the OpenSearch XML
                elif source == DataSource.ARXIV.value:
                    url = f"{Config.ARXIV_API_URL}?search_query=all:{clean_query}&max_results=0"
                    resp = throttled_request(url)
                    root = ET.fromstring(resp.content)
                    ns = {'os': 'http://a9.com/-/spec/opensearch/1.1/'}
                    total_node = root.find('os:totalResults', ns)
                    results[source] = int(total_node.text) if total_node is not None else 0

                # Semantic Scholar: Use the 'total' field in response metadata
                elif source == "Semantic Scholar":
                    params = {'query': query, 'limit': 1, 'fields': 'title'} # Added fields
                    resp = throttled_request(url, params=params, headers=headers).json()
                    # Debug print here would show you the raw JSON if it's 0
                    results[source] = int(resp.get('total', 0))

                # CORE: Use 'totalHits' field
                elif source == "CORE":
                    payload = {"q": query, "limit": 0} # limit 0 is faster for just counts
                    resp = throttled_request(Config.CORE_API_URL, params=payload, headers=headers).json()
                    # CORE v3 usually returns a 'totalHits' at the top level
                    results[source] = int(resp.get('totalHits', 0))
                
                # BioRxiv: The 'messages' array contains the total 'count'
                elif source == DataSource.BIORXIV.value:
                    # Note: BioRxiv search is usually date-based in your current config
                    # This assumes you are fetching the last N days as per Config
                    url = f"{Config.BIORXIV_API_URL}/biorxiv/last/{Config.BIORXIV_LOOKBACK_DAYS}"
                    resp = throttled_request(url).json()
                    results[source] = int(resp.get('messages', [{}])[0].get('count', 0))

            except Exception as e:
                st.warning(f"Could not fetch count for {source}: {e}")
                results[source] = 0
                
        return results