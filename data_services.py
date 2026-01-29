# ============================================================================
# FILE: data_services.py
# All data fetching services in one file
# ============================================================================

from Bio import Entrez
import requests
import xml.etree.ElementTree as ET
import urllib.parse
from pypdf import PdfReader
from datetime import datetime, timedelta
from typing import List
import streamlit as st
from config import Config, DataSource
from models import Paper
from utils import QueryCleaner
import time

# Global variable to track the last time an API request was made
_last_request_time = 0.0

def throttled_request(url: str, params: dict = None, headers: dict = None, method: str = "GET") -> requests.Response:
    """Ensures all outgoing requests respect a 1-request-per-second limit."""
    global _last_request_time
    
    # Calculate time since last request
    elapsed = time.time() - _last_request_time
    if elapsed < 1.1:  # 1.1 seconds for safety buffer
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
            # Search for IDs
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results
            )
            id_list = Entrez.read(search_handle)["IdList"]
            
            if not id_list:
                return []
            
            # Fetch details
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
                    source=DataSource.PUBMED.value,
                    id=str(citation['PMID']),
                    title=citation['Article']['ArticleTitle'],
                    abstract=str(abstract_text)
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
    
    CATEGORIES = "cat:q-bio.PE OR cat:q-bio.QM OR cat:stat.AP"
    
    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        """Fetch papers from arXiv."""
        clean_query = QueryCleaner.clean_for_general_search(query)
        
        if not clean_query:
            return []
        
        encoded_query = urllib.parse.quote(clean_query)
        url = (
            f"{Config.ARXIV_API_URL}"
            f"?search_query=all:{encoded_query}+AND+({ArXivService.CATEGORIES})"
            f"&max_results={max_results}"
        )
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            papers = []
            for entry in root.findall('atom:entry', namespace):
                paper_id = entry.find('atom:id', namespace).text.split('/')[-1]
                title = entry.find('atom:title', namespace).text.strip()
                abstract = entry.find('atom:summary', namespace).text.strip()
                
                papers.append(Paper(
                    source=DataSource.ARXIV.value,
                    id=paper_id,
                    title=title,
                    abstract=abstract
                ))
            
            return papers
            
        except Exception as e:
            st.error(f"arXiv fetch error: {e}")
            return []


class BioRxivService:
    """Handles bioRxiv data fetching."""
    
    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        """Fetch preprints from bioRxiv with keyword matching."""
        clean_query = QueryCleaner.clean_for_general_search(query).lower()
        keywords = [
            k for k in clean_query.split()
            if len(k) > Config.MIN_KEYWORD_LENGTH
        ]
        
        if not keywords:
            return []
        
        filtered_papers = []
        cursor = 0
        attempts = 0
        
        start_date = (
            datetime.now() - timedelta(days=Config.BIORXIV_LOOKBACK_DAYS)
        ).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        while (len(filtered_papers) < max_results and
               attempts < Config.BIORXIV_MAX_ATTEMPTS):
            
            url = f"{Config.BIORXIV_API_URL}/{start_date}/{end_date}/{cursor}"
            
            try:
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                data = response.json()
                
                collection = data.get('collection', [])
                if not collection:
                    break
                
                for preprint in collection:
                    text = f"{preprint['title']} {preprint['abstract']}".lower()
                    relevance = sum(1 for k in keywords if k in text)
                    
                    if relevance > 0:
                        filtered_papers.append(Paper(
                            source=DataSource.BIORXIV.value,
                            id=preprint['doi'],
                            title=preprint['title'],
                            abstract=preprint['abstract'],
                            score=relevance
                        ))
                    
                    if len(filtered_papers) >= max_results:
                        break
                
                cursor += Config.BIORXIV_BATCH_SIZE
                attempts += 1
                
            except Exception as e:
                st.error(f"bioRxiv fetch error: {e}")
                break
        
        return sorted(filtered_papers, key=lambda x: x.score or 0, reverse=True)


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
                papers.append(Paper(
                    source="Semantic Scholar",
                    id=item.get("paperId", "N/A"),
                    title=item.get("title", "Untitled"),
                    abstract=item.get("abstract") or "No abstract available."
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
    
    # This map links the UI selection to the Python class
    SERVICE_MAP = {
        DataSource.PUBMED.value: PubMedService.fetch,
        DataSource.BIG3_JOURNALS.value: TopJournalsService.fetch,
        DataSource.ARXIV.value: ArXivService.fetch,
        DataSource.BIORXIV.value: BioRxivService.fetch,
        "Semantic Scholar": SemanticScholarService.fetch,
        "CORE": COREService.fetch
    }
    
    @staticmethod
    def fetch_all(query: str, active_sources: List[str], max_per_source: int = 10, uploaded_files=None, limit: int = None) -> List[Paper]:
        """The main entry point called by app.py to start the search."""
        all_papers = []
        
        # If 'limit' is passed (from Brainstorm), we use it to cap the search per source
        # to keep the quick summary fast.
        search_count = limit if limit is not None else max_per_source
        
        for source in active_sources:
            # Handle local files separately
            if source == DataSource.LOCAL_PDF.value:
                if uploaded_files:
                    papers = PDFService.process_files(uploaded_files)
                    all_papers.extend(papers)
                    st.write(f"✅ {source}: {len(papers)} files processed")
            
            # Handle API-based sources
            elif source in DataAggregator.SERVICE_MAP:
                st.write(f"🔍 Searching {source}...")
                fetch_func = DataAggregator.SERVICE_MAP[source]
                
                # Use the adjusted search_count (either the 5-paper limit or the full max)
                papers = fetch_func(query, search_count)
                all_papers.extend(papers)
                st.write(f"✅ {source}: {len(papers)} papers found")
        
        total_before = len(all_papers)
        if total_before > 0:
            from utils import Deduplicator
            unique, duplicates = Deduplicator.run(all_papers)
            
            # Store duplicates in session state for the UI to see
            st.session_state['last_duplicates'] = duplicates
            all_papers = unique
            
            removed = len(duplicates)
            if removed > 0:
                st.success(f"🔍 Removed {removed} duplicates.")

        # If a hard limit was requested (e.g., 5 papers), ensure we return exactly that or fewer
        if limit is not None:
            return all_papers[:limit]
            
        return all_papers