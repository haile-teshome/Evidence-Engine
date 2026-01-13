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


class DataAggregator:
    """Aggregates data from multiple sources."""
    
    SERVICE_MAP = {
        DataSource.PUBMED.value: PubMedService.fetch,
        DataSource.BIG3_JOURNALS.value: TopJournalsService.fetch,
        DataSource.ARXIV.value: ArXivService.fetch,
        DataSource.BIORXIV.value: BioRxivService.fetch,
    }
    
    @staticmethod
    def fetch_all(
        query: str,
        active_sources: List[str],
        max_per_source: int,
        uploaded_files=None
    ) -> List[Paper]:
        """Fetch papers from all active sources."""
        all_papers = []
        
        for source in active_sources:
            if source == DataSource.LOCAL_PDF.value:
                if uploaded_files:
                    papers = PDFService.process_files(uploaded_files)
                    all_papers.extend(papers)
                    st.write(f"✅ {source}: {len(papers)} files")
            elif source in DataAggregator.SERVICE_MAP:
                fetch_func = DataAggregator.SERVICE_MAP[source]
                papers = fetch_func(query, max_per_source)
                all_papers.extend(papers)
                st.write(f"✅ {source}: {len(papers)} papers")
        
        return all_papers


