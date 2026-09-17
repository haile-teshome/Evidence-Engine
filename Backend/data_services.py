from Bio import Entrez
import re
import requests
import xml.etree.ElementTree as ET
import urllib.parse
from pypdf import PdfReader
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from config import Config, DataSource
from models import Paper
import time

_last_request_time = 0.0

def throttled_request(url: str, params: dict = None, headers: dict = None, method: str = "GET", max_retries: int = 3, timeout: int = 30) -> requests.Response:
    """Ensures all outgoing requests respect a 1-request-per-second limit with retry logic."""
    global _last_request_time
    
    elapsed = time.time() - _last_request_time
    if elapsed < 1.1:  
        time.sleep(1.1 - elapsed)
    
    for attempt in range(max_retries):
        try:
            if method.upper() == "POST":
                response = requests.post(url, json=params, headers=headers, timeout=timeout)
            else:
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
            
            _last_request_time = time.time()
            return response
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                time.sleep(wait_time)
                continue
            raise  # Re-raise on final attempt
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            raise
    
    _last_request_time = time.time()
    return response

def contact_email() -> str:
    """The user's own contact address from their profile, or the configured
    fallback. NCBI asks for a real address so it can warn before blocking."""
    try:
        from request_creds import get_cred
        e = (get_cred("contact_email") or "").strip()
        if e and "@" in e and not e.lower().endswith(("example.com", "example.org")):
            return e
    except Exception:
        pass
    return Config.ENTREZ_EMAIL


def entrez_abstract(article: dict) -> str:
    """Join every section of a PubMed abstract into one string.

    Entrez returns ``AbstractText`` as a LIST, one entry per labelled section of
    a structured abstract (BACKGROUND / METHODS / RESULTS / CONCLUSIONS). Taking
    only ``[0]`` handed the screener the background paragraph alone, which is
    precisely the section that never states what data or methods a study used:
    measured against PubMed, that dropped 1981 characters to 158 on a typical
    structured abstract and about half the corpus was affected.
    """
    parts = (article.get("Abstract") or {}).get("AbstractText") or []
    if isinstance(parts, str):
        return parts.strip()
    out: List[str] = []
    for p in parts:
        text = str(p).strip()
        if not text:
            continue
        # Biopython attaches the section label as an XML attribute.
        label = ""
        try:
            label = str((p.attributes or {}).get("Label", "")).strip()
        except AttributeError:
            pass
        out.append(f"{label}: {text}" if label else text)
    return " ".join(out).strip()


class PubMedService:
    """Handles PubMed data fetching."""

    _mesh_cache: Dict[str, Tuple[str, List[str]]] = {}

    @staticmethod
    def mesh_lookup(term: str) -> Tuple[str, List[str]]:
        """Ground a free-text concept against real MeSH via NCBI E-utilities.

        Returns (official_descriptor, entry_terms). Empty ("", []) when no
        descriptor matches. Cached per process. This is what turns an
        LLM-suggested (possibly hallucinated) heading into a validated MeSH
        descriptor plus its official synonym (entry) terms, so the assembled
        query uses controlled vocabulary that actually exists in PubMed.
        """
        term = (term or "").strip()
        if not term:
            return "", []
        key = term.lower()
        if key in PubMedService._mesh_cache:
            return PubMedService._mesh_cache[key]
        Entrez.email = contact_email()
        try:
            from request_creds import get_cred
            _k = get_cred("ncbi") or getattr(Config, "NCBI_API_KEY", "")
            if _k:
                Entrez.api_key = _k
        except Exception:
            pass
        heading, entries = "", []
        try:
            sh = Entrez.esearch(db="mesh", term=term, retmax=1)
            ids = Entrez.read(sh).get("IdList", [])
            sh.close()
            if ids:
                summ = Entrez.esummary(db="mesh", id=ids[0])
                recs = Entrez.read(summ)
                summ.close()
                terms = list((recs[0] if recs else {}).get("DS_MeshTerms", []) or [])
                if terms:
                    heading = str(terms[0])
                    entries = [str(t) for t in terms[1:]]
        except Exception as e:
            print(f"[mesh_lookup] {term!r}: {e}")
        PubMedService._mesh_cache[key] = (heading, entries)
        return heading, entries

    @staticmethod
    def fetch_mesh_for_pmids(pmids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch MeSH headings + title for known PubMed records (seed studies),
        so their controlled-vocabulary terms can seed query expansion. Returns
        {pmid: {"mesh": [...], "title": str}}."""
        pmids = [str(p).strip() for p in pmids if str(p).strip().isdigit()]
        if not pmids:
            return {}
        Entrez.email = contact_email()
        try:
            from request_creds import get_cred
            _k = get_cred("ncbi") or getattr(Config, "NCBI_API_KEY", "")
            if _k:
                Entrez.api_key = _k
        except Exception:
            pass
        out: Dict[str, Dict[str, Any]] = {}
        try:
            h = Entrez.efetch(db="pubmed", id=",".join(pmids[:50]), rettype="medline", retmode="xml")
            recs = Entrez.read(h)
            h.close()
            for art in recs.get("PubmedArticle", []):
                cit = art["MedlineCitation"]
                pmid = str(cit.get("PMID", ""))
                mesh = [str(m["DescriptorName"]) for m in cit.get("MeshHeadingList", []) or []]
                art_info = cit.get("Article", {})
                title = str(art_info.get("ArticleTitle", "") or "")
                out[pmid] = {"mesh": mesh, "title": title}
        except Exception as e:
            print(f"[fetch_mesh_for_pmids] {e}")
        return out

    @staticmethod
    def fetch(query: str, max_results: int, sort: str = "relevance",
              year_from: int = None, year_to: int = None) -> List[Paper]:
        """Fetch papers from PubMed.

        ``sort`` decides WHICH records are kept when a query matches more than
        ``max_results``: "relevance" keeps PubMed's Best Match top-N (the default,
        appropriate for a capped sample), "recent" keeps the most recently
        published. When the total matches are fewer than the cap, every match is
        returned regardless of ``sort`` (esearch returns all of them).

        ``year_from`` / ``year_to`` restrict by publication year (inclusive) via
        the esearch date filter, so the limit is applied to a date-scoped pool."""
        Entrez.email = contact_email()
        # An optional free NCBI key raises the rate limit from 3 to 10 req/s.
        from request_creds import get_cred
        _ncbi_key = get_cred("ncbi") or getattr(Config, "NCBI_API_KEY", "")
        if _ncbi_key:
            Entrez.api_key = _ncbi_key

        # Add title/abstract search if not specified
        if "[tiab]" not in query.lower() and "[" not in query:
            query = f"({query})[tiab]"

        # Map the selection strategy to an Entrez sort key.
        esort = "pub_date" if str(sort).lower() in ("recent", "pub_date", "date") else "relevance"

        # Optional publication-date window (researcher-set filter).
        date_kwargs = {}
        if year_from or year_to:
            date_kwargs = {
                "datetype": "pdat",
                "mindate": str(int(year_from)) if year_from else "1800",
                "maxdate": str(int(year_to)) if year_to else "3000",
            }

        try:
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort=esort,
                **date_kwargs,
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
                
                abstract_text = entrez_abstract(citation['Article'])

                papers.append(Paper(
                    source=DataSource.PUBMED.value,
                    id=pmid,
                    title=citation['Article']['ArticleTitle'],
                    abstract=abstract_text,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                ))
            
            return papers
            
        except Exception as e:
            print(f"PubMed fetch error: {e}")
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
        Entrez.email = contact_email()
        
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
                citation = article['MedlineCitation']['Article']
                papers.append(Paper(
                    source=DataSource.PUBMED.value,
                    id=str(citation['PMID']),
                    title=citation['ArticleTitle'],
                    abstract=entrez_abstract(citation)
                ))
            
            return papers
            
        except Exception as e:
            print(f"PubMed fetch error: {e}")
            return []


class ArXivService:
    """Handles arXiv data fetching."""
    
    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        """Fetch papers from arXiv."""
        from utils import QueryCleaner
        clean_query = QueryCleaner.clean_for_general_search(query)
        params = {
            'search_query': f'all:{clean_query}',
            'start': 0,
            'max_results': max_results
        }
        
        try:
            response = throttled_request(Config.ARXIV_API_URL, params=params)
            
            # Check if response is valid XML before parsing
            content_type = response.headers.get('content-type', '').lower()
            if 'xml' not in content_type and not response.content.strip().startswith(b'<'):
                print(f"ArXiv API returned non-XML response (content-type: {content_type})")
                return []
            
            # Check for HTTP error status
            if response.status_code != 200:
                print(f"ArXiv API returned status {response.status_code} - skipping")
                return []
            
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
            print(f"ArXiv fetch error (silent): {e}")
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
            print(f"BioRxiv fetch error: {e}")
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
                print(f"Failed to process {file.name}: {e}")
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
        # Semantic Scholar works fully keyless; a key (per-request or env) only
        # raises the rate limit, so it is applied when present and skipped otherwise.
        from request_creds import get_cred
        s2_key = get_cred("semantic_scholar") or getattr(Config, "SEMANTIC_SCHOLAR_KEY", "")
        headers = {"x-api-key": s2_key} if s2_key else {}

        try:
            url = getattr(Config, "SEMANTIC_SCHOLAR_URL", "https://api.semanticscholar.org/graph/v1/paper/search")
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
            print(f"Semantic Scholar Error: {e}")
            return []

class COREService:
    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        params = {"q": query, "limit": max_results}
        # CORE is the only source that genuinely requires a key. Without one
        # (per-request or env) we skip it quietly so the rest of the search still
        # runs, rather than erroring on a 401.
        from request_creds import get_cred
        core_key = get_cred("core") or getattr(Config, "CORE_API_KEY", "")
        if not core_key:
            return []
        headers = {"Authorization": f"Bearer {core_key}"}

        try:
            url = getattr(Config, "CORE_API_URL", "https://api.core.ac.uk/v3/search/works")
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
            print(f"CORE API Error: {e}")
            return []

def to_europepmc_query(query: str) -> str:
    """Translate a PubMed-syntax query into Europe PMC syntax.

    Deleting the field tags (the previous behaviour) does NOT preserve the query:
    an unqualified term in Europe PMC matches FULL TEXT, so `"machine learning"`
    hits any paper that mentions the phrase anywhere. Measured on one 4-block
    query: PubMed 5 hits, tags-deleted 1311, this translation 2. That ~260x
    inflation is why a single source could dominate a corpus with records that
    never matched the question.

    MeSH clauses are dropped rather than mapped. Europe PMC's MESH: field is not
    OR-safe — `MESH:"Dentistry" OR MESH:"Chronic Disease"` returns 12190, FEWER
    than `MESH:"Dentistry"` alone at 18624 — so those clauses cannot be trusted.
    Its MeSH coverage is partial anyway (preprints and non-MEDLINE records carry
    none), and the [tiab] synonyms alongside them carry the same concepts.

    A query already in Europe PMC syntax has no bracket tags, so this is a no-op.
    """
    q = query or ""
    # MeSH clauses, plus an adjacent OR so the boolean stays well-formed.
    q = re.sub(r'(?:"[^"]*"|[^\s()]+)\s*\[(?:mesh|mesh terms|mh)(?::noexp)?\]\s*(?:OR\s+)?', "", q, flags=re.I)
    # Title/abstract tags -> Europe PMC's field prefix.
    q = re.sub(r'("[^"]+"|[^\s()]+?)\s*\[(?:tiab|title/abstract|tw|text ?word|ti|title)\]',
               r"TITLE_ABS:\1", q, flags=re.I)
    # Any remaining tags (date, language, publication type) and their terms.
    q = re.sub(r'(?:"[^"]*"|[^\s()]+)\s*\[[^\]]+\]\s*(?:OR\s+)?', "", q)
    q = re.sub(r"\[[^\]]+\]", "", q)
    # Tidy the boolean wreckage the removals leave behind (empty groups, dangling
    # operators). Repeated because collapsing one group can expose another.
    for _ in range(3):
        q = re.sub(r"\(\s*(?:AND|OR)\s+", "(", q, flags=re.I)
        q = re.sub(r"\s*(?:AND|OR)\s*\)", ")", q, flags=re.I)
        q = re.sub(r"\(\s*\)", "", q)
        q = re.sub(r"\b(AND|OR)\s+(AND|OR)\b", r"\1", q, flags=re.I)
        q = re.sub(r"^\s*(?:AND|OR)\s+", "", q, flags=re.I)
        q = re.sub(r"\s+(?:AND|OR)\s*$", "", q, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip()
    return q


class EuropePMCService:
    """Handles Europe PMC data fetching."""

    @staticmethod
    def fetch(query: str, max_results: int, sort: str = "relevance",
              year_from: int = None, year_to: int = None) -> List[Paper]:
        try:
            epmc_query = to_europepmc_query(query)
            # Optional publication-year window.
            if year_from or year_to:
                yf = int(year_from) if year_from else 1800
                yt = int(year_to) if year_to else 3000
                epmc_query = f"({epmc_query}) AND (PUB_YEAR:[{yf} TO {yt}])"
            url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            # Europe PMC caps pageSize at 1000 and, above that, returns HTTP 200
            # with an EMPTY result list rather than an error. A single request for
            # more than 1000 therefore yields zero papers silently, which is why
            # this source contributed nothing whenever its planned yield was
            # large. Page with cursorMark instead.
            PAGE = 1000
            MAX_PAGES = 60                      # ~60k ceiling; guards runaway loops
            papers: List[Paper] = []
            cursor = "*"
            for _ in range(MAX_PAGES):
                if len(papers) >= max_results:
                    break
                params = {
                    "query": epmc_query or query,
                    "format": "json",
                    "pageSize": min(PAGE, max_results - len(papers)),
                    "cursorMark": cursor,
                    # "core" returns the abstract text; "lite" omits it. Without
                    # abstracts the downstream PICO appraisal has nothing to
                    # anchor quotes against and every cell collapses to NA.
                    "resultType": "core",
                }
                # Europe PMC ranks by relevance by default; for "recent" ask it to
                # sort by publication date descending so the kept N are the newest.
                if str(sort).lower() in ("recent", "pub_date", "date"):
                    params["sort"] = "P_PDATE_D desc"
                resp = throttled_request(url, params=params).json()
                batch = resp.get("resultList", {}).get("result", []) or []
                for r in batch:
                    pid = r.get("id") or r.get("pmid") or r.get("doi") or ""
                    src_code = r.get("source", "MED")
                    paper_url = f"https://europepmc.org/article/{src_code}/{pid}" if pid else ""
                    papers.append(Paper(
                        source="Europe PMC",
                        id=str(pid),
                        title=r.get("title", "") or "",
                        abstract=r.get("abstractText", "") or "",
                        url=paper_url,
                    ))
                nxt = resp.get("nextCursorMark") or ""
                # Stop on an empty page, an exhausted cursor, or a cursor that
                # stops advancing (the documented end-of-results signal).
                if not batch or not nxt or nxt == cursor:
                    break
                cursor = nxt
            return papers[:max_results]
        except Exception as e:
            print(f"Europe PMC fetch error: {e}")
            return []


def _reconstruct_inverted(idx: dict) -> str:
    """Rebuild plain text from OpenAlex's inverted-index abstract format."""
    if not idx:
        return ""
    positions = []
    for word, locs in idx.items():
        for loc in locs:
            positions.append((loc, word))
    positions.sort()
    return " ".join(w for _, w in positions)


class OpenAlexService:
    """OpenAlex — 250M+ scholarly works, no API key required."""

    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        try:
            clean = re.sub(r"\[[^\]]+\]", "", query).strip()
            url = "https://api.openalex.org/works"
            params = {
                "search": clean or query,
                "per_page": min(max(max_results, 1), 200),
                "select": "id,title,abstract_inverted_index,doi,open_access,publication_year",
            }
            resp = throttled_request(url, params=params).json()
            papers: List[Paper] = []
            for w in resp.get("results", []):
                abs_idx = w.get("abstract_inverted_index") or {}
                abstract = _reconstruct_inverted(abs_idx) if abs_idx else ""
                doi = (w.get("doi") or "").replace("https://doi.org/", "")
                oa = w.get("open_access", {}) or {}
                paper_id = (w.get("id") or "").split("/")[-1] or doi
                paper_url = oa.get("oa_url") or (f"https://doi.org/{doi}" if doi else (w.get("id") or ""))
                papers.append(Paper(
                    source="OpenAlex",
                    id=str(paper_id),
                    title=w.get("title", "") or "",
                    abstract=abstract,
                    url=paper_url,
                ))
            return papers[:max_results]
        except Exception as e:
            print(f"OpenAlex fetch error: {e}")
            return []

    @staticmethod
    def related(seed_title: str, seed_doi: str = "", max_results: int = 50) -> List[Paper]:
        """OpenAlex 'related_works' for a seed paper — algorithmic topical
        similarity (shared concepts), NOT citations. Complements citation
        snowballing by surfacing same-topic papers that have no citation link."""
        try:
            mail = {}  # no contact email is sent to third-party APIs
            # 1. resolve the seed to an OpenAlex work and read its related_works
            if seed_doi:
                doi = seed_doi.replace("https://doi.org/", "")
                w = throttled_request(f"https://api.openalex.org/works/doi:{doi}",
                                      params={**mail, "select": "id,related_works"}).json()
            else:
                r = throttled_request("https://api.openalex.org/works",
                                      params={**mail, "search": seed_title, "per_page": 1,
                                              "select": "id,related_works"}).json()
                w = (r.get("results") or [None])[0]
            rel_ids = [rid.split("/")[-1] for rid in ((w or {}).get("related_works") or [])][:max_results]
            if not rel_ids:
                return []
            # 2. batch-fetch the related works
            resp = throttled_request("https://api.openalex.org/works",
                                     params={**mail, "filter": "ids.openalex:" + "|".join(rel_ids),
                                             "per_page": min(len(rel_ids), 200),
                                             "select": "id,title,abstract_inverted_index,doi,open_access"}).json()
            papers: List[Paper] = []
            for work in resp.get("results", []):
                abs_idx = work.get("abstract_inverted_index") or {}
                doi = (work.get("doi") or "").replace("https://doi.org/", "")
                oa = work.get("open_access", {}) or {}
                pid = (work.get("id") or "").split("/")[-1] or doi
                papers.append(Paper(
                    source="OpenAlex (similar)",
                    id=str(pid),
                    title=work.get("title", "") or "",
                    abstract=_reconstruct_inverted(abs_idx) if abs_idx else "",
                    url=oa.get("oa_url") or (f"https://doi.org/{doi}" if doi else (work.get("id") or "")),
                ))
            return papers
        except Exception as e:
            print(f"OpenAlex related error: {e}")
            return []


class CrossRefService:
    """CrossRef — 150M+ DOI records across all disciplines."""

    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        try:
            clean = re.sub(r"\[[^\]]+\]", "", query).strip()
            url = "https://api.crossref.org/works"
            params = {
                "query": clean or query,
                "rows": min(max_results, 100),
                "select": "DOI,title,abstract,URL,author",
            }
            headers = {"User-Agent": "EvidenceEngine/1.0"}
            resp = throttled_request(url, params=params, headers=headers).json()
            papers: List[Paper] = []
            for it in resp.get("message", {}).get("items", []):
                title = " ".join(it.get("title") or []) or ""
                doi = it.get("DOI", "")
                raw_abs = it.get("abstract", "") or ""
                # CrossRef abstracts are JATS XML fragments — strip tags.
                abstract = re.sub(r"<[^>]+>", " ", raw_abs).strip()
                abstract = re.sub(r"\s+", " ", abstract)
                papers.append(Paper(
                    source="CrossRef",
                    id=doi,
                    title=title,
                    abstract=abstract,
                    url=it.get("URL") or (f"https://doi.org/{doi}" if doi else ""),
                ))
            return papers
        except Exception as e:
            print(f"CrossRef fetch error: {e}")
            return []


class MedRxivService:
    """medRxiv preprints, retrieved via Europe PMC with the preprint source filter."""

    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        try:
            clean = re.sub(r"\[[^\]]+\]", "", query).strip()
            epmc_query = f"({clean}) AND SRC:PPR"
            url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
            params = {
                "query": epmc_query,
                "format": "json",
                "pageSize": max_results * 2,  # over-fetch; we filter for medrxiv specifically
                # "core" returns abstractText; "lite" omits it, leaving the
                # downstream PICO appraisal nothing to anchor quotes against.
                "resultType": "core",
            }
            resp = throttled_request(url, params=params).json()
            papers: List[Paper] = []
            for it in resp.get("resultList", {}).get("result", []):
                journal = (it.get("bookOrReportDetails", {}) or {}).get("publisher", "") or ""
                publisher = (it.get("publisher") or "")
                jt = (it.get("journalTitle") or "").lower()
                doi = (it.get("doi") or "")
                if "medrxiv" not in (jt + journal + publisher + doi).lower():
                    continue
                pid = it.get("id") or doi
                papers.append(Paper(
                    source="medRxiv",
                    id=str(pid),
                    title=it.get("title", "") or "",
                    abstract=it.get("abstractText", "") or "",
                    url=(f"https://www.medrxiv.org/content/{doi}" if doi else f"https://europepmc.org/article/PPR/{pid}"),
                ))
                if len(papers) >= max_results:
                    break
            return papers
        except Exception as e:
            print(f"medRxiv fetch error: {e}")
            return []


class DOAJService:
    """DOAJ — Directory of Open Access Journals; all articles are open access."""

    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        try:
            from urllib.parse import quote
            clean = re.sub(r"\[[^\]]+\]", "", query).strip()
            url = f"https://doaj.org/api/v2/search/articles/{quote(clean or query)}"
            params = {"pageSize": min(max_results, 100)}
            resp = throttled_request(url, params=params).json()
            papers: List[Paper] = []
            for it in resp.get("results", []):
                bib = it.get("bibjson", {}) or {}
                title = bib.get("title", "") or ""
                abstract = bib.get("abstract", "") or ""
                doi = ""
                for ident in bib.get("identifier", []) or []:
                    if (ident.get("type") or "").lower() == "doi":
                        doi = ident.get("id", "")
                        break
                link = ""
                for ln in bib.get("link", []) or []:
                    if (ln.get("type") or "").lower() == "fulltext":
                        link = ln.get("url", "")
                        break
                paper_id = it.get("id") or doi
                papers.append(Paper(
                    source="DOAJ",
                    id=str(paper_id),
                    title=title,
                    abstract=abstract,
                    url=link or (f"https://doi.org/{doi}" if doi else f"https://doaj.org/article/{it.get('id', '')}"),
                ))
            return papers
        except Exception as e:
            print(f"DOAJ fetch error: {e}")
            return []


class ClinicalTrialsService:
    """ClinicalTrials.gov registered trials via the keyless API v2. Surfaces
    registered and often-unpublished studies (grey literature), the main defense
    against publication bias that PRISMA/Cochrane expect a review to search."""

    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        try:
            clean = re.sub(r"\[[^\]]+\]", "", query).strip() or query
            url = "https://clinicaltrials.gov/api/v2/studies"
            params = {"query.term": clean, "pageSize": min(max(max_results, 1), 100)}
            resp = throttled_request(url, params=params).json()
            papers: List[Paper] = []
            for study in resp.get("studies", []):
                ps = study.get("protocolSection", {}) or {}
                idm = ps.get("identificationModule", {}) or {}
                nct = idm.get("nctId", "") or ""
                if not nct:
                    continue
                title = idm.get("briefTitle") or idm.get("officialTitle") or nct
                desc = ps.get("descriptionModule", {}) or {}
                abstract = desc.get("briefSummary") or desc.get("detailedDescription") or ""
                status = (ps.get("statusModule", {}) or {}).get("overallStatus", "") or ""
                papers.append(Paper(
                    source="ClinicalTrials.gov",
                    id=nct,
                    title=title,
                    # Prefix the recruitment status so screening can see, e.g., an
                    # unpublished completed trial vs one still recruiting.
                    abstract=(f"[{status}] {abstract}".strip() if status else abstract),
                    url=f"https://clinicaltrials.gov/study/{nct}",
                ))
                if len(papers) >= max_results:
                    break
            return papers
        except Exception as e:
            print(f"ClinicalTrials.gov fetch error: {e}")
            return []


class SpringerService:
    """Springer Nature (Springer, Nature, BMC, Palgrave) via the free Meta API.
    Requires a free API key from dev.springernature.com."""

    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        from request_creds import get_cred
        key = get_cred("springer") or getattr(Config, "SPRINGER_API_KEY", "")
        if not key:
            return []
        try:
            clean = re.sub(r"\[[^\]]+\]", "", query).strip() or query
            url = "https://api.springernature.com/meta/v2/json"
            params = {"q": clean, "api_key": key, "p": min(max(max_results, 1), 50)}
            resp = throttled_request(url, params=params).json()
            papers: List[Paper] = []
            for rec in resp.get("records", []):
                doi = rec.get("doi", "") or ""
                urls = rec.get("url", []) or []
                link = (urls[0].get("value", "") if urls else "") or (f"https://doi.org/{doi}" if doi else "")
                papers.append(Paper(
                    source="Springer Nature",
                    id=doi or rec.get("identifier", "") or (rec.get("title", "") or "")[:80],
                    title=rec.get("title", "") or "",
                    abstract=rec.get("abstract", "") or "",
                    url=link,
                ))
                if len(papers) >= max_results:
                    break
            return papers
        except Exception as e:
            print(f"Springer fetch error: {e}")
            return []


class IEEEService:
    """IEEE Xplore metadata via the free API key from developer.ieee.org."""

    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        from request_creds import get_cred
        key = get_cred("ieee") or getattr(Config, "IEEE_API_KEY", "")
        if not key:
            return []
        try:
            clean = re.sub(r"\[[^\]]+\]", "", query).strip() or query
            url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
            params = {"apikey": key, "querytext": clean, "max_records": min(max(max_results, 1), 50), "format": "json"}
            resp = throttled_request(url, params=params).json()
            papers: List[Paper] = []
            for art in resp.get("articles", []):
                doi = art.get("doi", "") or ""
                papers.append(Paper(
                    source="IEEE Xplore",
                    id=doi or str(art.get("article_number", "") or ""),
                    title=art.get("title", "") or "",
                    abstract=art.get("abstract", "") or "",
                    url=art.get("html_url", "") or (f"https://doi.org/{doi}" if doi else ""),
                ))
                if len(papers) >= max_results:
                    break
            return papers
        except Exception as e:
            print(f"IEEE fetch error: {e}")
            return []


class ScopusService:
    """Scopus (Elsevier). Free API key to register at dev.elsevier.com; full
    results need institutional entitlements (on-campus IP or an inst token)."""

    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        from request_creds import get_cred
        key = get_cred("scopus") or getattr(Config, "SCOPUS_API_KEY", "")
        if not key:
            return []
        try:
            clean = re.sub(r"\[[^\]]+\]", "", query).strip() or query
            url = "https://api.elsevier.com/content/search/scopus"
            headers = {"X-ELS-APIKey": key, "Accept": "application/json"}
            params = {"query": clean, "count": min(max(max_results, 1), 25)}
            resp = throttled_request(url, params=params, headers=headers).json()
            entries = (resp.get("search-results", {}) or {}).get("entry", []) or []
            papers: List[Paper] = []
            for ent in entries:
                if ent.get("error"):
                    continue
                doi = ent.get("prism:doi", "") or ""
                sid = ent.get("dc:identifier", "") or ""
                link = (f"https://doi.org/{doi}" if doi
                        else next((l.get("@href", "") for l in ent.get("link", []) if l.get("@ref") == "scopus"), ""))
                papers.append(Paper(
                    source="Scopus",
                    id=doi or sid,
                    title=ent.get("dc:title", "") or "",
                    abstract=ent.get("dc:description", "") or "",
                    url=link,
                ))
                if len(papers) >= max_results:
                    break
            return papers
        except Exception as e:
            print(f"Scopus fetch error: {e}")
            return []


class WebOfScienceService:
    """Web of Science via the Starter API (developer.clarivate.com). Free tier plus
    an institutional subscription. Starter records omit abstracts."""

    @staticmethod
    def fetch(query: str, max_results: int) -> List[Paper]:
        from request_creds import get_cred
        key = get_cred("wos") or getattr(Config, "WOS_API_KEY", "")
        if not key:
            return []
        try:
            clean = re.sub(r"\[[^\]]+\]", "", query).strip() or query
            url = "https://api.clarivate.com/apis/wos-starter/v1/documents"
            headers = {"X-ApiKey": key, "Accept": "application/json"}
            params = {"q": f"TS=({clean})", "limit": min(max(max_results, 1), 50), "db": "WOS", "page": 1}
            resp = throttled_request(url, params=params, headers=headers).json()
            papers: List[Paper] = []
            for hit in resp.get("hits", []):
                doi = ((hit.get("identifiers", {}) or {}).get("doi", "")) or ""
                uid = hit.get("uid", "") or ""
                papers.append(Paper(
                    source="Web of Science",
                    id=doi or uid,
                    title=hit.get("title", "") or "",
                    abstract="",  # Starter API does not return abstracts
                    url=(f"https://doi.org/{doi}" if doi else f"https://www.webofscience.com/wos/woscc/full-record/{uid}"),
                ))
                if len(papers) >= max_results:
                    break
            return papers
        except Exception as e:
            print(f"Web of Science fetch error: {e}")
            return []


class DataAggregator:
    """Aggregates data from all active sources while respecting rate limits."""

    SERVICE_MAP = {
        DataSource.PUBMED.value: PubMedService.fetch,
        DataSource.ARXIV.value: ArXivService.fetch,
        DataSource.BIORXIV.value: BioRxivService.fetch,
        "medRxiv": MedRxivService.fetch,
        "Europe PMC": EuropePMCService.fetch,
        "Semantic Scholar": SemanticScholarService.fetch,
        "OpenAlex": OpenAlexService.fetch,
        "CrossRef": CrossRefService.fetch,
        "DOAJ": DOAJService.fetch,
        "CORE": COREService.fetch,
        "ClinicalTrials.gov": ClinicalTrialsService.fetch,
        "Springer Nature": SpringerService.fetch,
        "IEEE Xplore": IEEEService.fetch,
        "Scopus": ScopusService.fetch,
        "Web of Science": WebOfScienceService.fetch,
    }

    @staticmethod
    def fetch_all(query: str, active_sources: List[str], max_per_source: int = 10, uploaded_files=None, limit: int = None, sort: str = "relevance", year_from: int = None, year_to: int = None):
        """
        Aggregates raw data from all active sources.
        Deduplication is removed to ensure PRISMA counts accurately reflect total records.

        ``sort`` is the selection strategy used when a source matches more than its
        cap ("relevance" = keep the best matches, "recent" = keep the newest).
        ``year_from`` / ``year_to`` restrict by publication year. Both are forwarded
        only to sources whose fetch() accepts them; the rest are unaffected.
        """
        import inspect
        all_papers = []
        source_counts = {}

        search_count = limit if limit is not None else max_per_source
        # Extra kwargs forwarded per source only when its fetch() declares them.
        extra = {"sort": sort, "year_from": year_from, "year_to": year_to}

        for source in active_sources:
            papers = []
            try:
                if source == DataSource.LOCAL_PDF.value:
                    if uploaded_files:
                        papers = PDFService.process_files(uploaded_files)

                elif source in DataAggregator.SERVICE_MAP:
                    fetch_func = DataAggregator.SERVICE_MAP[source]
                    accepted = inspect.signature(fetch_func).parameters
                    kwargs = {k: v for k, v in extra.items() if k in accepted}
                    papers = fetch_func(query, search_count, **kwargs)

                count = len(papers)
                all_papers.extend(papers)
                source_counts[source] = count

            except Exception as e:
                print(f"Error fetching from {source}: {str(e)}")
                source_counts[source] = 0

        if limit is not None:
            return all_papers[:limit], source_counts

        return all_papers, source_counts

    @staticmethod
    def simulate_yield(query: str, active_sources: List[str]) -> Dict[str, int]:
        """
        Returns the absolute total of papers matching the query in each database 
        without downloading full records.
        """
        from utils import QueryCleaner
        results = {}
        clean_query = QueryCleaner.clean_for_general_search(query)
        
        for source in active_sources:
            try:
                print(f"Processing source: {source}")
                
                # 1. PubMed & Top Journals
                if source == DataSource.PUBMED.value:
                    Entrez.email = contact_email()
                    
                    # Construct search term for PubMed
                    search_term = query
                    
                    # retmax=0 makes the request instant as no records are downloaded
                    try:
                        handle = Entrez.esearch(db="pubmed", term=search_term, retmax=0)
                        record = Entrez.read(handle)
                        if record is not None:
                            count = record.get("Count", 0)
                            results[source] = int(count) if str(count).isdigit() else 0
                        else:
                            results[source] = 0
                    except Exception as e:
                        print(f"PubMed search error for {source}: {e}")
                        results[source] = 0

                # 2. ArXiv (Parsing OpenSearch XML for total results)
                elif source == DataSource.ARXIV.value:
                    try:
                        url = f"{Config.ARXIV_API_URL}?search_query=all:{clean_query}&max_results=0"
                        resp = throttled_request(url)
                        root = ET.fromstring(resp.content)
                        
                        # Debug: Print the XML response to see what we're getting
                        print(f"ArXiv XML response for query '{clean_query}':")
                        print(resp.text[:500] + "..." if len(resp.text) > 500 else resp.text)
                        
                        # ArXiv uses opensearch namespace for result counts
                        ns = {'os': 'http://a9.com/-/spec/opensearch/1.1/'}
                        total_node = root.find('os:totalResults', ns)
                        
                        if total_node is not None and total_node.text:
                            total_text = total_node.text.strip()
                            print(f"ArXiv totalResults text: '{total_text}'")
                            
                            # Try to convert to int, handle non-numeric gracefully
                            try:
                                results[source] = int(total_text)
                            except ValueError:
                                # If not a pure number, try to extract digits
                                digits = re.findall(r'\d+', total_text)
                                if digits:
                                    results[source] = int(''.join(digits))
                                else:
                                    print(f"Could not extract numeric count from: '{total_text}'")
                                    results[source] = 0
                        else:
                            print("ArXiv totalResults node not found")
                            results[source] = 0
                    except Exception as e:
                        print(f"ArXiv search error for {source}: {e}")
                        import traceback
                        traceback.print_exc()
                        results[source] = 0

                # 3. BioRxiv (Metadata-only request)
                elif source == DataSource.BIORXIV.value:
                    try:
                        # BioRxiv API provides counts for a date range in the 'messages' field
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=Config.BIORXIV_LOOKBACK_DAYS)
                        date_str = f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
                        
                        # We use the 'details' endpoint which returns a 'messages' count for the range
                        url = f"{Config.BIORXIV_API_URL}/{date_str}/0"
                        resp = throttled_request(url).json()
                        
                        # Note: BioRxiv count is for the time window; keyword filtering 
                        # for counts usually requires fetching, so this is an upper-bound estimate.
                        messages = resp.get('messages', [])
                        if messages:
                            total = messages[0].get('total', 0)
                            results[source] = int(total) if str(total).isdigit() else 0
                        else:
                            results[source] = 0
                    except Exception as e:
                        print(f"BioRxiv search error for {source}: {e}")
                        results[source] = 0

                # 3b. Europe PMC (Accessing 'hitCount' in JSON response)
                elif source == "Europe PMC":
                    try:
                        # Europe PMC accepts free-text + a subset of fielded operators.
                        # Strip PubMed-only tags ([Mesh], [tiab]) before sending.
                        # Must use the SAME translation as the fetch, or the
                        # planned yield describes a different query from the one
                        # screening actually runs.
                        epmc_query = to_europepmc_query(query)
                        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                        params = {
                            "query": epmc_query or query,
                            "format": "json",
                            "pageSize": 1,
                            "resultType": "lite",
                        }
                        resp = throttled_request(url, params=params).json()
                        total = resp.get("hitCount", 0)
                        results[source] = int(total) if str(total).isdigit() else 0
                    except Exception as e:
                        print(f"Europe PMC search error for {source}: {e}")
                        results[source] = 0

                # 4. Semantic Scholar (Accessing 'total' in JSON response)
                elif source == "Semantic Scholar":
                    try:
                        params = {'query': query, 'limit': 0} 
                        headers = {'x-api-key': Config.SEMANTIC_SCHOLAR_KEY} if Config.SEMANTIC_SCHOLAR_KEY else {}
                        url = "https://api.semanticscholar.org/graph/v1/paper/search"
                        resp = throttled_request(url, params=params, headers=headers).json()
                        total = resp.get('total', 0)
                        results[source] = int(total) if str(total).isdigit() else 0
                    except Exception as e:
                        print(f"Semantic Scholar search error for {source}: {e}")
                        results[source] = 0

                # 4b. OpenAlex (uses meta.count from results)
                elif source == "OpenAlex":
                    try:
                        oa_query = re.sub(r"\[[^\]]+\]", "", query).strip()
                        url = "https://api.openalex.org/works"
                        params = {"search": oa_query or query, "per_page": 1, "select": "id"}
                        resp = throttled_request(url, params=params).json()
                        count = (resp.get("meta") or {}).get("count", 0)
                        results[source] = int(count) if str(count).isdigit() else 0
                    except Exception as e:
                        print(f"OpenAlex count error: {e}")
                        results[source] = 0

                # 4c. CrossRef (uses message.total-results)
                elif source == "CrossRef":
                    try:
                        cr_query = re.sub(r"\[[^\]]+\]", "", query).strip()
                        url = "https://api.crossref.org/works"
                        params = {"query": cr_query or query, "rows": 0}
                        headers = {"User-Agent": "EvidenceEngine/1.0"}
                        resp = throttled_request(url, params=params, headers=headers).json()
                        total = (resp.get("message") or {}).get("total-results", 0)
                        results[source] = int(total) if str(total).isdigit() else 0
                    except Exception as e:
                        print(f"CrossRef count error: {e}")
                        results[source] = 0

                # 4d. medRxiv (via Europe PMC preprint source filter)
                elif source == "medRxiv":
                    try:
                        clean = re.sub(r"\[[^\]]+\]", "", query).strip()
                        epmc_query = f"({clean}) AND SRC:PPR AND (publisher:medRxiv OR journal:medRxiv)"
                        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
                        params = {"query": epmc_query, "format": "json", "pageSize": 1}
                        resp = throttled_request(url, params=params).json()
                        total = resp.get("hitCount", 0)
                        results[source] = int(total) if str(total).isdigit() else 0
                    except Exception as e:
                        print(f"medRxiv count error: {e}")
                        results[source] = 0

                # 4e. DOAJ (uses 'total' in JSON response)
                elif source == "DOAJ":
                    try:
                        from urllib.parse import quote
                        clean = re.sub(r"\[[^\]]+\]", "", query).strip()
                        url = f"https://doaj.org/api/v2/search/articles/{quote(clean or query)}"
                        params = {"pageSize": 1}
                        resp = throttled_request(url, params=params).json()
                        total = resp.get("total", 0)
                        results[source] = int(total) if str(total).isdigit() else 0
                    except Exception as e:
                        print(f"DOAJ count error: {e}")
                        results[source] = 0

                # 5. CORE (Accessing 'totalHits' in JSON response)
                elif source == "CORE":
                    try:
                        headers = {"Authorization": f"Bearer {Config.CORE_API_KEY}"} if Config.CORE_API_KEY else {}
                        # CORE v3 uses 'limit: 0' for count-only queries
                        payload = {"q": query, "limit": 0}
                        resp = throttled_request(Config.CORE_API_URL, params=payload, headers=headers, method="POST").json()
                        total_hits = resp.get('totalHits', 0)
                        results[source] = int(total_hits) if str(total_hits).isdigit() else 0
                    except Exception as e:
                        print(f"CORE search error for {source}: {e}")
                        results[source] = 0

                # 6. Local PDFs. This is a yield ESTIMATE for remote databases and
                # runs with no upload context, so it cannot know how many local
                # files the user has staged. The caller supplies that count.
                elif source == DataSource.LOCAL_PDF.value:
                    results[source] = 0

                else:
                    print(f"Unknown source: {source}")
                    results[source] = 0

            except Exception as e:
                # This is the outer catch-all for any unexpected errors
                print(f"Unexpected error simulating yield for {source}: {e}")
                import traceback
                traceback.print_exc()  # Print full stack trace
                results[source] = 0
                
        return results

    @staticmethod
    def get_total_counts(query: str, sources: List[str]) -> Dict[str, int]:
        """Fetches only the total result count for a query from selected sources."""
        from utils import QueryCleaner
        results = {}
        clean_query = QueryCleaner.clean_for_general_search(query)
        
        for source in sources:
            try:
                # PubMed: Use esearch with retmax=0
                if source == DataSource.PUBMED.value:
                    Entrez.email = contact_email()
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
                print(f"Could not fetch count for {source}: {e}")
                results[source] = 0
                
        return results

    @staticmethod
    def get_all_counts(query: str, selected_sources: List[str] = None) -> Dict[str, int]:
        """Hits the 'count' endpoints of APIs to quickly gauge yield."""
        results = {}
        
        # Default to all sources if none specified
        if selected_sources is None:
            selected_sources = ["PubMed", "arXiv", "Semantic Scholar"]
        
        # Only query the selected sources
        
        # PubMed Count
        if "PubMed" in selected_sources:
            try:
                Entrez.email = contact_email()
                
                # Use modern Entrez API (esearch instead of deprecated egquery)
                handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
                record = Entrez.read(handle)
                
                # Get count from esearch result
                count = int(record["Count"]) if record.get("Count", "0").isdigit() else 0
                results["PubMed"] = count
            except Exception as e:
                print(f"❌ PubMed count error: {e}")
                results["PubMed"] = 0
        
        # arXiv Count
        if "arXiv" in selected_sources:
            try:
                from utils import QueryCleaner
                clean_arxiv_query = QueryCleaner.clean_for_general_search(query)
                url = f"{Config.ARXIV_API_URL}?search_query=all:{urllib.parse.quote(clean_arxiv_query)}&max_results=0"
                resp = throttled_request(url)
                
                # Validate response before parsing
                if resp.status_code != 200:
                    print(f"⚠️ arXiv count unavailable (status {resp.status_code})")
                    results["arXiv"] = 0
                elif not resp.content.strip().startswith(b'<'):
                    print("⚠️ arXiv returned non-XML response")
                    results["arXiv"] = 0
                else:
                    root = ET.fromstring(resp.content)
                    total_results = root.find('{http://a9.com/-/spec/opensearch/1.1/}totalResults').text
                    results["arXiv"] = int(total_results) if total_results and total_results.isdigit() else 0
            except Exception as e:
                print(f"❌ arXiv count error: {e}")
                results["arXiv"] = 0

        # Semantic Scholar Count
        if "Semantic Scholar" in selected_sources:
            try:
                url = Config.SEMANTIC_SCHOLAR_URL
                params = {'query': query, 'limit': 1}
                resp = throttled_request(url, params=params).json()
                results["Semantic Scholar"] = int(resp.get('total', 0))
            except Exception as e:
                print(f"❌ Semantic Scholar count error: {e}")
                results["Semantic Scholar"] = 0
        
        return results