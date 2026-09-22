"""Translate a PubMed search string into each database's own query language.

Every search this platform builds is written in PubMed syntax, because that is
the richest of the dialects and the one the block-decomposition strategy targets
(`AIService.generate_mesh_query`). Sending that string anywhere else is not a
search for the same question:

  * Unqualified terms mean different things. In Europe PMC a bare word matches
    FULL TEXT, so a query that looks narrow silently matches any paper that
    mentions the phrase once in its references.
  * Several engines have no boolean parser at all. CrossRef and Semantic
    Scholar's default endpoint treat `("a"[tiab] OR "b"[tiab]) AND ...` as a bag
    of characters to relevance-rank, so the AND is decorative.
  * Field tags left in place are matched as literal text. `"Artificial
    Intelligence"[Mesh]` sent to a keyword engine searches for the string
    "Mesh".

The previous approach, `re.sub(r"\\[[^\\]]+\\]", "", query)`, deleted the tags and
sent the rest. That keeps the booleans but drops the field scoping, which
quietly widens every block from title/abstract to whatever the target engine
searches by default.

So the query is PARSED once into an intermediate form and each database gets its
own emitter. The parse is the valuable part: once a query is blocks of terms, a
dialect is a small function, and a dialect that cannot express something (no
boolean support) degrades in a way we choose rather than by accident.

Sources for each dialect are cited on its emitter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Intermediate form
# ---------------------------------------------------------------------------

@dataclass
class Block:
    """One AND-ed concept: an OR of free-text terms plus controlled headings.

    Mirrors the structure `generate_mesh_query` builds, where each block is one
    facet of the question and the AND between blocks is what makes the search
    specific.
    """
    terms: List[str] = field(default_factory=list)   # free text (from [tiab])
    mesh: List[str] = field(default_factory=list)    # controlled headings ([Mesh])

    @property
    def all_terms(self) -> List[str]:
        """Free text first, then headings, de-duplicated case-insensitively.

        MeSH headings are usable as free text in engines with no controlled
        vocabulary ("Electronic Health Records" is also a phrase people write),
        so they are kept rather than dropped when a dialect has no MeSH field.
        """
        seen, out = set(), []
        for t in list(self.terms) + list(self.mesh):
            k = t.lower()
            if k not in seen:
                seen.add(k)
                out.append(t)
        return out


# Tags that mean "title and/or abstract" across the PubMed-ish dialects.
_TIAB_TAGS = {"tiab", "title/abstract", "tw", "text word", "textword", "ti", "title", "ab", "abstract"}
_MESH_TAGS = {"mesh", "mesh terms", "mh", "mesh:noexp", "majr"}


def _split_top_level(s: str, op: str) -> List[str]:
    """Split on `op` only at parenthesis depth zero.

    A naive `s.split(" AND ")` would cut inside a nested group and produce
    fragments that no longer parse.
    """
    parts, depth, cur, i = [], 0, [], 0
    pat = f" {op} "
    while i < len(s):
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if depth == 0 and s[i:i + len(pat)].upper() == pat:
            parts.append("".join(cur))
            cur = []
            i += len(pat)
            continue
        cur.append(c)
        i += 1
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _unwrap(s: str) -> str:
    """Strip one layer of parentheses when they wrap the whole expression."""
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        for i, c in enumerate(s):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and i < len(s) - 1:
                    return s            # the parens close early; not a wrapper
        s = s[1:-1].strip()
    return s


def parse_pubmed_query(query: str) -> List[Block]:
    """Parse a PubMed query into AND-ed blocks of OR-ed terms.

    Deliberately forgiving: this runs on model-generated strings, and a query
    that parses imperfectly still beats one sent verbatim to an engine that
    cannot read it. Anything unrecognised is kept as a free-text term rather
    than discarded, because dropping a term changes the question.
    """
    q = (query or "").strip()
    if not q:
        return []

    blocks: List[Block] = []
    for raw_block in _split_top_level(q, "AND"):
        b = Block()
        for raw_term in _split_top_level(_unwrap(raw_block), "OR"):
            t = _unwrap(raw_term).strip()
            if not t:
                continue
            m = re.match(r'^\s*(".*?"|\S+?)\s*\[([^\]]+)\]\s*$', t, flags=re.S)
            if m:
                value, tag = m.group(1).strip('"').strip(), m.group(2).strip().lower()
                if not value:
                    continue
                if tag in _MESH_TAGS:
                    b.mesh.append(value)
                elif tag in _TIAB_TAGS:
                    b.terms.append(value)
                else:
                    # An unknown tag (publication type, language, date). Keep the
                    # value as free text: it is still a word the user asked for.
                    b.terms.append(value)
            else:
                b.terms.append(t.strip('"').strip())
        if b.terms or b.mesh:
            blocks.append(b)
    return blocks


def _q(t: str) -> str:
    """Quote a term if it contains whitespace or punctuation an engine splits on."""
    return f'"{t}"' if re.search(r"[\s\-/:,]", t) else t


def _join(blocks: List[Block], wrap, sep_or: str = " OR ", sep_and: str = " AND ") -> str:
    """Render blocks as (wrap(t) OR ...) AND (wrap(t) OR ...)."""
    out = []
    for b in blocks:
        terms = [wrap(t) for t in b.all_terms if t.strip()]
        if terms:
            out.append("(" + sep_or.join(terms) + ")")
    return sep_and.join(out)


def _keywords(blocks: List[Block], per_block: int = 2, limit: int = 8) -> str:
    """Flatten to a keyword string for engines with no boolean support.

    Takes the first terms of EACH block rather than the first terms overall, so
    every facet of the question is represented. Taking them in order would spend
    the whole budget on the first block and lose the very terms that make the
    question specific.
    """
    # Interleaved round-robin, not block-by-block. Taking each block's terms in
    # turn and then truncating spends the budget on the early blocks and cuts
    # the last one entirely, which is the same "the specific term vanished"
    # failure this whole module exists to prevent: with four blocks and a limit
    # of six, "dental" was dropped every time.
    out: List[str] = []
    for round_i in range(per_block):
        for b in blocks:
            terms = b.all_terms
            if round_i < len(terms):
                out.append(terms[round_i])
    seen, dedup = set(), []
    for t in out:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            dedup.append(t)
    # Never cut below one term per block: a facet with no representation makes
    # the search a different question.
    return " ".join(_q(t) for t in dedup[:max(limit, len(blocks))])


# ---------------------------------------------------------------------------
# Emitters, one per database.
#
# Each is named for the dialect it produces and cites the syntax reference it
# was written against. Where a dialect cannot express the query faithfully, the
# docstring says what is lost, because the caller needs to know a search is
# broader than it looks.
# ---------------------------------------------------------------------------

def _cap(blocks: List[Block], n: int) -> List[Block]:
    """Trim each block's term list. Free-text terms are ordered most-specific
    first by the generator, so truncation drops the weakest synonyms."""
    return [Block(terms=b.terms[:n], mesh=b.mesh[:max(1, n // 4)]) for b in blocks]


def to_scopus_query(query: str, max_terms: int = 20) -> str:
    """Scopus: TITLE-ABS-KEY(...) AND TITLE-ABS-KEY(...).

    TITLE-ABS-KEY is the closest equivalent to PubMed's [tiab], covering title,
    abstract and keywords. Loose phrases take double quotes; braces would force
    an exact-character match and would lose plural and hyphenation variants.
    MeSH headings become free-text terms, since Scopus indexes its own thesaurus
    rather than MeSH.

    Ref: https://dev.elsevier.com/sc_search_tips.html
    """
    blocks = _cap(parse_pubmed_query(query), max_terms)
    return _join(blocks, lambda t: f'TITLE-ABS-KEY({_q(t)})')


def to_wos_query(query: str, max_terms: int = 20) -> str:
    """Web of Science: TS=(...) AND TS=(...).

    TS is Topic: title, abstract, author keywords and Keywords Plus. WoS applies
    quotes per phrase inside the TS group, so one TS= per block keeps the group
    boolean intact rather than emitting TS= per term.

    Ref: https://webofscience.zendesk.com/hc/en-us/articles/20016122409105-Search-Operators
    """
    blocks = _cap(parse_pubmed_query(query), max_terms)
    out = []
    for b in blocks:
        terms = [_q(t) for t in b.all_terms if t.strip()]
        if terms:
            out.append("TS=(" + " OR ".join(terms) + ")")
    return " AND ".join(out)


def to_ieee_query(query: str, max_terms: int = 10) -> str:
    """IEEE Xplore: plain boolean over all indexed metadata.

    The Metadata API's querytext supports AND / OR / NOT, quoted phrases and
    parentheses, but documents no field-qualified syntax, so blocks cannot be
    scoped to title/abstract and the search is correspondingly broader. Terms
    are capped hard because querytext goes in the URL and IEEE rejects
    over-long requests.

    Ref: https://developer.ieee.org/docs/read/metadata_api_details/Leveraging_Boolean_Logic
    """
    blocks = _cap(parse_pubmed_query(query), max_terms)
    return _join(blocks, _q)


def to_springer_query(query: str, max_terms: int = 12) -> str:
    """Springer Nature: boolean over free text.

    The q parameter supports AND / OR / NOT / NEAR and field prefixes such as
    title: and keyword:. Only title: is a documented text field, and scoping
    every block to titles alone would lose most true hits, so plain terms are
    used: broader than [tiab], but faithful to the boolean structure.

    Ref: https://dev.springernature.com/docs/advanced-querying/complex-queries-boolean-ops/
    """
    blocks = _cap(parse_pubmed_query(query), max_terms)
    return _join(blocks, _q)


def to_openalex_query(query: str, max_terms: int = 12) -> str:
    """OpenAlex: boolean search over title and abstract.

    OpenAlex added boolean support: AND / OR / NOT must be UPPERCASE, and
    parentheses and quoted phrases work. Intended for the
    `title_and_abstract.search` filter, which scopes to the fields [tiab] means;
    the bare `search` parameter also covers full text and is looser.

    Ref: https://docs.openalex.org/api-entities/works/search-works
    """
    blocks = _cap(parse_pubmed_query(query), max_terms)
    return _join(blocks, _q)


def to_arxiv_query(query: str, max_terms: int = 8) -> str:
    """arXiv: (abs:"x" OR abs:"y") AND (abs:"z").

    arXiv uses ANDNOT rather than NOT, and prefixes fields as ti:, abs:, au:,
    all:. abs: is used because it is the closest to [tiab]; `all:` would also
    match the full text and comments.

    Ref: https://info.arxiv.org/help/api/user-manual.html
    """
    blocks = _cap(parse_pubmed_query(query), max_terms)
    return _join(blocks, lambda t: f'abs:{_q(t)}')


def to_doaj_query(query: str, max_terms: int = 12) -> str:
    """DOAJ: Elasticsearch query-string syntax over bibjson fields.

    Fields use JSON dot notation, so title and abstract are bibjson.title and
    bibjson.abstract. Wildcards and regexes are disabled by DOAJ for
    performance, which costs nothing here because the generator emits explicit
    variants instead of stems.

    Ref: https://doaj.org/api/v4/docs
    """
    blocks = _cap(parse_pubmed_query(query), max_terms)
    return _join(blocks, lambda t: f"(bibjson.title:{_q(t)} OR bibjson.abstract:{_q(t)})")


def to_core_query(query: str, max_terms: int = 12) -> str:
    """CORE: Elasticsearch-style field queries over title and abstract.

    CORE v3 accepts AND / OR / NOT and field-qualified terms such as
    title:coronavirus. Unqualified terms also match full text, so scoping is
    what keeps a block from matching a passing mention in a reference list.

    Ref: https://api.core.ac.uk/docs/v3
    """
    blocks = _cap(parse_pubmed_query(query), max_terms)
    return _join(blocks, lambda t: f"(title:{_q(t)} OR abstract:{_q(t)})")


def to_ctgov_query(query: str, max_terms: int = 10) -> str:
    """ClinicalTrials.gov: Essie expression syntax.

    query.term accepts plain terms, quoted phrases, parentheses and
    AND / OR / NOT, alongside AREA[Field] clauses. No AREA[] is emitted: the
    blocks describe topic, and Essie already searches the study text.

    Ref: https://clinicaltrials.gov/data-api/about-api/search-areas
    """
    blocks = _cap(parse_pubmed_query(query), max_terms)
    return _join(blocks, _q)


# --- Engines with no boolean parser ----------------------------------------
# These relevance-rank a bag of words. Sending boolean syntax makes the
# operators part of the text being matched, so the query is flattened instead
# and the loss is explicit rather than silent.

def to_crossref_query(query: str) -> str:
    """CrossRef: free-text keywords for query.bibliographic.

    CrossRef's search has no boolean parser; every query.* parameter is scored
    as free text. AND-ing is therefore impossible and the result is a recall-
    oriented keyword search that must be screened.

    Ref: https://api.crossref.org/swagger-ui/index.html
    """
    return _keywords(parse_pubmed_query(query))


def to_semantic_scholar_query(query: str) -> str:
    """Semantic Scholar: free-text keywords for /paper/search.

    The relevance endpoint takes a plain phrase. (The separate /paper/search/bulk
    endpoint does accept +, | and -, but it returns a different result shape and
    ranking, so switching endpoints is a bigger change than a translation.)

    Ref: https://api.semanticscholar.org/api-docs/graph
    """
    return _keywords(parse_pubmed_query(query))


def to_preprint_query(query: str) -> str:
    """bioRxiv / medRxiv: free-text keywords.

    Their public APIs are content-feed endpoints keyed by date, DOI or server
    rather than a query language, so whatever search exists upstream is
    keyword-based.

    Ref: https://api.biorxiv.org/
    """
    return _keywords(parse_pubmed_query(query), per_block=2, limit=6)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Keyed by the source names the UI uses (ALL_SOURCES in apiClient.ts).
def to_openreview_query(query: str, max_terms: int = 12) -> str:
    """OpenReview: a term search, not a boolean one.

    `/notes/search` takes `term` with `type=terms`, which scores documents
    against the words given. It has no field tags and no AND / OR, so a PubMed
    string sent verbatim would be searched for its brackets and its `[tiab]`
    markers as if they were words.

    Flattening to keywords across every block is what keeps the question's
    facets represented: taking the first terms overall would spend the budget
    on the first concept and drop the one that makes the question specific.

    Quotation marks are stripped. Measured against the live API: a term of
    several quoted phrases returns ZERO results, while the same words unquoted
    return the expected matches. Phrase precision is lost, but `type=terms`
    scores by term overlap regardless, so the quotes were buying nothing and
    costing everything.

    Ref: https://docs.openreview.net/reference/api-v2
    """
    blocks = _cap(parse_pubmed_query(query), max_terms)
    flat = _keywords(blocks, per_block=2, limit=10) or query
    words = flat.replace('"', " ").replace("'", " ")
    return " ".join(words.split())


_TRANSLATORS = {
    "OpenReview": to_openreview_query,
    "Scopus": to_scopus_query,
    "Web of Science": to_wos_query,
    "IEEE Xplore": to_ieee_query,
    "Springer Nature": to_springer_query,
    "OpenAlex": to_openalex_query,
    "arXiv": to_arxiv_query,
    "DOAJ": to_doaj_query,
    "CORE": to_core_query,
    "ClinicalTrials.gov": to_ctgov_query,
    "CrossRef": to_crossref_query,
    "Semantic Scholar": to_semantic_scholar_query,
    "bioRxiv": to_preprint_query,
    "medRxiv": to_preprint_query,
}


def translate_query(query: str, source: str) -> str:
    """Render `query` (PubMed syntax) in `source`'s dialect.

    PubMed gets the string unchanged, since that is what it was written in.
    Europe PMC keeps its existing dedicated translator in data_services, which
    encodes findings about that API this generic path does not model.

    A source with no translator is returned unchanged rather than mangled: an
    unknown engine is more likely to be PubMed-ish than to want a guess.
    """
    if source == "PubMed" or not query:
        return query
    if source == "Europe PMC":
        from data_services import to_europepmc_query
        return to_europepmc_query(query)
    fn = _TRANSLATORS.get(source)
    if not fn:
        return query
    try:
        out = fn(query)
    except Exception as e:                      # never fail a search on a translation
        print(f"[translate_query] {source}: {e}")
        return query
    return out or query
