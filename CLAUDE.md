# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Evidence-Engine is an AI-powered systematic literature review platform built with Streamlit. It guides researchers through the PICO framework (Population, Intervention, Comparator, Outcome) to formulate research questions, search multiple literature databases, screen papers with AI, and generate PRISMA 2020 flow diagrams.

This is a student's project. Development work by collaborators should happen on **feature branches** and only merge to `main` after manual testing through the Streamlit UI.

## Running the Application

```bash
# Install system dependency
sudo apt install graphviz

# Set up virtual environment with uv
uv venv .venv
uv pip install -r requirements.txt

# Run the app (serves at http://localhost:8501)
streamlit run app.py
```

There are no automated tests, linter, or build step configured.

## Architecture

### Data Flow

```
User enters research question
  → AIService.infer_pico_and_query() extracts PICO + inclusion/exclusion criteria + MeSH query
  → Quick 5-paper fetch for brainstorming summary + refinement suggestions
  → User reviews/edits PICO, criteria, and search string in Strategy Review panel
  → "Run Full Systematic Review" triggers:
      → DataAggregator.fetch_all() searches selected sources
      → Deduplicator.run() removes duplicates (by DOI + normalized title)
      → AIService.screen_paper() evaluates each paper against criteria
      → Results table + PRISMA flow diagram rendered
```

### Module Responsibilities

| File | Role |
|------|------|
| `app.py` | Streamlit entry point. Orchestrates the full workflow: chat history display, input handling, search execution, screening loop, results rendering. |
| `config.py` | All constants: API endpoints, keys, thresholds, defaults. `Config` class + `DataSource` enum. |
| `models.py` | Three dataclasses: `PICOCriteria`, `Paper`, `ScreeningResult`. |
| `state_manager.py` | `SessionState` class wrapping `st.session_state` initialization and reset. |
| `ui_components.py` | `UIComponents` static class: sidebar, results table, PRISMA flow (Graphviz), dedup report. |
| `utils.py` | `AIService` (all LLM calls), `Deduplicator`, `QueryCleaner`. |
| `data_services.py` | `DataAggregator` + individual service classes: `PubMedService`, `TopJournalsService`, `ArXivService`, `BioRxivService`, `SemanticScholarService`, `COREService`, `PDFService`. Rate-limited at 1.1s between API calls. |

### LLM Integration

`AIService.get_model()` returns a LangChain chat model based on the user's sidebar selection:
- **Local**: Ollama (llama3, mistral, phi3, etc.) — default is `llama3`
- **Cloud**: OpenAI (gpt-4o), Anthropic (claude-3.5-sonnet), Google (gemini-1.5-flash)

All AI methods accept a `model_name` parameter and instantiate the model per-call.

### Literature Sources

Seven sources via `DataAggregator`: PubMed (Entrez API), Big 3 Journals (AJE/IJE/EJE via PubMed), arXiv, bioRxiv, Semantic Scholar, CORE, and local PDF upload. Each has its own service class in `data_services.py`.

### Session State

Streamlit's `st.session_state` is the central data store. Key fields: `pico` (PICOCriteria), `query` (MeSH string), `history` (list of past iterations), `results` (DataFrame), `inclusion_list`, `exclusion_list`, `prisma_counts`. Managed via `SessionState` class plus supplemental initialization in `app.py`.

## Key Patterns

- **AI output parsing**: `AIService._extract_json()` uses regex to find JSON in LLM responses. This is fragile — LLM outputs vary. Changes to prompts should be tested against multiple models.
- **PICO refinement loop**: Each user input (or suggestion click) creates a new history entry with PICO, criteria, summary, and suggestions. The Strategy Review panel below the chat lets users edit the latest PICO/criteria before running the full review.
- **Criteria format**: Inclusion/exclusion criteria are stored as lists of strings, displayed and edited as comma-separated text in the UI.
- **PRISMA counts**: Tracked in `st.session_state.prisma_counts` dict, updated after the screening loop completes.

## Known Issues

- `devcontainer.json` references old path `Systematic_Review_Project/app.py` — files are now at repo root.
- API keys are hardcoded in `config.py` (should use environment variables).
- `langchain-google-genai` and `langchain-ollama` are imported in `utils.py` but not listed in `requirements.txt`.
