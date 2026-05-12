# Evidence Engine — Backend API

FastAPI HTTP layer that exposes the existing Python services (PubMed/arXiv/Europe PMC fetchers, AI screening, table extraction) to the React frontend.

## Run

```bash
cd Backend
cp .env.example .env       # fill in ANTHROPIC_API_KEY / OPENAI_API_KEY / etc.
pip install -r requirements.txt
./run_api.sh               # or: uvicorn api:app --reload --port 8000
```

Then start the frontend from the project root:

```bash
pnpm dev                   # vite proxies /api -> http://localhost:8000
```

Open <http://localhost:5173>.

## Architecture

- [streamlit_shim.py](streamlit_shim.py) — replaces `st.session_state` with a plain dict and silences `st.error/st.write/st.spinner` so modules written for `streamlit run` can be imported headlessly. Must be installed before `utils` or `data_services`.
- [api.py](api.py) — FastAPI app, CORS, all routers. Mirrors the TypeScript service contract in [src/app/lib/apiClient.ts](../src/app/lib/apiClient.ts).
- [utils.py](utils.py), [data_services.py](data_services.py), [config.py](config.py), [models.py](models.py) — original Streamlit-era code, untouched.

## Endpoints

| Method | Path                          | Mock contract                        |
|--------|-------------------------------|--------------------------------------|
| POST   | /api/pico/infer               | AIService.inferPicoAndQuery          |
| POST   | /api/pico/formal-question     | AIService.generateFormalQuestion     |
| POST   | /api/pico/summary             | AIService.generateComprehensiveSummary |
| POST   | /api/pico/suggestions         | AIService.getRefinementSuggestions   |
| POST   | /api/pico/adversarial         | AIService.generateAdversarialQuery   |
| POST   | /api/pico/brainstorm          | AIService.getPicoSuggestion          |
| POST   | /api/papers/fetch             | DataAggregator.fetchAll              |
| POST   | /api/papers/dedupe            | Deduplicator.run                     |
| POST   | /api/simulation/yield         | DataAggregator.simulateYield         |
| POST   | /api/simulation/agentic       | AIService.agenticOptimizePerSource   |
| POST   | /api/screen/abstract          | AIService.screenPaperMultiAgent      |
| POST   | /api/screen/abstract-batch    | (batch parallel screening)           |
| POST   | /api/screen/fulltext          | AIService.screenFullTextMultiAgent   |
| POST   | /api/citations                | AIService.fetchCitations             |
| POST   | /api/fulltext/fetch           | AIService.fetchFullText              |
| POST   | /api/extract/text             | AIService.extractFromText            |
| POST   | /api/extract/tables           | AIService.extractTables              |
| POST   | /api/quality/assess           | QualityService.assessPaper           |
| GET    | /api/health                   | Provider key + default-model check   |

Interactive Swagger docs at <http://localhost:8000/docs>.

## Notes

- The frontend's selected model (from the sidebar) is pushed into `apiConfig.model` by [store.tsx](../src/app/lib/store.tsx) and forwarded to AI endpoints.
- API keys live in `.env` on the backend — never shipped to the browser.
- `Deduplicator.run` stays client-side (pure logic) to keep the UI responsive.
- Auth is not yet enforced. Before exposing this server publicly, add Supabase JWT verification — the existing [supabaseClient.ts](../src/app/lib/supabaseClient.ts) already calls `getSession()` for the Supabase-hosted functions; the same token can be forwarded here.
