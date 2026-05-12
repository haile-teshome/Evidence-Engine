# Screening Architecture Benchmark

A standalone harness for evaluating LLM screening architectures (and the models that run them) against gold-standard systematic-review decisions.

## What gets compared

**Architectures** (in `architectures/`):

| Slug | Approach |
|---|---|
| `single_combined` | Baseline — one LLM call evaluates all criteria at once (matches the current app behavior). |
| `cascade_triage` | Title-only YES/NO/MAYBE pass first, then full per-criterion pass on YES/MAYBE only. |
| `per_criterion_cot` | One focused LLM call per inclusion/exclusion criterion, with chain-of-thought reasoning. |
| `decompose_match` | Call A extracts the paper's PICO from the abstract; Call B matches extracted PICO against user criteria. |
| `self_consistency` | Default one-shot; on hedging/uncertain outputs, re-sample 3× and majority-vote. |
| `single_agent_tools` | LLM with a tool box (search_abstract, lookup_pico, count_match) reasons in steps. |
| `multi_agent` | Reviewer → Critic → Adjudicator. Three calls with distinct roles; adjudicator decides. |
| `leads_native` | The LEADS paper's exact PICO-element prompt + 4-way YES/PARTIAL/NO/UNCERTAIN verdict + averaged-score aggregation. |
| `leads_multi_persona` | Three expert personas (Methodologist, Domain Specialist, Information Specialist) each apply the LEADS prompt; aggregate by average / majority / any. |
| `cascade_leads_strict` | Two-stage: LEADS-native high-recall pass, then strict per-criterion confirmation on the includes. |
| `leads_native_fewshot` | LEADS prompt prepended with 3 worked-include + 3 worked-exclude exemplars (van_Dis_2020 specific). |

**Models** (in `models.py`):

| Tier | Default name | Override env var |
|---|---|---|
| `small` | `llama3.2:3b` | `BENCH_MODEL_SMALL` |
| `medium` | `qwen2.5:7b` | `BENCH_MODEL_MEDIUM` |
| `specialized` | `medgemma:27b` | `BENCH_MODEL_SPECIALIZED` |
| `large` | `qwen3.5:27b` | `BENCH_MODEL_LARGE` |
| `leads` | `hf.co/mradermacher/leads-mistral-7b-v1-GGUF:latest` | `BENCH_MODEL_LEADS` |
| `leading` | `claude-sonnet-4-6` (or `gpt-4o`, configurable) | `BENCH_MODEL_LEADING` |

Models routed by name like the main app: `gpt*`→OpenAI, `claude*`→Anthropic, `gemini*`→Google, anything else→local Ollama at `http://localhost:11434`.

## Datasets

Each dataset has a YAML describing PICO + inclusion/exclusion criteria, plus a CSV of `paper_id,title,abstract,label` where `label ∈ {0,1}`.

- **`sample`** — a tiny hand-crafted dataset (12 papers, metformin vs placebo for T2DM) bundled in `data/sample/` so you can smoke-test without any downloads.
- **`synergy/<review_name>`** — drop a SYNERGY-format CSV in `data/synergy/<name>/records.csv` and a hand-written `criteria.yaml`. SYNERGY datasets are available at <https://github.com/asreview/synergy-dataset>.
- **`custom/<your_name>`** — same layout as `synergy`; bring your own labelled SR.

To download a SYNERGY review and stratify-sample it, see `scripts/download_synergy.py`.

## Run

```bash
cd benchmark
pip install -r requirements.txt

# Smoke test (uses the bundled tiny dataset, current Ollama models)
python run_benchmark.py --datasets sample --architectures single_combined cascade_triage --models small medium

# Full matrix on local Ollama
python run_benchmark.py --datasets sample synergy/<review> --architectures all --models all

# A realistic run: 4 architectures × 2 tiers on a SYNERGY review, with 3 repeats + field stratification
python run_benchmark.py \
    --datasets synergy/van_Dis_2020 \
    --architectures leads_native single_combined cascade_leads_strict leads_multi_persona \
    --models leads medium \
    --repeat 3 --workers 4 --field-stratify
```

The runner produces, per `reports/<run_id>/`:

- `predictions.csv` — every paper × architecture × model decision
- `metrics.csv` — per-rep metrics (sensitivity, specificity, F1, MCC, accuracy, WSS@95)
- `metrics_aggregated.csv` — averaged metrics across repeats
- `bootstrap_ci.csv` / `pairwise_mcnemar.csv` / `interrater_panel.csv` / `field_stratified.csv` / `stability.csv` — statistical tables
- `summary.md` — human-readable comparison

## Post-processing scripts (in `scripts/`)

| Script | Purpose |
|---|---|
| `download_synergy.py` | Pull a SYNERGY review's IDs + titles/abstracts from OpenAlex; stratify-sample to N papers. |
| `threshold_sweep.py` | Sweep the LEADS aggregate-score threshold on an existing predictions.csv to find the recall/specificity Pareto. |
| `cross_model_ensemble.py` | Build voting / averaging ensembles across (architecture, tier) cells in existing predictions. |
| `cost_recall_curves.py` | Reframe predictions as operational SR-screening metrics (WSS@95, papers-to-95%-recall, screening burden). |
| `external_validity_analysis.py` | Per-dataset threshold sweep across multiple SYNERGY datasets. |
| `stability_analysis.py` | Cross-replicate stability (Cohen's κ, Fleiss κ, per-paper consistency) on repeat runs. |
| `fewshot_eval.py` | Compare few-shot LEADS to baseline LEADS on the held-out 282-paper subset. |
| `make_figures.py` | Generate all benchmark figures to `reports/figures/`. |
| `merge_runs.py` | Merge multiple `reports/<run_id>/` directories into one. |
| `configure_ollama.sh` | Helper to pull recommended Ollama models. |
| `launch_optimized.sh` | Convenience launcher for the typical full sweep. |

## Notes

- API-keyed models (Claude / GPT) require the same env vars as the main app (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). If a key is missing, that model tier is skipped with a warning.
- All architectures share the same `BaseChatModel` so the only thing that changes per cell is the prompting strategy.
- Per-paper concurrency is configurable via `--workers`; Ollama serializes anyway with one model loaded.
- Sensible default: `--workers 4` for Ollama on an M-series Mac; tune based on memory pressure.
