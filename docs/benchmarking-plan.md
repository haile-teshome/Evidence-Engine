# Per-stage benchmarking plan for the JAMIA revision (public datasets only)

Reviewer concern: only 1 of the 10 pipeline stages (abstract screening) was quantitatively benchmarked.

Constraint: every stage is evaluated against a **known, publicly available, downloadable, citable dataset**. Meta-analysis / effect-size pooling is excluded from the quantitative suite (no clean public gold pooling benchmark). All datasets below were verified by fetching the actual repo, OSF node, or HuggingFace page; sizes and licenses are as confirmed there.

## Master table (public datasets only; one row per stage)

| # | Stage (platform tab) | Public dataset | Eval size | Access | License | Metric | Status |
|---|---|---|---|---|---|---|---|
| 1 | Question & PICO | EBM-NLP (gold expert test); PICO-Corpus | 191 expert-labeled abstracts (of 4,993); +1,011 | github.com/bepnye/EBM-NLP; github.com/sociocom/PICO-Corpus | none stated (flag) | Entity/token F1 per element | Harness started |
| 2 | Search design | CLEF eHealth TAR 2017/18/19; ielab seed collection | 30 / 30 / 31 test topics; 40 topics | github.com/CLEF-TAR/tar; github.com/ielab/sysrev-seed-collection | MIT | Query recall vs qrels / screening burden | To build |
| 3 | Deduplication | ASySD (primary); SRA-DM (secondary) | 5 sets, 1,845 to 79,880 records w/ exact dup counts | OSF osf.io/c9evs; Zenodo 7937612 | CC BY 4.0 | Sensitivity, specificity / false-removals | To run |
| 4 | Abstract screening | SYNERGY; CSMeD-abstract | 26 datasets, 169,288 recs / 2,834 incl | github.com/asreview/synergy-dataset | CC0 | Recall, WSS@95, specificity | DONE |
| 5 | Full-text acquisition | SYNERGY/CSMeD DOIs x OpenAlex/Unpaywall OA-status | DOIs from the corpora; OA truth from OpenAlex | s3://openalex/ (no-sign); europepmc.org/downloads | CC0 | Retrieval success / correct-article precision | Compose |
| 6 | Full-text screening | CSMeD-FT | 29 reviews / 636 docs, 43.7% incl | github.com/WojciechKusa/systematic-review-datasets | open | Recall, specificity, per-criterion acc | MOSTLY DONE |
| 7 | Citation snowballing | SYNERGY (embedded OpenAlex graph); ielab seed collection; darrenkjr 27-SR set | 2,834 includes as seeds; 40 topics; 27 SRs | github.com/asreview/synergy-dataset; github.com/ielab/sysrev-seed-collection; github.com/darrenkjr/automated_citation_search_study | CC0 / MIT / open | Recall gain / candidate burden | Compose |
| 8 | Data & table extraction | PubTabNet (tables); Evidence Inference 2.0 (direction); llm-meta-analysis (numeric fields) | 9,115 val tables; ~1,220 test prompts; 110 RCTs / 656 ICO test | github.com/ibm-aur-nlp/PubTabNet; github.com/jayded/evidence-inference; github.com/hyesunyun/llm-meta-analysis | CDLA-Permissive / MIT / Apache-2.0 | TEDS/cell-acc; direction acc; numeric F1 | Tables DONE; rest to run |
| 9 | Risk of bias | ROBoto2 (gold); Trialstreamer (silver, scale) | 245 expert RoB 2 assessments + 1,202 evidence passages; millions (overall only) | github.com/larchlab/ROBoto2; Zenodo 10.5281/zenodo.3767068 | CC BY 4.0 / CC0 | Per-domain F1 + evidence retrieval | To run |

## Per-stage detail

### Stage 1. Question formulation and PICO
- Datasets: **EBM-NLP** (Nye et al. 2018) — 4,993 RCT abstracts; train 4,741 crowd (silver), **test 191 abstracts labeled by 3 medical experts (gold)**; github.com/bepnye/EBM-NLP; no explicit license (PubMed-derived), flag it. **PICO-Corpus** (Mutinda et al. 2022) — 1,011 breast-cancer abstracts, gold BRAT, adds explicit Comparator and numeric-outcome spans; github.com/sociocom/PICO-Corpus; no formal license, flag.
- Metric: entity-level and token-level F1 per P/I/C/O element, exact and partial. Evaluate on the 191-abstract expert test set; train on the crowd split. PICO-Corpus is the cross-domain second set.
- Comparator: published EBM-NLP supervised baselines (LSTM-CRF, SciBERT/PICO-BERT); LEADS PICO numbers.
- Effort: LOW (`eval_pico_clef.py` already started).

### Stage 2. Search strategy design
- Datasets: **CLEF eHealth TAR** (github.com/CLEF-TAR/tar, MIT): 2017 = 50 topics (20 train / 30 test); 2018 = 80 (50 / 30); 2019 = 92 train / 31 test. Each topic ships the original Boolean query, retrieved PMIDs, and relevance qrels at abstract and full-text level (abstracts pulled from PubMed by PMID). **ielab seed collection** (github.com/ielab/sysrev-seed-collection, MIT): 40 topics with seed studies + Boolean query + relevant docs, purpose-built for query *formulation*.
- Metric: recall and precision of the AI query against the qrels, and relative recall versus the topic's expert Boolean query, paired with retrieved-set size (screening burden), as a recall-burden curve.
- Comparator: the expert Boolean strategy shipped with each topic.
- Effort: MEDIUM.

### Stage 3. Deduplication
- Datasets: **ASySD** (Hair et al. 2023, OSF osf.io/c9evs, CC BY 4.0) is the primary gold standard: Diabetes 1,845 records / 1,261 duplicates; Neuroimaging 3,438 / 1,298; Cardiac 8,948 / 3,530; SRSR 53,001 / 16,855; Depression 79,880 / 10,135 (tool archived at Zenodo 7937612; R package camaradesuk/ASySD, GPL-3). **SRA-DM** (Rathbone 2015) is the older secondary set (empyema 1,988 / 799, plus Cytology 1,856, Stroke 1,292, Haematology 1,415; per-set duplicate counts not all reported, and no canonical first-party download, so obtain via the ASySD or 2024 Deduplicator repos).
- Metric: sensitivity, specificity, precision, F1, and the safety metric = count of unique studies wrongly removed.
- Comparator: EndNote, ASySD, Deduklick published numbers on the same sets.
- Effort: LOW.

### Stage 4. Title and abstract screening (DONE)
- Datasets: **SYNERGY** (26 datasets, 169,288 records, 2,834 includes, CC0, github.com/asreview/synergy-dataset; your four subsets 288/292/288/271) and **CSMeD-abstract**.
- Metric: recall, WSS@95%, specificity, kappa stability; abstain-not-reject error analysis.
- Comparator: ASReview; supervised ML.
- Action: consolidate for the paper.

### Stage 5. Full-text acquisition
- No off-the-shelf benchmark exists, so compose from public inputs only: take DOIs from the public SYNERGY and CSMeD records, and use **OpenAlex** (CC0, public S3 `s3://openalex/` with `open_access.oa_status` and `oa_locations` per work; Unpaywall data is delivered through it) as ground truth for whether an open-access full text exists and where. Cross-check fetchability against the **Europe PMC OA subset** (5.7M CC-licensed full texts, europepmc.org/downloads) and **CORE**.
- Metric: retrieval success rate (recall of OA full texts OpenAlex marks obtainable) with CIs; correct-article precision on an audited sample.
- Comparator: the raw OpenAlex/Unpaywall OA hit rate.
- Effort: LOW-MEDIUM (all inputs public and openly licensed; no new annotation).

### Stage 6. Full-text eligibility screening (MOSTLY DONE)
- Dataset: **CSMeD-FT** (github.com/WojciechKusa/systematic-review-datasets; 29 reviews, 636 documents, 43.7% include, real Cochrane criteria, full text).
- Metric: recall, specificity, F1, per-criterion accuracy. Full-text-by-scale interaction already found.
- Comparator: LEADS/MedGemma cross-model.
- Effort: LOW (consolidate).

### Stage 7. Citation snowballing
- Datasets: **SYNERGY** is the primary set because each record carries OpenAlex IDs, DOIs, and a baked-in `referenced_works` citation graph plus cited-by counts, so backward/forward snowballing recall can be measured entirely within the dataset (hold out a fraction of the 2,834 includes and measure recovery). **ielab seed collection** ships explicit snowballing splits (`seed_snowballing_document.tsv`, `screened_snowballing_document.tsv`). **darrenkjr automated_citation_search_study** (27 SRs, github.com/darrenkjr/automated_citation_search_study) provides a published baseline (median recall ~35.8% via one-hop OpenAlex + Semantic Scholar) to beat.
- Metric: recall gain (held-out includes recovered / held-out total), candidate precision, marginal yield beyond database search, paired with screening burden.
- Comparator: citationchaser (public R package) on the same seeds; the darrenkjr baseline.
- Effort: MEDIUM.

### Stage 8. Data and table extraction
- Tables (DONE): **PubTabNet** (val 9,115 labeled tables, CDLA-Permissive-1.0); TEDS 0.977, cell-accuracy 0.958. Silver (auto-generated), so it validates parsing fidelity, not medical correctness.
- Outcome/direction values: **Evidence Inference 2.0** (DeYoung et al. 2020) — 12,616 prompts over 3,346 articles, **test ~1,220 prompts**, MIT, gold (annotated by hired MDs); github.com/jayded/evidence-inference, HF bigbio/evidence_inference. Metric: outcome-direction accuracy and evidence-span F1.
- Numeric result extraction: **llm-meta-analysis** (Yun et al. 2024, MLHC) — 120 RCTs / 699 ICO triplets (183 binary 2x2 tables, 516 continuous mean/SD/N), **test 110 RCTs / 656 ICO**, Apache-2.0, gold; github.com/hyesunyun/llm-meta-analysis. This is the genuinely public numeric-extraction gold set. Metric: exact-match and numeric-tolerance F1 per cell.
- Optional: **RCT-ART** (Apache-2.0) for result-sentence relation extraction.
- Effort: tables DONE; EI 2.0 LOW-MEDIUM; llm-meta-analysis LOW-MEDIUM.

### Stage 9. Risk-of-bias assessment
- Gold dataset (now available): **ROBoto2** (Larch Lab, EMNLP 2025; github.com/larchlab/ROBoto2, CC BY 4.0) — RoB 2, all five domains, with supporting evidence passages: **245 human-expert gold assessments** (plus 276 LLM-assisted), 8,954 signaling-question answers, 1,202 evidence passages, from pediatric RCTs in Cochrane CENTRAL. Small and pediatric-focused; note a README export bug that currently ships ~203 of the 276 LLM samples, but the 245 human assessments are the gold reference.
- Scale reference (silver): **Trialstreamer** (Zenodo 10.5281/zenodo.3767068, CC0/CC-BY) — millions of RCTs but only a single overall abstract-level RoB label, machine-generated. Use for coverage/concordance, not gold accuracy.
- Not public: the RobotReviewer / CDSR training corpus is Cochrane-licensed and not redistributable (confirmed via the RobotReviewer repo and Cochrane data-download terms), so RobotReviewer is a tool comparator only, not a gold dataset.
- Metric: per-domain accuracy/F1 and agreement kappa versus ROBoto2 gold; evidence-passage retrieval precision/recall.
- Comparator: RobotReviewer run on the same RCTs.
- Effort: LOW-MEDIUM on ROBoto2 (small); the honest caveat is size and pediatric scope.

## Cross-cutting statistical and reporting protocol (identical across stages)

- 95% CIs by 1,000-sample bootstrap on every point estimate.
- Paired significance versus the comparator: McNemar for classification, bootstrap difference for continuous metrics.
- External validity: >=2 public datasets or domains per stage where a second public option exists (PICO, search, dedup, extraction, RoB all qualify).
- Per-stage contamination statement (extend the CSMeD contamination note); flag that PubTabNet, EBM-NLP crowd split, and Trialstreamer are silver.
- License note per dataset (EBM-NLP and PICO-Corpus have no formal license; SRA-DM has no canonical first-party download).
- Release predictions, seeds, and dataset versions; one reproducibility appendix.
- One failure-mode analysis per stage, following the abstain-not-reject template.

## Prioritized execution roadmap

- Tier A, write-up only (public data already run): Stage 4 abstract screening, Stage 6 full-text screening, Stage 8 tables. Locks 3 stages.
- Tier B, public gold, low effort (1-2 weeks on the cluster): Stage 1 PICO (EBM-NLP + PICO-Corpus), Stage 3 dedup (ASySD), Stage 8 EI 2.0 + llm-meta-analysis, Stage 9 RoB (ROBoto2). Brings coverage to 7.
- Tier C, public composition: Stage 2 search (CLEF-TAR + ielab), Stage 5 acquisition (SYNERGY/CSMeD DOIs x OpenAlex), Stage 7 snowballing (SYNERGY embedded graph + ielab + darrenkjr baseline). Completes the suite.

Result: every non-excluded stage carries a quantitative result against a verified, downloadable, openly licensed public dataset, with RoB now backed by a real gold set (ROBoto2) rather than tool concordance alone. Meta-analysis pooling is the only stage excluded from the suite.
