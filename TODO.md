# Evidence-Engine: PICO Refinement UX Improvements

## Phase 1: Guided Feedback & Discovery

- [ ] AI-guided field-by-field refinement
  - After draft PICO, AI reviews each element and flags weak/vague spots
  - Suggests specific alternatives (e.g., "Outcome is vague — incidence, prevalence, or mortality?")
  - User picks from suggestions or writes their own
- [ ] Outcome discovery from literature
  - Use the quick 5-paper fetch to identify outcomes actually measured in the literature
  - Present feasible outcomes as selectable options before user commits
  - Helps bridge the gap between what the user wants and what's available in the evidence

## Phase 2: Criteria UX

- [ ] Structured criteria editing
  - Replace comma-separated text areas with add/remove item lists
  - Each criterion is individually editable, deletable, and reorderable
  - AI can suggest new criteria one at a time
  - Visual distinction between AI-suggested and user-added criteria

## Phase 3: Iterative Tightening

- [ ] Lightweight AI reconciliation on PICO edits
  - Editing a PICO field triggers targeted AI check (not a full re-run)
  - AI suggests how inclusion/exclusion criteria and search string should adapt
  - All-way reconciliation: changes to any element propagate suggestions to related elements
