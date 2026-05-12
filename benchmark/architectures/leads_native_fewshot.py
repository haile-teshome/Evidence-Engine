"""LEADS-native with 3 INCLUDE + 3 EXCLUDE few-shot exemplars prepended.

Same LEADS prompt + 4-way verdict + score aggregation as leads_native, but
the prompt is prefixed with worked examples drawn from van_Dis_2020 so the
model can pattern-match what eligibility decisions look like for clear
includes vs. clear excludes.

Exemplars are held out from evaluation: when `screen()` is called on an
exemplar paper, we short-circuit with prediction=label and a marker
confidence=-99.0 (filtered downstream in scripts/fewshot_eval.py).

Picked exemplars:
  Includes (1):
    W2026629696 — Internet self-help for social anxiety, RCT, "gains maintained a year later"
    W2080388003 — Online CBT vs sertraline for GAD RCT, 12-month follow-up
    W2067729524 — PTSD exposure/cognitive restructuring RCT, 6-month follow-up (PARTIAL on outcome)
  Excludes (0):
    W2046280100 — CBT/sertraline trial in 7-17yos — wrong population
    W2331047521 — EEG prediction study, not a CBT outcome RCT — wrong intervention/outcome focus
    W68076029   — Hyperbaric oxygen for TBI — wrong intervention and population

Designed for van_Dis_2020 (anxiety-CBT-long-term). For other datasets the
exemplars would need to be replaced.
"""

from __future__ import annotations

import time
from typing import Any, List

from .base import Paper, ScreeningArchitecture, ScreeningContext, ScreeningResult, invoke
from .leads_native import (
    LEADS_SCREENING_PROMPT, _build_pico_criteria, _parse_evaluations, _score, _stringify,
)


# (paper_id, title, abstract, label, [per-criterion verdicts as (P, I, C, O) tuples with rationales])
EXEMPLARS: list[tuple[str, str, str, int, list[tuple[str, str]]]] = [
    (
        "W2026629696",
        "Guided and unguided self-help for social anxiety disorder: randomised controlled trial",
        "Internet-delivered self-help programmes with added therapist guidance have shown efficacy in social anxiety disorder, but unguided self-help has been insufficiently studied. To evaluate the efficacy of guided and unguided self-help for social anxiety disorder, participants followed a cognitive-behavioural self-help programme in the form of either pure bibliotherapy or an internet-based treatment with therapist guidance. Participants (n=235) were randomised to one of three conditions. Pure bibliotherapy and the internet-based treatment were better than waiting list on measures of social anxiety. Gains were well maintained a year later. Unguided self-help through bibliotherapy can produce enduring improvement for individuals with social anxiety disorder.",
        1,
        [
            ("YES", "Adults with social anxiety disorder, diagnostically defined."),
            ("YES", "Cognitive-behavioural self-help (bibliotherapy and internet-based), a CBT modality."),
            ("YES", "Waiting list control — an eligible comparator."),
            ("YES", "Social anxiety symptoms with gains maintained one year (≥12 months) later."),
        ],
    ),
    (
        "W2080388003",
        "Effectiveness of an online e-health application compared to attention placebo or Sertraline in the treatment of Generalised Anxiety Disorder",
        "Generalised Anxiety Disorder (GAD) is a high prevalence disorder. This study compares an Internet Intervention for GAD and an SSRI (sertraline) with an online attention placebo. To evaluate effectiveness of a web-based intervention for GAD over a 10 week period, with follow-up at 6 and 12 months. 152 people aged 18-30 years who met criteria for GAD on the MINI were randomized. Primary outcome was anxiety symptoms (GAD-7).",
        1,
        [
            ("YES", "Adults (18-30) meeting DSM/MINI criteria for GAD."),
            ("YES", "Internet-delivered CBT intervention (web-based for GAD)."),
            ("YES", "Online attention placebo and sertraline as active comparators."),
            ("YES", "Anxiety symptoms measured at 12-month follow-up."),
        ],
    ),
    (
        "W2067729524",
        "Treatment of posttraumatic stress disorder by exposure and/or cognitive restructuring: a controlled study.",
        "87 patients with posttraumatic stress disorder of at least 6 months' duration were randomly assigned to 10 sessions of prolonged exposure alone; cognitive restructuring alone; combined; or relaxation. Exposure and cognitive restructuring improved PTSD markedly. Gains continued to 6-month follow-up.",
        1,
        [
            ("YES", "Adults with PTSD of at least 6 months duration."),
            ("YES", "Prolonged exposure and cognitive restructuring — both core CBT components."),
            ("YES", "Relaxation control — an eligible active comparator."),
            ("PARTIAL", "PTSD symptoms assessed at 6-month follow-up; abstract does not explicitly confirm ≥12-month follow-up but reports sustained gains."),
        ],
    ),
    (
        "W2046280100",
        "Predictors and moderators of treatment response in childhood anxiety disorders: Results from the CAMS trial.",
        "488 youths ages 7-17 years (50% female; 74% ≤ 12 years) meeting DSM-IV criteria for separation anxiety disorder, social phobia, or generalized anxiety disorder were randomly assigned to receive CBT, sertraline, their combination, or pill placebo in the Child/Adolescent Anxiety Multimodal Study (CAMS). We examined predictors and moderators of treatment outcomes.",
        0,
        [
            ("NO", "Population is youths aged 7-17 years (children/adolescents), not adults."),
            ("YES", "CBT was one of the randomized arms."),
            ("YES", "Sertraline and pill placebo control — eligible comparators."),
            ("PARTIAL", "Outcomes were measured but the focus is predictors/moderators at week 12, not long-term anxiety follow-up."),
        ],
    ),
    (
        "W2331047521",
        "Prediction of Treatment Outcome in Patients with OCD with Low-Resolution Brain Electromagnetic Tomography: A Prospective EEG Study",
        "This prospective study investigated brain electric activity (LORETA) for predicting response to treatment in OCD. 41 unmedicated patients with DSM-IV OCD were included. EEG was obtained before and after 10 weeks of standardized treatment with sertraline and behavioral therapy. Results suggest LORETA could prospectively identify treatment responders in OCD.",
        0,
        [
            ("YES", "Adults with DSM-IV OCD diagnosis."),
            ("PARTIAL", "Behavioural therapy is administered alongside sertraline, but study is observational EEG-prediction, not a randomized comparison of CBT vs. another arm."),
            ("NO", "No comparator arm; single-arm prospective study, not an RCT comparison."),
            ("NO", "Primary outcome is EEG-based prediction of response, not long-term anxiety symptom severity at 12+ months."),
        ],
    ),
    (
        "W68076029",
        "Hyperbaric side effects in a traumatic brain injury randomized clinical trial.",
        "To catalog the side effects of 2.4 atmospheres absolute hyperbaric oxygen vs. sham on post-concussion symptoms in military service members with combat-related, mild traumatic brain injury. Fifty subjects diagnosed with TBI were randomized to sham or treatment hyperbaric profile.",
        0,
        [
            ("NO", "Population is military service members with TBI, not adults with an anxiety disorder."),
            ("NO", "Intervention is hyperbaric oxygen therapy, not CBT."),
            ("YES", "Sham hyperbaric profile is a comparator."),
            ("NO", "Outcome is post-concussion side effects, not anxiety symptom severity at long-term follow-up."),
        ],
    ),
]

EXEMPLAR_IDS = {e[0] for e in EXEMPLARS}


def _format_exemplars(criteria_names: list[str]) -> str:
    blocks = []
    for i, (pid, title, abstract, label, verdicts) in enumerate(EXEMPLARS, start=1):
        evals_json = "[\n" + ",\n".join(
            f'        {{"eligibility": "{v[0]}", "rationale": "{v[1]}"}}'
            for v in verdicts
        ) + "\n    ]"
        decision = "INCLUDE" if label == 1 else "EXCLUDE"
        blocks.append(
            f"## Example {i} — Final decision: {decision}\n"
            f"Title: {title}\n"
            f"Abstract: {abstract}\n"
            f"Evaluation:\n```json\n{{\n    \"evaluations\": {evals_json}\n}}\n```"
        )
    header = (
        "# WORKED EXAMPLES #\n"
        "Below are six worked examples (three INCLUDE, three EXCLUDE) showing how to apply the "
        "evaluation criteria. Study them carefully — they illustrate which combinations of YES/PARTIAL/NO/UNCERTAIN "
        "decisions on the criteria justify inclusion vs. exclusion, and when to use PARTIAL or UNCERTAIN.\n\n"
    )
    return header + "\n\n".join(blocks) + "\n\n---\n# NOW EVALUATE THE FOLLOWING PAPER #\n"


class LeadsNativeFewshot(ScreeningArchitecture):
    """LEADS-native with 6-shot prompt (3 includes + 3 excludes from van_Dis_2020).

    Exemplars are short-circuited at screen() time so they are not evaluated
    against themselves. Post-processing must filter their predictions to
    compute valid metrics on the remaining 282-paper holdout.
    """

    name = "leads_native_fewshot"
    threshold: float = 0.0

    def screen(self, paper: Paper, ctx: ScreeningContext, model: Any) -> ScreeningResult:
        t0 = time.time()

        # Skip exemplars: return gold label with sentinel confidence
        if paper.paper_id in EXEMPLAR_IDS:
            label_int = paper.label if paper.label is not None else 0
            return ScreeningResult(
                paper_id=paper.paper_id,
                prediction=int(label_int),
                confidence=-99.0,  # sentinel: this is a held-out exemplar
                reasoning="EXEMPLAR — held out from evaluation",
                per_criterion={},
                llm_calls=0,
                wall_time_s=time.time() - t0,
                raw_outputs=[],
            )

        criteria, num_criteria = _build_pico_criteria(ctx.pico)
        examples_block = _format_exemplars(criteria)
        prompt = examples_block + LEADS_SCREENING_PROMPT.format(
            paper_content=f"Title: {paper.title}\n\nAbstract: {paper.abstract}",
            num_criteria=num_criteria,
            criteria_text=_stringify(criteria),
        )
        raw = invoke(model, prompt)
        evaluations = _parse_evaluations(raw)
        if not evaluations:
            evaluations = [{"eligibility": "UNCERTAIN", "rationale": "Parse failure."} for _ in range(num_criteria)]

        score = _score(evaluations)
        prediction = 1 if score >= self.threshold else 0
        per_crit: dict[str, str] = {}
        for crit, ev in zip(criteria, evaluations[:num_criteria]):
            per_crit[crit] = str(ev.get("eligibility", "UNCERTAIN")).upper()
        reasoning = f"LEADS-fewshot score={score:+.2f}. " + "; ".join(
            f"{crit.split(':')[0]}={ev.get('eligibility','?')}"
            for crit, ev in zip(criteria, evaluations[:num_criteria])
        )

        return ScreeningResult(
            paper_id=paper.paper_id,
            prediction=prediction,
            confidence=(score + 1) / 2,
            reasoning=reasoning,
            per_criterion=per_crit,
            llm_calls=1,
            wall_time_s=time.time() - t0,
            raw_outputs=[raw],
        )
