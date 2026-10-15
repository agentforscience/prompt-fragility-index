# Planning: The Prompt Fragility Index (PFI)

## Motivation & Novelty Assessment

### Why This Research Matters
Practitioners routinely edit prompts (tidy whitespace, re-case, rephrase, reorder instructions) assuming
such edits are harmless. If some edit types silently change model accuracy by several points, then
(a) benchmark numbers reported with a single prompt are not reproducible, and (b) prompt maintenance in
production is risky. A per-edit-type, per-task-type fragility measure tells practitioners which edits
need re-validation and which are safe.

### Gap in Existing Work
(No literature review was pre-gathered for this run, the resource-finder stage was skipped and the
paper-search service returned no results, so this summary is from the author's knowledge of the field.)
Prior work shows LLMs are sensitive to prompt formatting (Sclar et al., 2024, "Quantifying Language
Models' Sensitivity to Spurious Features in Prompt Design" / FormatSpread), to paraphrased instructions
(Mizrahi et al., 2024, "State of What Art? A Call for Multi-Prompt LLM Evaluation"), to adversarial
character/word/sentence perturbations (Zhu et al., 2023, PromptBench), to few-shot ordering
(Lu et al., 2022), and proposes sensitivity scores (ProSA's PromptSensiScore, Zhuo et al., 2024; POSIX,
Chatterjee et al., 2024). Missing: a controlled, *benign*-edit taxonomy (the edits a human editor makes,
not adversarial attacks), compared on equal footing across task *types*, with an explicit test of
whether edit types **compound** super-additively when combined.

### Our Novel Contribution
1. A Prompt Fragility Index, PFI = relative performance drop of an edited prompt versus the original,
   reported per edit type × task type, along with an item-level flip rate.
2. A seeded edit generator in which a combined edit contains the same local edits as its single-edit
   counterparts, so the interaction (compounding) term can be measured directly:
   compounding ratio = drop(combo) / Σ drop(singles).

### Experiment Justification
- Exp 1 (single edit types × 4 tasks): measures the main effect of each edit type. Needed for H1 (reorder/synonym > 8% vs. whitespace/caps < 1%).
- Exp 2 (task-type comparison): same edits on classification / multiple-choice / generation tasks. Needed for H2 (classification PFI ≈ 0.05 vs. generation PFI ≈ 0.22).
- Exp 3 (combinations of 2, 3 and 6 edit types): needed for H3 (3+ edits compound 30–50% beyond additive).
- Exp 4 (re-run of the original prompt): noise floor from server nondeterminism, so we don't attribute noise to edits.
- Exp 5 (if compute allows): a second model, to check generality.

## Research Question
How much do small, meaning-preserving edits to an instruction prompt (whitespace, capitalization,
punctuation, typos, synonym substitution, sentence reordering) change LLM task performance; does this
depend on task type; and do edits compound super-additively?

## Hypothesis Decomposition
- H1a: reorder and synonym edits each cause > 8% accuracy loss (we test both absolute points and relative PFI).
- H1b: whitespace and capitalization edits each cause < 1% loss.
- H2: mean PFI (over edit types) is lowest for classification (≈0.05) and highest for generation (≈0.22).
- H3: for combos of ≥3 edit types, drop(combo) exceeds Σ drop(individual) by 30–50% (ratio 1.3–1.5).

## Proposed Methodology

### Compute reality and resulting design constraints
- **No LLM API key is available** in this session (OPENROUTER_KEY unset; the pipeline .env entry is
  commented out). We therefore use **real open-weight instruction-tuned LLMs run locally** with llama.cpp
  (Q8_0 GGUF, CPU). This is not a simulation: every number comes from actual model outputs.
- No GPU; the 32-core host is shared and heavily oversubscribed (load average ≈ 60). The measured prefill
  rate was ≈ 40 tokens/s. This forces a small model (Qwen2.5-1.5B-Instruct as the primary model, with
  Qwen2.5-3B-Instruct or Llama-3.2-3B-Instruct as a secondary model on a subset if time allows), short
  outputs, and n = 100 items per task.

### Tasks (4 tasks, 3 task types)
| Task | Type | Metric | Items |
|---|---|---|---|
| SST-2 (validation) | classification (2-way) | accuracy | 100 |
| AG News (test) | classification (4-way) | accuracy | 100 |
| ARC-Challenge (test) | multiple-choice QA | accuracy | 100 |
| SQuAD v1.1 (validation, passages ≤110 words) | free-form short-answer generation | token F1 (EM secondary) | 100 |

Each prompt = 4-sentence instruction + separator + task input. Edits touch only the instruction.

### Edit types (each with 2 random seeds)
whitespace (extra spaces/tabs/newlines), capitalization (all-lowercase, or random word re-casing),
punctuation (deletion/swap), typo (character swap/drop/duplication/keyboard-neighbor substitution in ~15% of words),
synonym (curated near-synonym lexicon, p = 0.6), reorder (non-identity permutation of instruction sentences).
Task-critical label words are protected from synonym and typo edits.

### Combinations
cosmetic2 (ws+caps), semantic2 (syn+reorder), cosmetic3 (ws+caps+punct), semantic3 (syn+reorder+typo),
mixed3 (caps+syn+reorder), all6. Each at seeds 0 and 1.

### Decoding
Greedy (temperature 0), seed 42, max_tokens 8 (classification/MC) or 32 (generation). The default chat template is used.

### Baselines
The unedited original prompt (plus a rerun of it as a noise-floor control).

### Evaluation Metrics
- Score drop Δ = score(orig) − score(edit), in absolute points.
- PFI = Δ / score(orig) (relative drop; positive = worse). We also report |PFI| because edits can help.
- Flip rate = fraction of items whose correctness changes (sensitive to churn even when the mean is unchanged).
- Compounding ratio = Δ(combo) / Σ Δ(singles at the same seed).

### Statistical Analysis Plan
- Per edit type and task: paired item-level differences (pooled over the 2 seeds); bootstrap 95% CI (10k resamples, resampling items); exact McNemar test for binary metrics, Wilcoxon signed-rank for F1. Holm correction across the 6 edit types × 4 tasks.
- H2: compare the PFI distribution across task types (bootstrap CI on mean PFI per task type).
- H3: bootstrap CI over items for Δ(combo) − ΣΔ(singles) (the interaction term).
- α = 0.05.

## Expected Outcomes
H1 is supported if the reorder/synonym CIs lie above 8 points and the ws/caps CIs lie below 1 point. H2 is supported
if generation mean PFI > MC > classification with non-overlapping CIs. H3 is supported if the interaction
term for ≥3-edit combos is positive and the ratio is within roughly 1.3–1.5.

## Timeline
Setup and benchmarking (done, ~1 h, mostly diagnosing a saturated CPU) → core run (~1.5–2 h) → combos
(~1.5 h) → second model if possible → analysis and report (~1 h).

## Potential Challenges
- CPU contention may make runs very slow. Mitigation: resumable caching, prefix caching (requests grouped by condition), running phases by priority (core first).
- Small model with low baseline on ARC: PFI is relative, so a low baseline inflates it; we report absolute Δ as well.
- Parsing failures under perturbed prompts (e.g., verbose answers): we use lenient parsing and report the parse-failure rate as a separate error category.

## Success Criteria
All four tasks × 6 edit types × 2 seeds, plus the combos, completed on at least one real model, with CIs, tests and an honest verdict on H1–H3.

## Deviations from Plan (recorded after execution)
- **Second model:** Llama-3.2-1B-Instruct was run on the *full* design (not a subset), because host load dropped mid-run.
- **ARC output budget:** changed from 3 to 8 tokens after a pilot showed truncation-induced parse failures ("The correct answer…") on the original prompt. That run was discarded before any analysis.
- **Post-hoc exclusion:** Llama AG News (baseline 25%, the model predicted "World" for 98% of items) is excluded from pooled conclusions but reported in full.
- **Added analyses not in the plan:**
  - a "union" no-interaction null for compounding, because additive sums double-count saturated binary items;
  - a format-versus-content decomposition;
  - a max-tokens/lenient-parser robustness re-run (Llama ARC);
  - an edit-magnitude correlation.
