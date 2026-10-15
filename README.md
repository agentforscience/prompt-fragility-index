# The Prompt Fragility Index (PFI)

This project measures how six kinds of small, meaning-preserving edits to an instruction prompt change LLM accuracy: whitespace, capitalization, punctuation, typos, synonym substitution, and sentence reordering. The edits are applied singly and in combination, across classification, multiple-choice and generation tasks. We ran 20,800 real generations from **Qwen2.5-1.5B-Instruct** and **Llama-3.2-1B-Instruct**, served locally with llama.cpp on CPU, because no API key was available.

Full write-up: **[REPORT.md](REPORT.md)**.

## Key findings

- **The hypothesis was refuted as stated.** Reorder and synonym edits were *not* the most damaging (Qwen: ≤4 points everywhere). Capitalization was *not* harmless: it cost Qwen −25 points on AG News, and punctuation cost −16. Only whitespace was consistently negligible.
- **What matters is what the edit touches, not its category.** Edits that alter the label list (removing the commas in "World, Sports, Business, and Sci/Tech", or re-casing it) shift the model's label prior. Edits that weaken the "answer only with…" constraint trigger preambles and verbose answers.
- **Generation was not the most fragile task type.** Mean single-edit PFI on SQuAD was 0.03 (Qwen) and 0.00 (Llama), versus the predicted 0.22. The most fragile cells were AG News for Qwen (0.10) and ARC for Llama (0.10).
- **Compounding is task-dependent.** Combined edits were super-additive only for generation (Qwen SQuAD: combo/sum ratio 1.29–1.61; all six edits together lost +10 points beyond the union null, CI [4, 16]). Classification and multiple-choice were sub-additive (ratio 0.4–0.8), because items saturate.
- **Much small-model fragility is format fragility.** With a 48-token budget and a lenient parser, Llama's ARC drops of 28–50 points under edited prompts shrank to 0–10 points. Net Δ also hides churn: flip rates are often several times the net drop. Re-running the identical prompt flips 0–2% of items.

## Reproduce

Requires about 2–3 CPU-hours per model on a shared 32-core machine; no GPU.

```bash
uv venv --python 3.11 && source .venv/bin/activate && uv sync    # or: uv pip install -r requirements.txt
# datasets: see datasets/README.md (HF downloads)
# llama.cpp b11624 release binary -> tools/llamacpp/llama-b11624/ ; GGUF models from bartowski/*-GGUF (Q8_0)
scripts/serve.sh <model.gguf> 18393 8 6 &                          # one OpenAI-compatible server per task (ports 18393-18396)
scripts/run_phase.sh qwen2.5-1.5b core      # original + 6 edits x 2 seeds
scripts/run_phase.sh qwen2.5-1.5b combos    # 6 combos x 2 seeds
scripts/run_phase.sh qwen2.5-1.5b rerun     # noise-floor rerun of the original
python src/analyze.py --model qwen2.5-1.5b && python src/error_analysis.py --model qwen2.5-1.5b
python src/summarize.py                     # cross-model tables/figures (after both models)
python src/robustness_maxtokens.py --model-tag llama3.2-1b --port 18397 --task arc \
       --conditions original,synonym@0,semantic3@0,mixed3@0 --max-tokens 48
```

Runs are resumable: each (condition, item) response is appended to `results/raw/<model>/<task>.jsonl` and skipped on re-run. To use an API model instead, point `OpenAI(base_url=...)` in `src/run_experiment.py` at OpenRouter.

## Layout

```
src/perturb.py              edit operators (seeded per edit/sentence so combos = union of singles)
src/tasks.py                templates, item sampling, scoring (SST-2, AG News, ARC-C, SQuAD)
src/run_experiment.py       runner (conditions, caching, llama.cpp/OpenAI-compatible client)
src/analyze.py              per-edit stats, bootstrap CIs, McNemar/Wilcoxon + Holm, compounding, figures
src/error_analysis.py       parse failures, label shifts, SQuAD verbosity, edit magnitude, flip examples
src/summarize.py            cross-model tables + figures
src/robustness_maxtokens.py larger-budget / lenient-parser re-run
scripts/                    server, phase and chain launchers; throughput benchmarks
results/raw/                all prompts and responses (JSONL)
results/*.csv, *.md         summary tables, error analyses
figures/                    plots used in REPORT.md
planning.md                 pre-registered plan (with motivation and novelty)
```
