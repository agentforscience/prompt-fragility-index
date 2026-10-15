# Resources used

The automated resource-finder stage was **skipped** for this run (`.neurico/pipeline_state.json`: `"skipped": true`), so no papers, code or literature review were pre-gathered. The paper-finder service also returned no results. Resources were assembled during the experiment:

- **Datasets** (HuggingFace Hub): SST-2, AG News, ARC-Challenge and SQuAD v1.1. GSM8K was downloaded but not used. See `datasets/README.md`.
- **Models** (GGUF Q8_0, `bartowski/*` on HF): Qwen2.5-1.5B-Instruct and Llama-3.2-1B-Instruct, both used. Qwen2.5-3B-Instruct and Llama-3.2-3B-Instruct were downloaded but not used, because they were too slow on the contended CPU.
- **Inference:** llama.cpp release b11624 (`llama-server`, OpenAI-compatible API), installed in `tools/`, which is git-ignored.
- **No LLM API** was used, because `OPENROUTER_KEY` was not set in the session.
- **Literature:** cited from the author's knowledge; see REPORT.md §2 and References.

Process notes:
- `transformers` CPU decoding and multi-thread llama.cpp were benchmarked first. Under host load, throughput was about 40 prefill tok/s per server, so the design was sized accordingly: n = 100, short outputs, 1–1.5B models, and several independent servers.
- An ARC run with `max_tokens = 3` was discarded because of truncation-induced parse failures. It is kept in `results/raw/arc_maxtok3_discarded.jsonl`.
