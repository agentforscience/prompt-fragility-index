# Datasets

Downloaded from the HuggingFace Hub with `datasets` (data files are git-ignored):

| File | HF id / split | Used items |
|---|---|---|
| `sst2_validation.jsonl` | `stanfordnlp/sst2`, validation (872) | 100, random seed 42 |
| `ag_news_test.jsonl` | `fancyzhx/ag_news`, test (7,600) | 100, random seed 42 |
| `arc_challenge_test.jsonl` | `allenai/ai2_arc` ARC-Challenge, test (1,172) | 100 four-option items, seed 42 |
| `squad_validation.jsonl` | `rajpurkar/squad`, validation (10,570) | 100; passages of ≤110 words, one question per passage, seed 42 |
| `gsm8k_test.jsonl` | `openai/gsm8k` main, test | downloaded but not used (too expensive for CPU-only decoding) |

Re-download:
```python
from datasets import load_dataset
load_dataset("stanfordnlp/sst2", split="validation").to_json("datasets/sst2_validation.jsonl")
load_dataset("fancyzhx/ag_news", split="test").to_json("datasets/ag_news_test.jsonl")
load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").to_json("datasets/arc_challenge_test.jsonl")
load_dataset("rajpurkar/squad", split="validation").to_json("datasets/squad_validation.jsonl")
```
Item sampling is deterministic, see `src/tasks.py::load_items`.
