"""Robustness check: is format-related fragility an artifact of the short generation budget?

Re-runs selected (task, condition) pairs with a larger max_tokens budget and a lenient answer
extractor, and compares strict (main-experiment) vs. lenient accuracy.

Usage: python src/robustness_maxtokens.py --model-tag llama3.2-1b --port 18397 --task arc \
          --conditions original,synonym@0,semantic3@0,mixed3@0 --max-tokens 48
"""
import argparse
import concurrent.futures as cf
import json
import re
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment import COMBOS, build_prompt  # noqa: E402
from tasks import load_items, score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def lenient_letter(out):
    """Prefer explicit 'answer is X' / 'option X' statements, then a leading letter, then any '(X)' / 'X.'."""
    for pat in [r"answer (?:is|would be)\s*:?\s*\(?\**([A-D])\b", r"\boption\s*\(?([A-D])\b",
                r"^\s*\**\(?([A-Da-d])[\)\.:\s]", r"\(([A-D])\)", r"\b([A-D])\.\s"]:
        m = re.search(pat, out, flags=re.I if pat.startswith("answer") or "option" in pat else 0)
        if m:
            return m.group(1).upper()
    return None


def edits_of(cond):
    name, seed = (cond.split("@") + ["0"])[:2]
    if name == "original":
        return (), 0
    return (COMBOS[name] if name in COMBOS else (name,)), int(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--task", default="arc")
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="local", timeout=900)
    items = load_items(args.task, args.n)
    out_path = ROOT / "results" / f"robustness_maxtokens_{args.model_tag}_{args.task}.jsonl"
    done = set()
    if out_path.exists():
        done = {(d["condition"], d["item_id"]) for d in map(json.loads, open(out_path))}

    def call(p):
        r = client.chat.completions.create(model=args.model_tag, messages=[{"role": "user", "content": p}],
                                           temperature=0.0, max_tokens=args.max_tokens, seed=42)
        return r.choices[0].message.content or ""

    for cond in args.conditions.split(","):
        ed, seed = edits_of(cond)
        todo = [it for it in items if (cond, it["id"]) not in done]
        with cf.ThreadPoolExecutor(6) as ex, open(out_path, "a") as f:
            futs = {ex.submit(call, build_prompt(args.task, it, ed, seed)): it for it in todo}
            for fu in cf.as_completed(futs):
                it = futs[fu]
                o = fu.result()
                lp = lenient_letter(o)
                f.write(json.dumps({"condition": cond, "item_id": it["id"], "output": o, "gold": it["gold"],
                                    "strict": score(args.task, o, it["gold"])["score"],
                                    "lenient_parsed": lp, "lenient": float(lp == it["gold"])}) + "\n")
        print(cond, "done", flush=True)


if __name__ == "__main__":
    main()
