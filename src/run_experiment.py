"""Run the prompt-fragility experiment against a local llama.cpp server (OpenAI-compatible API).

Every (condition, item) response is appended to results/raw/<model_tag>/<task>.jsonl, so the
run is resumable; already-completed (condition, item) pairs are skipped.

Usage:
  python src/run_experiment.py --model-tag qwen2.5-1.5b --port 18393 --n 100 --phase core
"""
import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perturb import EDIT_TYPES, build_instruction  # noqa: E402
from tasks import TASKS, load_items, score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

COMBOS = {
    "cosmetic2": ("whitespace", "capitalization"),
    "semantic2": ("synonym", "reorder"),
    "cosmetic3": ("whitespace", "capitalization", "punctuation"),
    "semantic3": ("synonym", "reorder", "typo"),
    "mixed3": ("capitalization", "synonym", "reorder"),
    "all6": tuple(EDIT_TYPES),
}


def conditions(phase, seeds):
    """Return list of (cond_name, edits, seed) for the requested phase."""
    if phase == "core":
        conds = [("original", (), 0)]
        conds += [(f"{e}@{s}", (e,), s) for s in seeds for e in EDIT_TYPES]
        return conds
    if phase == "combos":
        return [(f"{c}@{s}", COMBOS[c], s) for s in seeds for c in COMBOS]
    if phase == "rerun":
        return [("original_rerun", (), 0)]
    raise ValueError(phase)


def build_prompt(task, item, edits, seed):
    t = TASKS[task]
    instr, sep = build_instruction(t["sentences"], edits, seed, t["protected"])
    return instr + sep + item["input"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--port", type=int, default=18393)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tasks", default="sst2,agnews,arc,squad")
    ap.add_argument("--phase", default="core")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="local", timeout=900)
    seeds = [int(s) for s in args.seeds.split(",")]
    outdir = ROOT / "results" / "raw" / args.model_tag
    outdir.mkdir(parents=True, exist_ok=True)

    def call(prompt, max_tokens):
        for attempt in range(5):
            try:
                r = client.chat.completions.create(
                    model=args.model_tag, messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=max_tokens, seed=42)
                return r.choices[0].message.content or "", r.usage.prompt_tokens, r.usage.completion_tokens
            except Exception as ex:  # transient server errors: back off and retry
                print(f"  retry {attempt}: {ex}", flush=True)
                time.sleep(5 * (attempt + 1))
        raise RuntimeError("server failed repeatedly")

    tasks = args.tasks.split(",")
    for cond, edits, seed in conditions(args.phase, seeds):
        for task in tasks:
            path = outdir / f"{task}.jsonl"
            done = set()
            if path.exists():
                done = {(d["condition"], d["item_id"]) for d in map(json.loads, open(path))}
            items = [it for it in load_items(task, args.n) if (cond, it["id"]) not in done]
            if not items:
                continue
            t0 = time.time()
            prompts = [build_prompt(task, it, edits, seed) for it in items]
            with cf.ThreadPoolExecutor(args.workers) as ex, open(path, "a") as f:
                futs = {ex.submit(call, p, TASKS[task]["max_tokens"]): (it, p) for it, p in zip(items, prompts)}
                for fu in cf.as_completed(futs):
                    it, p = futs[fu]
                    out, pt, ct = fu.result()
                    sc = score(task, out, it["gold"])
                    rec = {"model": args.model_tag, "task": task, "task_type": TASKS[task]["type"],
                           "condition": cond, "edits": list(edits), "seed": seed, "item_id": it["id"],
                           "prompt": p, "output": out, "gold": it["gold"], "parsed": sc["parsed"],
                           "score": sc["score"], "em": sc["em"], "prompt_tokens": pt, "completion_tokens": ct}
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
            print(f"[{time.strftime('%H:%M:%S')}] {task:7s} {cond:18s} n={len(items)} "
                  f"{time.time() - t0:6.1f}s", flush=True)


if __name__ == "__main__":
    main()
