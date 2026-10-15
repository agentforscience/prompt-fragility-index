"""Error analysis: parse failures, label-distribution shifts, edit magnitude vs. effect, flip examples.

Usage: python src/error_analysis.py --model qwen2.5-1.5b
Writes results/error_analysis_<model>.md and results/edit_magnitude_<model>.csv
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load, TASK_ORDER  # noqa: E402
from perturb import build_instruction, char_edit_distance_ratio  # noqa: E402
from run_experiment import COMBOS  # noqa: E402
from tasks import TASKS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def edits_of(cond):
    name, seed = cond.split("@") if "@" in cond else (cond, "0")
    if name in COMBOS:
        return COMBOS[name], int(seed)
    if name.startswith("original"):
        return (), 0
    return (name,), int(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    df = load(args.model)
    lines = [f"# Error analysis: {args.model}\n"]

    # 1. Parse failures (no recognizable label/letter) per task x condition
    lines.append("## Parse-failure rate (% of outputs with no recognizable label/letter)\n")
    cls = df[df.task != "squad"].copy()
    cls["fail"] = cls["parsed"].isna()
    pf = cls.pivot_table(index="condition", columns="task", values="fail", aggfunc="mean") * 100
    lines.append(pf.round(1).to_markdown() + "\n")

    # 2. Prediction distribution shifts (classification)
    lines.append("## Predicted-label distribution per condition\n")
    for task in ["sst2", "agnews", "arc"]:
        sub = df[df.task == task]
        if sub.empty:
            continue
        tab = sub.assign(parsed=sub["parsed"].fillna("NONE")).pivot_table(
            index="condition", columns="parsed", values="item_id", aggfunc="count", fill_value=0)
        lines.append(f"### {task}\n\n" + tab.to_markdown() + "\n")

    # 3. SQuAD output length (verbosity) per condition
    sq = df[df.task == "squad"]
    if len(sq):
        lines.append("## SQuAD: mean output length (words) and EM per condition\n")
        t = sq.assign(len_words=sq["output"].str.split().str.len()).groupby("condition").agg(
            f1=("score", "mean"), em=("em", "mean"), len_words=("len_words", "mean"))
        lines.append(t.round(3).to_markdown() + "\n")

    # 4. Edit magnitude vs. effect
    mag = []
    for task in TASK_ORDER:
        sub = df[df.task == task]
        if sub.empty or "original" not in set(sub.condition):
            continue
        orig_instr, _ = build_instruction(TASKS[task]["sentences"], (), 0, TASKS[task]["protected"])
        base = sub[sub.condition == "original"].set_index("item_id")["score"]
        for cond in sorted(set(sub.condition) - {"original", "original_rerun"}):
            ed, seed = edits_of(cond)
            instr, _ = build_instruction(TASKS[task]["sentences"], ed, seed, TASKS[task]["protected"])
            s = sub[sub.condition == cond].set_index("item_id")["score"]
            common = base.index.intersection(s.index)
            mag.append({"task": task, "condition": cond, "k_edits": len(ed),
                        "char_dist": char_edit_distance_ratio(orig_instr, instr),
                        "delta_pts": 100 * (base[common] - s[common]).mean(),
                        "flip": float(((base[common] >= 0.5) != (s[common] >= 0.5)).mean())})
    mag = pd.DataFrame(mag)
    mag.to_csv(ROOT / "results" / f"edit_magnitude_{args.model}.csv", index=False)
    if len(mag) > 3:
        r1 = stats.spearmanr(mag.char_dist, mag.delta_pts)
        r2 = stats.spearmanr(mag.char_dist, mag.flip)
        lines.append("## Edit magnitude vs. effect (all perturbed conditions, all tasks)\n")
        lines.append(f"Spearman(char edit distance, drop) = {r1.statistic:.3f} (p={r1.pvalue:.3g}); "
                     f"Spearman(char edit distance, flip rate) = {r2.statistic:.3f} (p={r2.pvalue:.3g}); n={len(mag)}\n")

    # 5. Example flips (correct under original -> wrong under edit) for the most damaging condition per task
    lines.append("## Example flips (original correct -> edited wrong)\n")
    for task in TASK_ORDER:
        m = mag[mag.task == task] if len(mag) else mag
        if m.empty:
            continue
        worst = m.sort_values("delta_pts").iloc[-1]["condition"]
        sub = df[df.task == task]
        o = sub[sub.condition == "original"].set_index("item_id")
        e = sub[sub.condition == worst].set_index("item_id")
        common = o.index.intersection(e.index)
        flips = [i for i in common if o.loc[i, "score"] >= 0.5 > e.loc[i, "score"]][:3]
        lines.append(f"### {task}: worst condition = `{worst}`\n")
        lines.append("Edited prompt instruction (first item):\n\n```\n" +
                     e.loc[common[0], "prompt"].split("Review:")[0].split("Article:")[0].split("Question:")[0]
                     .split("Passage:")[0][:600] + "\n```\n")
        for i in flips:
            gold = o.loc[i, "gold"] if task != "squad" else o.loc[i, "gold"][0]
            lines.append(f"- `{i}` gold=`{gold}` | original→`{o.loc[i, 'output'][:60]!s}` | edited→`{e.loc[i, 'output'][:60]!s}`")
        lines.append("")

    out = ROOT / "results" / f"error_analysis_{args.model}.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines)[:6000])


if __name__ == "__main__":
    main()
