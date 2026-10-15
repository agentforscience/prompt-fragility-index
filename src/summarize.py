"""Cross-model summary tables and figures for the report.

Run after analyze.py for each model. Writes:
  results/h1_edit_types.csv      per model x edit type, pooled over valid tasks
  results/format_vs_content.csv  parse-failure rate and accuracy-among-parsed per model x task x edit
  results/duplicate_consistency.json
  figures/edit_type_by_model.png, figures/pfi_by_task_model.png, figures/flip_vs_delta.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from perturb import EDIT_TYPES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R, F = ROOT / "results", ROOT / "figures"
MODELS = ["qwen2.5-1.5b", "llama3.2-1b"]
# Llama-3.2-1B on AG News is degenerate (predicts "World" for 98/100 items, accuracy = 25% = chance),
# so relative PFI is undefined/meaningless there; it is excluded from pooled summaries (reported separately).
EXCLUDE = {("llama3.2-1b", "agnews")}
RNG = np.random.default_rng(0)


def raw(model):
    rows = []
    for f in sorted((R / "raw" / model).glob("*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    return pd.DataFrame(rows)


def main():
    et = pd.concat([pd.read_csv(R / f"edit_type_{m}.csv").assign(model=m) for m in MODELS])
    et["valid"] = [(m, t) not in EXCLUDE for m, t in zip(et.model, et.task)]

    # H1: pooled over valid tasks (mean of per-task deltas), with min/max across tasks
    v = et[et.valid]
    h1 = v.groupby(["model", "edit"]).agg(mean_delta_pts=("delta_pts", "mean"),
                                          max_task_delta_pts=("delta_pts", "max"),
                                          mean_abs_delta_pts=("delta_pts", lambda x: np.abs(x).mean()),
                                          mean_flip=("flip_rate", "mean"),
                                          n_tasks_sig_holm=("p_holm", lambda x: int((x < 0.05).sum())),
                                          n_tasks=("task", "count")).reset_index()
    h1["edit"] = pd.Categorical(h1["edit"], EDIT_TYPES)
    h1 = h1.sort_values(["model", "edit"])
    h1.to_csv(R / "h1_edit_types.csv", index=False)
    print(h1.round(2).to_string(index=False))

    # Format vs. content decomposition (classification / MC tasks)
    fc = []
    for m in MODELS:
        df = raw(m).drop_duplicates(["task", "condition", "item_id"])
        df = df[df.task != "squad"]
        df["edit"] = df.condition.str.split("@").str[0]
        df["fail"] = df.parsed.isna()
        for (t, e), g in df.groupby(["task", "edit"]):
            parsed = g[~g.fail]
            fc.append({"model": m, "task": t, "edit": e, "n": len(g), "parse_fail_pct": 100 * g.fail.mean(),
                       "acc_pct": 100 * g.score.mean(), "acc_among_parsed_pct": 100 * parsed.score.mean()})
    fc = pd.DataFrame(fc)
    fc.to_csv(R / "format_vs_content.csv", index=False)

    # Duplicate consistency (accidental double-runs of identical prompts on SQuAD for Llama)
    dup = {}
    for m in MODELS:
        df = raw(m)
        g = df.groupby(["task", "condition", "item_id"])
        multi = g.filter(lambda x: len(x) > 1)
        if len(multi):
            same = multi.groupby(["task", "condition", "item_id"])["output"].nunique().eq(1)
            dup[m] = {"n_pairs": int(len(same)), "identical_output_frac": float(same.mean())}
    json.dump(dup, open(R / "duplicate_consistency.json", "w"), indent=2)
    print("duplicates:", dup)

    # Figure: per-edit-type drop for each model and task
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    tasks = ["sst2", "agnews", "arc", "squad"]
    colors = {"sst2": "#4C72B0", "agnews": "#DD8452", "arc": "#55A868", "squad": "#C44E52"}
    for ax, m in zip(axes, MODELS):
        sub = et[et.model == m]
        w = 0.2
        for i, t in enumerate(tasks):
            s = sub[sub.task == t].set_index("edit").reindex(EDIT_TYPES)
            x = np.arange(len(EDIT_TYPES)) + (i - 1.5) * w
            y = s.delta_pts.values
            err = np.vstack([y - s.ci_lo_pts.values, s.ci_hi_pts.values - y])
            hatch = "//" if (m, t) in EXCLUDE else None
            ax.bar(x, y, w, yerr=err, capsize=2, color=colors[t], hatch=hatch, alpha=0.5 if hatch else 1,
                   label=t + (" (degenerate baseline)" if hatch else ""))
        ax.axhline(0, color="k", lw=0.8)
        ax.axhline(8, color="gray", ls="--", lw=0.8)
        ax.axhline(1, color="gray", ls=":", lw=0.8)
        ax.set_xticks(range(len(EDIT_TYPES)), EDIT_TYPES, rotation=30)
        ax.set_title(m)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("score drop vs. original (points)\nmean of 2 seeds, 95% bootstrap CI")
    fig.suptitle("Effect of single edit types (dashed = 8-pt, dotted = 1-pt thresholds from H1)", fontsize=10)
    plt.tight_layout()
    plt.savefig(F / "edit_type_by_model.png", dpi=150)
    plt.close()

    # Figure: task-level mean PFI with CI (single edits), both models
    tp = pd.concat([pd.read_csv(R / f"task_pfi_{m}.csv").assign(model=m) for m in MODELS])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for j, m in enumerate(MODELS):
        s = tp[tp.model == m].set_index("task").reindex(tasks)
        x = np.arange(len(tasks)) + (j - 0.5) * 0.35
        y = s.mean_pfi_single.values
        ax.bar(x, y, 0.35, yerr=np.vstack([y - s.ci_lo.values, s.ci_hi.values - y]), capsize=3, label=m)
    ax.axhline(0.05, color="#4C72B0", ls=":", lw=1, label="H2: classification 0.05")
    ax.axhline(0.22, color="#C44E52", ls="--", lw=1, label="H2: generation 0.22")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(tasks)), ["SST-2\n(classif.)", "AG News\n(classif.)", "ARC-C\n(mult. choice)",
                                      "SQuAD\n(generation)"])
    ax.set_ylabel("mean PFI over 12 single-edit variants")
    ax.legend(fontsize=7)
    ax.set_title("Prompt Fragility Index by task (Llama AG News baseline is at chance)", fontsize=10)
    plt.tight_layout()
    plt.savefig(F / "pfi_by_task_model.png", dpi=150)
    plt.close()

    # Figure: net delta vs flip rate per condition (churn hidden by net averages)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for m, mk in zip(MODELS, ["o", "^"]):
        s = pd.read_csv(R / f"summary_{m}.csv")
        s = s[~s.condition.isin(["original"])]
        s = s[[(m, t) not in EXCLUDE for t in s.task]]
        ax.scatter(s.delta_pts, 100 * s.flip_rate, marker=mk, alpha=0.6, label=m, s=18)
    ax.plot([0, 50], [0, 50], "k--", lw=0.8, label="|net drop| = flip rate")
    ax.plot([0, -50], [0, 50], "k--", lw=0.8)
    ax.set_xlabel("net score drop (points)")
    ax.set_ylabel("item flip rate (%)")
    ax.set_title("Net drop understates item-level churn", fontsize=10)
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(F / "flip_vs_delta.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
