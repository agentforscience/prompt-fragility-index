"""Analysis of prompt-fragility results.

Reads results/raw/<model>/*.jsonl and writes:
  results/summary_<model>.csv          per (task, condition) score, delta, PFI, flip rate
  results/edit_type_<model>.csv        per (task, edit type) pooled over seeds, bootstrap CI, tests
  results/task_type_<model>.csv        mean PFI per task type
  results/compounding_<model>.csv      combo delta vs sum of single-edit deltas
  figures/*.png
Usage: python src/analyze.py --model qwen2.5-1.5b
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perturb import EDIT_TYPES  # noqa: E402
from run_experiment import COMBOS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(42)
B = 10000
TASK_ORDER = ["sst2", "agnews", "arc", "squad"]
TASK_TYPE = {"sst2": "classification", "agnews": "classification", "arc": "multiple_choice", "squad": "generation"}


def load(model):
    rows = []
    for f in sorted((ROOT / "results" / "raw" / model).glob("*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(["task", "condition", "item_id"], keep="first")
    return df


def wide(df, task):
    """item x condition matrix of scores for one task."""
    return df[df.task == task].pivot(index="item_id", columns="condition", values="score")


def boot_ci(x, stat=np.mean):
    x = np.asarray(x)
    idx = RNG.integers(0, len(x), size=(B, len(x)))
    bs = stat(x[idx], axis=1) if stat is np.mean else np.array([stat(x[i]) for i in idx])
    return np.percentile(bs, [2.5, 97.5])


def mcnemar_exact(a, b):
    """Exact McNemar test on paired binary vectors (a = original, b = edited)."""
    a, b = np.asarray(a).round().astype(int), np.asarray(b).round().astype(int)
    n01, n10 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    if n01 + n10 == 0:
        return 1.0
    return stats.binomtest(n01, n01 + n10, 0.5).pvalue


def paired_test(task, orig, edit):
    if task == "squad":
        d = np.asarray(orig) - np.asarray(edit)
        if np.allclose(d, 0):
            return 1.0
        return stats.wilcoxon(orig, edit, zero_method="zsplit").pvalue
    return mcnemar_exact(orig, edit)


def flip_rate(task, orig, edit):
    orig, edit = np.asarray(orig), np.asarray(edit)
    if task == "squad":  # a "flip" = F1 changes by >= 0.5 (correct <-> incorrect at the usual 0.5 threshold)
        return float(np.mean((orig >= 0.5) != (edit >= 0.5)))
    return float(np.mean(orig != edit))


def per_condition(df):
    out = []
    for task in TASK_ORDER:
        W = wide(df, task)
        if "original" not in W:
            continue
        base = W["original"].mean()
        for cond in W.columns:
            col = W[cond].dropna()
            o = W.loc[col.index, "original"]
            d = (o - col).values
            out.append({"task": task, "task_type": TASK_TYPE[task], "condition": cond, "n": len(col),
                        "score": col.mean(), "orig_score": o.mean(), "delta_pts": 100 * d.mean(),
                        "pfi": d.mean() / base if base > 0 else np.nan,
                        "flip_rate": flip_rate(task, o.values, col.values),
                        "p_value": paired_test(task, o.values, col.values) if cond != "original" else np.nan})
    return pd.DataFrame(out)


def per_edit_type(df):
    """Pool the seeds of each single edit type: item-level delta averaged over seeds, then bootstrap over items."""
    out = []
    for task in TASK_ORDER:
        W = wide(df, task)
        if "original" not in W:
            continue
        base = W["original"].mean()
        for e in EDIT_TYPES:
            cols = [c for c in W.columns if c.startswith(e + "@")]
            if not cols:
                continue
            sub = W[["original"] + cols].dropna()
            d = sub["original"].values[:, None] - sub[cols].values  # items x seeds
            dm = d.mean(axis=1)
            lo, hi = boot_ci(dm)
            ps = [paired_test(task, sub["original"].values, sub[c].values) for c in cols]
            # pooled test: stack (orig, edit) pairs across seeds
            pooled_p = paired_test(task, np.tile(sub["original"].values, len(cols)), sub[cols].values.T.ravel())
            out.append({"task": task, "task_type": TASK_TYPE[task], "edit": e, "n_items": len(sub),
                        "n_seeds": len(cols), "orig_score": base, "edited_score": sub[cols].values.mean(),
                        "delta_pts": 100 * dm.mean(), "ci_lo_pts": 100 * lo, "ci_hi_pts": 100 * hi,
                        "pfi": dm.mean() / base, "pfi_ci_lo": lo / base, "pfi_ci_hi": hi / base,
                        "abs_delta_pts_per_seed": 100 * np.abs(d.mean(axis=0)).mean(),
                        "flip_rate": np.mean([flip_rate(task, sub["original"].values, sub[c].values) for c in cols]),
                        "seed_deltas_pts": ";".join(f"{100 * x:.1f}" for x in d.mean(axis=0)),
                        "p_pooled": pooled_p, "p_min_seed": min(ps)})
    res = pd.DataFrame(out)
    if len(res):
        res["p_holm"] = multipletests(res["p_pooled"], method="holm")[1]
    return res


def compounding(df):
    """For each combo@seed: delta(combo) vs sum of delta(single@seed) for its components, per item -> bootstrap."""
    out = []
    for task in TASK_ORDER:
        W = wide(df, task)
        if "original" not in W:
            continue
        for cname, comps in COMBOS.items():
            for c in [c for c in W.columns if c.startswith(cname + "@")]:
                s = c.split("@")[1]
                singles = [f"{e}@{s}" for e in comps]
                if not all(x in W for x in singles):
                    continue
                sub = W[["original", c] + singles].dropna()
                o = sub["original"].values
                d_combo = o - sub[c].values
                d_sum = sum(o - sub[x].values for x in singles)
                inter = d_combo - d_sum
                lo, hi = boot_ci(inter)
                max_single = max(o.mean() - sub[x].values.mean() for x in singles)
                # "Union" null: each edit independently breaks a fixed set of items; the combo breaks an
                # item iff any component breaks it (per-item min score). Unlike the additive sum it cannot
                # double-count an item, so it is the natural no-interaction baseline for bounded scores.
                d_union = o - np.minimum(o, sub[singles].values.min(axis=1))
                inter_u = d_combo - d_union
                lo_u, hi_u = boot_ci(inter_u)
                out.append({"task": task, "task_type": TASK_TYPE[task], "combo": cname, "k": len(comps), "seed": int(s),
                            "delta_combo_pts": 100 * d_combo.mean(), "sum_single_pts": 100 * d_sum.mean(),
                            "max_single_pts": 100 * max_single,
                            "interaction_pts": 100 * inter.mean(), "inter_ci_lo": 100 * lo, "inter_ci_hi": 100 * hi,
                            "ratio": d_combo.mean() / d_sum.mean() if abs(d_sum.mean()) > 1e-9 else np.nan,
                            "union_null_pts": 100 * d_union.mean(), "inter_union_pts": 100 * inter_u.mean(),
                            "inter_union_ci_lo": 100 * lo_u, "inter_union_ci_hi": 100 * hi_u,
                            "flip_combo": flip_rate(task, o, sub[c].values),
                            "flip_sum_singles": sum(flip_rate(task, o, sub[x].values) for x in singles)})
    return pd.DataFrame(out)


def compounding_pooled(df):
    """Per task and combo size k: pool all combos of size k and both seeds; bootstrap items for the
    interaction vs. the additive null and vs. the union null."""
    out = []
    for task in TASK_ORDER:
        W = wide(df, task)
        if "original" not in W:
            continue
        for k in sorted({len(v) for v in COMBOS.values()}):
            dc, ds, du = [], [], []
            for cname, comps in COMBOS.items():
                if len(comps) != k:
                    continue
                for c in [c for c in W.columns if c.startswith(cname + "@")]:
                    singles = [f"{e}@{c.split('@')[1]}" for e in comps]
                    if not all(x in W for x in singles):
                        continue
                    sub = W[["original", c] + singles].dropna()
                    o = sub["original"].values
                    dc.append(pd.Series(o - sub[c].values, index=sub.index))
                    ds.append(pd.Series(sum(o - sub[x].values for x in singles), index=sub.index))
                    du.append(pd.Series(o - np.minimum(o, sub[singles].values.min(axis=1)), index=sub.index))
            if not dc:
                continue
            # average over conditions per item, then bootstrap over items
            C, S, U = (pd.concat(x, axis=1).mean(axis=1) for x in (dc, ds, du))
            ia, iu = (C - S).values, (C - U).values
            la, ha = boot_ci(ia)
            lu, hu = boot_ci(iu)
            out.append({"task": task, "task_type": TASK_TYPE[task], "k": k, "n_conditions": len(dc),
                        "delta_combo_pts": 100 * C.mean(), "additive_null_pts": 100 * S.mean(),
                        "union_null_pts": 100 * U.mean(),
                        "ratio_vs_additive": C.mean() / S.mean() if abs(S.mean()) > 1e-9 else np.nan,
                        "inter_add_pts": 100 * ia.mean(), "inter_add_ci": f"[{100 * la:.1f}, {100 * ha:.1f}]",
                        "inter_union_pts": 100 * iu.mean(), "inter_union_ci": f"[{100 * lu:.1f}, {100 * hu:.1f}]"})
    return pd.DataFrame(out)


def task_pfi_ci(df):
    """Mean PFI over the 12 single-edit conditions per task, with a bootstrap CI over items."""
    out = []
    for task in TASK_ORDER:
        W = wide(df, task)
        if "original" not in W:
            continue
        cols = [c for c in W.columns if c.split("@")[0] in EDIT_TYPES]
        sub = W[["original"] + cols].dropna()
        base = sub["original"].mean()
        d = (sub["original"].values[:, None] - sub[cols].values).mean(axis=1)
        dabs_cond = np.abs((sub["original"].values[:, None] - sub[cols].values).mean(axis=0))
        lo, hi = boot_ci(d)
        allc = [c for c in W.columns if c not in ("original", "original_rerun")]
        sub2 = W[["original"] + allc].dropna()
        d2 = (sub2["original"].values[:, None] - sub2[allc].values).mean(axis=1)
        lo2, hi2 = boot_ci(d2)
        out.append({"task": task, "task_type": TASK_TYPE[task], "orig_score": base, "n_single_conds": len(cols),
                    "mean_pfi_single": d.mean() / base, "ci_lo": lo / base, "ci_hi": hi / base,
                    "mean_abs_pfi_single": dabs_cond.mean() / base,
                    "mean_pfi_all_conds": d2.mean() / base, "ci_lo_all": lo2 / base, "ci_hi_all": hi2 / base,
                    "n_all_conds": len(allc)})
    return pd.DataFrame(out)


def noise_floor(df):
    out = []
    for task in TASK_ORDER:
        W = wide(df, task)
        if "original_rerun" not in W:
            continue
        sub = W[["original", "original_rerun"]].dropna()
        out.append({"task": task, "n": len(sub), "delta_pts": 100 * (sub["original"] - sub["original_rerun"]).mean(),
                    "flip_rate": flip_rate(task, sub["original"].values, sub["original_rerun"].values),
                    "identical_frac": float((sub["original"] == sub["original_rerun"]).mean())})
    return pd.DataFrame(out)


def plots(model, cond, et, comp):
    fig_dir = ROOT / "figures"
    fig_dir.mkdir(exist_ok=True)
    tag = model.replace(".", "_")
    if len(et):
        # Heatmap of delta (points) edit type x task
        pv = et.pivot(index="edit", columns="task", values="delta_pts").reindex(index=EDIT_TYPES)
        pv = pv[[t for t in TASK_ORDER if t in pv]]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        lim = max(1, np.nanmax(np.abs(pv.values)))
        im = ax.imshow(pv.values, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_xticks(range(pv.shape[1]), [f"{t}\n({TASK_TYPE[t][:14]})" for t in pv.columns], fontsize=8)
        ax.set_yticks(range(pv.shape[0]), pv.index)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                ax.text(j, i, f"{pv.values[i, j]:+.1f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, label="score drop vs. original (points; + = worse)")
        ax.set_title(f"Score drop by edit type ({model}, mean of 2 seeds)")
        plt.tight_layout()
        plt.savefig(fig_dir / f"heatmap_delta_{tag}.png", dpi=150)
        plt.close()

        # Bar chart with CIs
        fig, axes = plt.subplots(1, len(pv.columns), figsize=(3.2 * len(pv.columns), 3.6), sharey=False)
        for ax, task in zip(np.atleast_1d(axes), pv.columns):
            sub = et[et.task == task].set_index("edit").reindex(EDIT_TYPES)
            y = sub["delta_pts"].values
            err = np.vstack([y - sub["ci_lo_pts"].values, sub["ci_hi_pts"].values - y])
            ax.bar(range(len(y)), y, yerr=err, color="#4C72B0", capsize=3)
            ax.axhline(0, color="k", lw=0.8)
            ax.axhline(8, color="#C44E52", ls="--", lw=0.8)
            ax.axhline(1, color="#55A868", ls=":", lw=0.8)
            ax.set_xticks(range(len(y)), EDIT_TYPES, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{task} (orig={100 * sub['orig_score'].iloc[0]:.0f})", fontsize=10)
        np.atleast_1d(axes)[0].set_ylabel("drop (points), 95% bootstrap CI")
        fig.suptitle("Score drop per edit type; dashed red = 8-pt H1 threshold, dotted green = 1 pt", fontsize=9)
        plt.tight_layout()
        plt.savefig(fig_dir / f"bars_delta_{tag}.png", dpi=150)
        plt.close()

        # Flip rate heatmap
        pf = et.pivot(index="edit", columns="task", values="flip_rate").reindex(index=EDIT_TYPES)[pv.columns]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        im = ax.imshow(100 * pf.values, cmap="Oranges", vmin=0)
        ax.set_xticks(range(pf.shape[1]), pf.columns)
        ax.set_yticks(range(pf.shape[0]), pf.index)
        for i in range(pf.shape[0]):
            for j in range(pf.shape[1]):
                ax.text(j, i, f"{100 * pf.values[i, j]:.0f}%", ha="center", va="center", fontsize=9)
        plt.colorbar(im, label="% items whose correctness flips")
        ax.set_title(f"Item-level flip rate ({model})")
        plt.tight_layout()
        plt.savefig(fig_dir / f"heatmap_flip_{tag}.png", dpi=150)
        plt.close()

    if len(comp):
        g = comp.groupby(["task", "combo", "k"], as_index=False)[["delta_combo_pts", "sum_single_pts"]].mean()
        fig, ax = plt.subplots(figsize=(5.5, 5))
        markers = {"sst2": "o", "agnews": "s", "arc": "^", "squad": "D"}
        for t, sub in g.groupby("task"):
            ax.scatter(sub["sum_single_pts"], sub["delta_combo_pts"], marker=markers[t], label=t, s=50)
            for _, r in sub.iterrows():
                ax.annotate(r["combo"], (r["sum_single_pts"], r["delta_combo_pts"]), fontsize=6, alpha=0.7)
        lo = min(g["sum_single_pts"].min(), g["delta_combo_pts"].min()) - 2
        hi = max(g["sum_single_pts"].max(), g["delta_combo_pts"].max()) + 2
        xs = np.linspace(lo, hi, 10)
        ax.plot(xs, xs, "k--", lw=1, label="additive (y=x)")
        ax.plot(xs, 1.4 * xs, color="#C44E52", ls=":", lw=1, label="H3: +40% (y=1.4x)")
        ax.set_xlabel("sum of single-edit drops (points)")
        ax.set_ylabel("combined-edit drop (points)")
        ax.set_title(f"Compounding of edits ({model})")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / f"compounding_{tag}.png", dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    df = load(args.model)
    tag = args.model
    cond = per_condition(df)
    et = per_edit_type(df)
    comp = compounding(df)
    rdir = ROOT / "results"
    cond.to_csv(rdir / f"summary_{tag}.csv", index=False)
    et.to_csv(rdir / f"edit_type_{tag}.csv", index=False)
    comp.to_csv(rdir / f"compounding_{tag}.csv", index=False)
    if len(et):
        # Task-type PFI: mean over edit types of the pooled PFI; CI via bootstrap of items within task
        tt = et.groupby(["task", "task_type"], as_index=False).agg(
            mean_pfi=("pfi", "mean"), mean_abs_pfi=("pfi", lambda x: np.abs(x).mean()),
            mean_delta_pts=("delta_pts", "mean"), mean_flip=("flip_rate", "mean"), orig=("orig_score", "first"))
        tt.to_csv(rdir / f"task_type_{tag}.csv", index=False)
        print(tt.to_string(index=False))
    pd.set_option("display.width", 200)
    print(et[["task", "edit", "orig_score", "edited_score", "delta_pts", "ci_lo_pts", "ci_hi_pts", "pfi",
              "flip_rate", "seed_deltas_pts", "p_pooled", "p_holm"]].round(4).to_string(index=False))
    if len(comp):
        print(comp.round(3).to_string(index=False))
    cp = compounding_pooled(df)
    cp.to_csv(rdir / f"compounding_pooled_{tag}.csv", index=False)
    tp = task_pfi_ci(df)
    tp.to_csv(rdir / f"task_pfi_{tag}.csv", index=False)
    nf = noise_floor(df)
    nf.to_csv(rdir / f"noise_floor_{tag}.csv", index=False)
    for name, t in [("POOLED COMPOUNDING", cp), ("TASK PFI", tp), ("NOISE FLOOR", nf)]:
        print(f"\n== {name} ==\n" + t.round(4).to_string(index=False))
    plots(args.model, cond, et, comp)


if __name__ == "__main__":
    main()
